"""The cascade covariate was leaky, and nothing could have caught it. These are the tests that can.

A cascade column is a forecast stored as a covariate. Whether it is honest depends entirely on
which hours its context was allowed to see — a fact the CSV does not carry and the numbers actively
hide, because a leaky covariate scores *better*. So the grid is recorded beside the frame and
checked before any member conditions on it.

The measured case these pin down: ``chronos2_oof`` rolls the origin in 336h blocks, so at cutoff
3648 the scored block [3984, 4320) drew its covariate from a context ending at 3983 — 336 hours past
the cutoff. Every CV cutoff sits on the same grid. See ``src.models.cascade_provenance``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.models import cascade_provenance as cp
from src.models import members as mem

H = 336
HORIZON = 672
N = 4320
# The grid `chronos2_oof.generate_train` actually produced: range(n-H, 0, -H).
ROLLING = [(s, H) for s in range(N - H, 0, -H)]


def _write(tmp_path, blocks, column="chronos2_forecast"):
    csv = tmp_path / "train_chronos.csv"
    csv.write_text("series_id,timestamp,chronos2_forecast\n")
    cp.write_provenance(
        csv,
        column=column,
        blocks=blocks,
        n_hours=N,
        generator="test",
        zero_shot=True,
    )
    return csv


# --------------------------------------------------------------------------- the recorded leak


@pytest.mark.parametrize("cut", [3648, 3312, 2976])
def test_the_rolling_grid_leaks_336_hours_at_every_cv_cutoff(cut):
    """The finding itself. Not one unlucky window — the cutoffs are all ON the block grid."""
    meta = {"blocks": ROLLING, "n_hours": N}
    assert cp.gap_leak_hours(meta, cut, HORIZON) == H


def test_the_leak_is_exactly_the_far_half_of_the_horizon():
    """Why the cascade looked good on far and bad on near, rather than uniformly good.

    Steps 1-336 land in [cut, cut+336), covered by the block starting AT the cutoff, whose context
    ends at cut-1 — honest. Steps 337-672 are covered by the next block, 336h later.
    """
    meta = {"blocks": ROLLING, "n_hours": N}
    cut = 3648
    near_ends = {cp.context_end_for_hour(meta, h) for h in range(cut, cut + H)}
    far_ends = {cp.context_end_for_hour(meta, h) for h in range(cut + H, cut + HORIZON)}
    assert near_ends == {cut - 1}, "the near half is honest"
    assert far_ends == {cut + H - 1}, "the far half saw 336 hours past the cutoff"


def test_a_block_starting_at_the_cutoff_is_honest_however_far_it_reaches():
    """The fix: one 672h block anchored at the cutoff. Long horizon, but context ends at cut-1."""
    honest = [(s, H) for s in range(3648 - H, 0, -H)] + [(3648, HORIZON)]
    assert cp.gap_leak_hours({"blocks": honest, "n_hours": N}, 3648, HORIZON) == 0
    assert cp.context_end_for_hour({"blocks": honest, "n_hours": N}, 4319) == 3647


def test_a_block_starting_one_hour_past_the_cutoff_already_counts():
    meta = {"blocks": [(3648, 1), (3649, HORIZON)], "n_hours": N}
    assert cp.gap_leak_hours(meta, 3648, HORIZON) == 1


def test_blocks_beyond_the_horizon_are_not_this_window_s_problem():
    meta = {"blocks": [(3648, HORIZON), (3648 + HORIZON, H)], "n_hours": N}
    assert cp.gap_leak_hours(meta, 3648, HORIZON) == 0


def test_the_warm_up_prefix_is_uncovered_not_leaky():
    """The first 288 hours have no prior origin: NaN, then imputed and flagged.

    288 = 4320 - 12*336, the remainder the backward roll cannot reach. Degradation, not leakage —
    and it is exactly the prefix the shipped frame's NaN mask shows, which is what made the
    reconstructed grid trustworthy.
    """
    meta = {"blocks": ROLLING, "n_hours": N}
    assert cp.context_end_for_hour(meta, 0) is None
    assert cp.context_end_for_hour(meta, 287) is None
    assert cp.context_end_for_hour(meta, 288) == 287


# --------------------------------------------------------------------------- the sidecar


def test_the_sidecar_sits_next_to_the_frame_and_round_trips(tmp_path):
    csv = _write(tmp_path, ROLLING)
    assert cp.provenance_path(csv).name == "train_chronos.provenance.json"
    meta = cp.read_provenance(csv)
    assert meta["column"] == "chronos2_forecast"
    assert cp._blocks(meta) == sorted(ROLLING)
    assert meta["zero_shot"] is True


def test_the_uniform_block_shorthand_is_still_readable():
    """The first backfill wrote block+starts; readers must not break on either shape."""
    shorthand = {"block": H, "starts": [s for s, _ in ROLLING], "n_hours": N}
    assert cp._blocks(shorthand) == sorted(ROLLING)


def test_blocks_are_stored_sorted_regardless_of_generation_order(tmp_path):
    """The generator rolls backwards; the lookup wants ascending."""
    csv = _write(tmp_path, list(reversed(ROLLING)))
    assert cp._blocks(cp.read_provenance(csv)) == sorted(ROLLING)


# --------------------------------------------------------------------------- fail closed


def test_a_frame_with_no_sidecar_is_refused_rather_than_trusted(tmp_path):
    """'Unknown' has to mean 'no'. An unverified covariate fails in the flattering direction."""
    csv = tmp_path / "mystery.csv"
    csv.write_text("series_id\n")
    with pytest.raises(ValueError, match="no provenance sidecar"):
        cp.check_gap_honest(csv, "chronos2_forecast", 3648, HORIZON)


def test_a_sidecar_describing_a_different_column_is_refused(tmp_path):
    csv = _write(tmp_path, [(3648, HORIZON)], column="lgbm_forecast")
    with pytest.raises(ValueError, match="describes column"):
        cp.check_gap_honest(csv, "chronos2_forecast", 3648, HORIZON)


def test_check_returns_the_metadata_when_the_frame_passes(tmp_path):
    csv = _write(tmp_path, [(3312, H), (3648, HORIZON)])
    meta = cp.check_gap_honest(csv, "chronos2_forecast", 3648, HORIZON)
    assert meta["n_hours"] == N


def test_the_refusal_names_the_regeneration_command(tmp_path):
    csv = _write(tmp_path, ROLLING)
    with pytest.raises(ValueError, match=r"--cut-idx 3648 --horizon 672"):
        cp.check_gap_honest(csv, "chronos2_forecast", 3648, HORIZON)


# --------------------------------------------------------------------------- the real artifact


def test_the_shipped_derived_frame_is_recorded_as_leaky():
    """Guards the backfill: if someone regenerates train_chronos.csv, this must be revisited.

    Skipped rather than failed when the frame is absent — `data/` is not in git (F7).
    """
    meta = cp.read_provenance("data/derived/train_chronos.csv")
    if meta is None:
        pytest.skip("data/derived/train_chronos.csv not rehydrated in this checkout")
    assert meta["reconstructed"] is True, "the grid was inferred, and that has to stay visible"
    assert cp.gap_leak_hours(meta, 3648, HORIZON) == H


def test_run_member_refuses_a_member_pointed_at_the_leaky_rolling_frame(tmp_path, monkeypatch):
    """End to end, on the real shipped frame — the one that produced the void 0.1372.

    ``tft_cascade`` itself now routes to a per-window frame, so this re-points a probe member at
    the rolling frame to keep the original failure covered rather than let the fix hide it.
    """
    if cp.read_provenance("data/derived/train_chronos.csv") is None:
        pytest.skip("data/derived/train_chronos.csv not rehydrated in this checkout")
    monkeypatch.setattr(mem, "_REGISTRY", dict(mem._REGISTRY))
    mem.register(
        mem.MemberSpec(
            name="casc_rolling_probe",
            kind="neural",
            run=lambda ctx: None,
            seedable=True,
            needs_gpu=False,
            cascade=("chronos2_forecast",),
            train_csv="data/derived/train_chronos.csv",
        )
    )
    frame = pd.DataFrame({"unique_id": ["u"], "ds": [0], "y": [1.0], "chronos2_forecast": [1.0]})
    with pytest.raises(ValueError, match="NOT gap-honest"):
        mem.run_member("casc_rolling_probe", mem.RunContext(long_df=frame, cut_idx=3648))


def test_the_guard_fires_before_the_gpu_spend(tmp_path, monkeypatch):
    """The check is worthless if it runs after training. Pin the order."""
    ran = []
    monkeypatch.setattr(mem, "_REGISTRY", dict(mem._REGISTRY))
    csv = _write(tmp_path, ROLLING)
    mem.register(
        mem.MemberSpec(
            name="casc_order_probe",
            kind="neural",
            run=lambda ctx: ran.append(1),
            seedable=True,
            needs_gpu=False,
            cascade=("chronos2_forecast",),
            train_csv=str(csv),
        )
    )
    frame = pd.DataFrame({"unique_id": ["u"], "ds": [0], "y": [1.0], "chronos2_forecast": [1.0]})
    with pytest.raises(ValueError, match="NOT gap-honest"):
        mem.run_member("casc_order_probe", mem.RunContext(long_df=frame, cut_idx=3648))
    assert ran == [], "the runner was invoked despite a leaky covariate"


def test_the_generator_writes_a_sidecar_the_checker_accepts(tmp_path):
    """Writer and reader agree — the sidecar is not write-only."""
    csv = tmp_path / "t.csv"
    csv.write_text("x\n")
    cp.write_provenance(
        csv,
        column="chronos2_forecast",
        blocks=[(3312, H), (3648, HORIZON)],
        n_hours=N,
        generator="src.models.chronos2_oof",
        zero_shot=True,
        note="honest",
    )
    assert json.loads(cp.provenance_path(csv).read_text())["blocks"] == [[3312, H], [3648, HORIZON]]
    cp.check_gap_honest(csv, "chronos2_forecast", 3648, HORIZON)
