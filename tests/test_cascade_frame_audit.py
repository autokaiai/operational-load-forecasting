"""A gap-honest frame can still be worthless to train on, and nothing in the stack says so.

``check_gap_honest`` is one-sided by design: it asks whether the covariate's context ran PAST the
cutoff, because that is the direction that flatters the score. S2 Stage 2 introduced the opposite
failure — a frame that is impeccably honest and simply **empty** over ``[0, cut)``, which is where
the TFT trains.

That is not hypothetical. The Stage 1 ``_cov`` frames are horizon-only (one block, ``[cut,
cut+672)``) because the screen never trained anything. Measured on the real
``train_tabpfn_ts_cov_cut3648.csv`` through ``members._load_window_long``: 350,208 of 350,208 train
rows NaN, and the ``*_missing`` flag 1.000 over the train region against 0.000 over the horizon.
``run_member``'s own assertion passes — the column is *present* — and the fill cannot rescue it,
because it is fitted on ``raw[hidx < cut]`` where there is no median to take. Nothing raises, and
4-8 GPU-hours go into learning a gate on a constant.

So these pin the coverage half of the contract, against the sidecar rather than against a threshold.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.cascade_frame_audit import audit_one
from src.models import cascade_provenance as cp
from src.models import members as mem
from src.models.members import MemberSpec

CHANNEL = "tabpfn_ts_cov_forecast"
N = 100
CUT = 40
HZ = 20
BLOCK = 10
# Rolled back from the cutoff, exactly as `foundation.generate` builds it: starts at 30, 20, 10,
# so the first 10 hours have no prior origin and stay NaN. Plus the horizon block at the cutoff.
TRAIN_BLOCKS = [(s, BLOCK) for s in range(CUT - BLOCK, 0, -BLOCK)]
FULL_GRID = [*TRAIN_BLOCKS, (CUT, HZ)]


@pytest.fixture
def registry_sandbox(monkeypatch):
    monkeypatch.setattr(mem, "_REGISTRY", dict(mem._REGISTRY))
    return mem._REGISTRY


def _frame(tmp_path, blocks, covered_from: int | None, *, horizon_gap: int = 0, name="f.csv"):
    """Write a 2-series frame whose NaN pattern is set independently of its sidecar.

    Decoupling the two is the point: the audit's job is to notice when the artifact and the
    provenance disagree, so a helper that derived one from the other could not express the failure.
    """
    rows = []
    for sid in ("unit_000", "unit_001"):
        for h in range(N):
            covered = covered_from is not None and h >= covered_from
            if covered and CUT <= h < CUT + horizon_gap:
                covered = False  # punch a hole in the graded block
            rows.append({"series_id": sid, "timestamp": h, CHANNEL: 1.0 if covered else None})
    csv = tmp_path / name
    pd.DataFrame(rows).to_csv(csv, index=False)
    cp.write_provenance(
        csv, column=CHANNEL, blocks=blocks, n_hours=N, generator="test", zero_shot=True
    )
    return csv


def _member(registry, tmp_path, csv, name="probe"):
    registry.pop(name, None)
    mem.register(
        MemberSpec(
            name=name,
            kind="neural",
            run=lambda ctx: None,
            seedable=True,
            needs_gpu=True,
            cascade=(CHANNEL,),
            train_csv=str(tmp_path / csv.name),
            status="untested",
        )
    )
    return name


# --------------------------------------------------------------- the failure that motivated this


def test_a_horizon_only_frame_is_refused(registry_sandbox, tmp_path):
    """The exact shape of every Stage 1 `_cov` frame: honest, complete over the horizon, empty
    where the TFT trains. This is the one the GPU budget depends on catching."""
    csv = _frame(tmp_path, [(CUT, HZ)], covered_from=CUT)
    name = _member(registry_sandbox, tmp_path, csv)

    res = audit_one(name, CUT, derived_root=tmp_path, horizon=HZ)

    assert not res["ok"]
    info = res["channels"][CHANNEL]
    assert info["honest"] is True, "it IS gap-honest — that is precisely why nothing else caught it"
    assert info["horizon_coverage"] == 1.0
    assert info["train_coverage"] == 0.0
    assert "HORIZON-ONLY" in info["verdict"]
    assert "--train-region" in info["verdict"], "the refusal must name the fix"


def test_a_frame_with_a_train_region_is_certified(registry_sandbox, tmp_path):
    """The positive control. Without it, a checker that refuses everything would look correct."""
    csv = _frame(tmp_path, FULL_GRID, covered_from=BLOCK)
    name = _member(registry_sandbox, tmp_path, csv)

    res = audit_one(name, CUT, derived_root=tmp_path, horizon=HZ)

    assert res["ok"], res
    info = res["channels"][CHANNEL]
    assert info["verdict"] == "ok"
    assert info["first_non_nan_hour"] == BLOCK == info["earliest_block_start"]
    assert info["train_coverage"] == pytest.approx((CUT - BLOCK) / CUT)


# ------------------------------------------------------- the artifact must match its own sidecar


def test_a_frame_shorter_than_its_sidecar_claims_is_refused(registry_sandbox, tmp_path):
    """A coverage *threshold* would wave this through — it has plenty of train coverage. What is
    wrong is that the sidecar promises blocks the frame does not carry, which is the same class of
    error as 3.10: the record and the thing diverging with nobody comparing them."""
    csv = _frame(tmp_path, FULL_GRID, covered_from=CUT - BLOCK)  # only the last train block landed

    name = _member(registry_sandbox, tmp_path, csv)
    res = audit_one(name, CUT, derived_root=tmp_path, horizon=HZ)

    assert not res["ok"]
    info = res["channels"][CHANNEL]
    assert info["train_coverage"] > 0.0, "not empty — a fraction-based rule would pass this"
    assert "DISAGREE" in info["verdict"]


def test_a_hole_in_the_graded_block_is_refused(registry_sandbox, tmp_path):
    """The horizon block is what we grade, so partial coverage there is never acceptable."""
    csv = _frame(tmp_path, FULL_GRID, covered_from=BLOCK, horizon_gap=HZ // 2)
    name = _member(registry_sandbox, tmp_path, csv)

    res = audit_one(name, CUT, derived_root=tmp_path, horizon=HZ)

    assert not res["ok"]
    assert res["channels"][CHANNEL]["horizon_coverage"] < 1.0
    assert "INCOMPLETE" in res["channels"][CHANNEL]["verdict"]


# ------------------------------------------------------------- the leakage half still applies


def test_a_leaky_frame_is_still_refused(registry_sandbox, tmp_path):
    """Coverage is an ADDITIONAL gate, not a replacement. A frame rolled on the 3.10 grid has ample
    train coverage and is exactly the thing `check_gap_honest` exists to stop."""
    # The 3.10 shape: the graded window [40, 60) is covered by TWO short blocks, and the second
    # starts at 50 — so its context ran to hour 49, ten hours past the cutoff it is scored at.
    rolling = [(s, BLOCK) for s in range(N - BLOCK, 0, -BLOCK)]
    csv = _frame(tmp_path, rolling, covered_from=0)
    name = _member(registry_sandbox, tmp_path, csv)

    res = audit_one(name, CUT, derived_root=tmp_path, horizon=HZ)

    assert not res["ok"]
    assert res["channels"][CHANNEL]["honest"] is False
    assert "not gap-honest" in res["channels"][CHANNEL]["error"].lower()


def test_a_missing_frame_is_refused_rather_than_skipped(registry_sandbox, tmp_path):
    csv = _frame(tmp_path, FULL_GRID, covered_from=BLOCK)
    name = _member(registry_sandbox, tmp_path, csv)
    csv.unlink()

    res = audit_one(name, CUT, derived_root=tmp_path, horizon=HZ)

    assert not res["ok"]
    assert any("MISSING" in n for n in res["notes"])


# --------------------------------------------------------------------------------- routing


def test_the_declared_channel_reaches_the_futr_list(registry_sandbox, tmp_path):
    """Asserted end-to-end rather than inferred from `register`'s validation.

    `register` already refuses a channel outside `CASCADE_FORECASTS_ALL`, so this cannot currently
    fail — which is the point of checking it anyway. 3.10's cascade looked functional while
    training a plain TFT, and the property that actually matters is "the column is in the
    conditioning set the runner builds", not "the name passed a validator".
    """
    csv = _frame(tmp_path, FULL_GRID, covered_from=BLOCK)
    name = _member(registry_sandbox, tmp_path, csv)

    from src.data.features import cascade_channels, futr_exog_list

    with cascade_channels(*mem.get_member(name).cascade):
        assert CHANNEL in futr_exog_list()
    assert CHANNEL not in futr_exog_list(), "the channel must not leak out of its scope"
