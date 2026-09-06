"""S4 / plan 7.2 — the matched-distribution ablation, and the arithmetic that makes it clean.

`tft_cascade` TRAINS on a Chronos-2 covariate rolled in 336h blocks, so every training row sits at
lead 1-336; at inference the scored block sits at lead 337-672 from the same anchor. The 672-block
grid closes that mismatch. What makes swapping the grid an *ablation* rather than two unrelated
experiments is ``6 * 672 == 12 * 336 == 4032``: both grids start at hour 288, leave an identical NaN
warm-up (hence a bit-identical ``chronos2_forecast_missing`` pattern) and spend the same number of
Chronos steps, so the only thing that moves is the lead profile.

Those are load-bearing coincidences of the numbers, not guarantees of the code, and a change to
``SCORE_LEN`` or to the grid would break them silently — the frames would still generate, the member
would still train, and the A/B would quietly be measuring two things at once. Hence these tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.features import ID, TIME
from src.models import cascade_provenance as cp
from src.models.chronos2_oof import (
    CHRONOS_COL,
    H,
    frame_tag,
    generate_train,
    rehost_gapped,
    stale_hours,
)
from src.models.members import (
    get_member,
    member_train_csv,
    vsn_importance,
    write_vsn_importance,
)

N_HOURS = 4320  # the public train timeline, and the number every property below depends on
MATCHED = 2 * H  # 672 — one gap + one scored block, the lead range the horizon actually spans
CUTS = (3648, 3312, 2976)


def _long(n_series: int = 2, n_hours: int = N_HOURS) -> pd.DataFrame:
    """Minimal neuralforecast-shaped frame — the dry-run backend needs only these three columns."""
    idx = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"unit_{s:03d}",
                    "ds": idx,
                    "y": [float((h % 168) + s) for h in range(n_hours)],
                }
            )
            for s in range(n_series)
        ],
        ignore_index=True,
    )


def _meta(block: int) -> dict:
    """The provenance a rolling run at this block length writes, without spending a run."""
    return {
        "blocks": [(s, block) for s in range(N_HOURS - block, 0, -block)],
        "n_hours": N_HOURS,
    }


# --------------------------------------------------------------------------- the grid itself
def test_generate_train_rolls_at_the_requested_block_length():
    """The grid is aligned to the train END, so the block length sets the starts and the count."""
    _, meta = generate_train(None, _long(), [], 0, True, MATCHED)
    # Generation order, not sorted: the origin is rolled BACKWARDS from the train end, which is
    # what aligns the train blocks with the validation block's k-ahead grid.
    starts = [s for s, _ in meta["blocks"]]
    assert starts == [3648, 2976, 2304, 1632, 960, 288]
    assert all(length == MATCHED for _, length in meta["blocks"])
    assert meta["n_hours"] == N_HOURS


def test_both_grids_leave_an_identical_warmup():
    """THE property that makes 7.2 an ablation: `6*672 == 12*336`, so the NaN prefix is the same.

    If it were not, the two arms would differ in their `chronos2_forecast_missing` pattern as well
    as in the lead profile, and a win could not be attributed to either.
    """
    _, default = generate_train(None, _long(), [], 0, True, H)
    _, matched = generate_train(None, _long(), [], 0, True, MATCHED)

    warmups = {g: min(s for s, _ in m["blocks"]) for g, m in (("336", default), ("672", matched))}
    assert warmups == {"336": 288, "672": 288}
    # ...and neither arm gets more Chronos compute, which would be the other way to confound it.
    assert sum(length for _, length in default["blocks"]) == sum(
        length for _, length in matched["blocks"]
    )


def test_the_matched_grid_covers_the_timeline_without_overlap():
    """Every hour from the warm-up to the end is covered once — no gaps, no double-writes."""
    _, meta = generate_train(None, _long(), [], 0, True, MATCHED)
    covered = [h for s, length in meta["blocks"] for h in range(s, s + length)]
    assert sorted(covered) == list(range(288, N_HOURS))
    assert len(covered) == len(set(covered))


# --------------------------------------------------------------------------- what it costs
@pytest.mark.parametrize("cut", [3648, 2976])
def test_matched_grid_is_already_honest_at_two_of_the_three_cutoffs(cut):
    """The 672 grid lands a block start exactly on these cuts, so they need no Chronos call at all.

    This is the whole reason S4 costs 7 blocks rather than 9, and it is a property of the numbers
    (4320 - k*672 hits 3648 and 2976) rather than of anything we wrote.
    """
    lo, hi = stale_hours(_meta(MATCHED), cut, MATCHED, N_HOURS)
    assert lo >= hi, f"cut {cut} should need no splice on the 672 grid"


def test_matched_grid_still_splices_the_middle_cutoff():
    """Cut 3312 is off the 672 grid, so its scored block is stale and must be recomputed."""
    assert stale_hours(_meta(MATCHED), 3312, MATCHED, N_HOURS) == (3648, 3984)


@pytest.mark.parametrize("cut", CUTS)
def test_the_default_grid_is_stale_at_every_cutoff(cut):
    """The control's grid needs a splice everywhere — that is plan 3.10's leak, still fenced."""
    lo, hi = stale_hours(_meta(H), cut, MATCHED, N_HOURS)
    assert (lo, hi) == (cut + H, min(cut + MATCHED, N_HOURS))


# --------------------------------------------------------------------------- the splice-free path
def test_rehost_gapped_accepts_no_new_block_when_nothing_is_stale(tmp_path):
    """A window that needs no call must produce a frame and a sidecar that still passes the fence.

    And the recorded blocks must be exactly the ones that were generated: fabricating a
    `(cut, horizon)` entry for a call we never made would put a claim in the provenance the frame
    cannot support, which is the failure mode `cascade_provenance` exists to prevent.
    """
    meta = _meta(MATCHED)
    base = pd.DataFrame(
        {
            ID: "unit_000",
            TIME: pd.date_range("2023-01-01", periods=N_HOURS, freq="h"),
            CHRONOS_COL: 1.0,
        }
    )
    out, blocks = rehost_gapped(base, meta, None, 3648, MATCHED, N_HOURS)

    assert blocks == sorted(meta["blocks"])  # every kept block, and NOTHING appended
    assert out[CHRONOS_COL].tolist() == base[CHRONOS_COL].tolist()

    csv = tmp_path / "train_chronos_h672_cut3648.csv"
    out.to_csv(csv, index=False)
    cp.write_provenance(
        csv, column=CHRONOS_COL, blocks=blocks, n_hours=N_HOURS, generator="test", zero_shot=True
    )
    cp.check_gap_honest(csv, CHRONOS_COL, 3648, MATCHED)  # must not raise


def test_rehost_gapped_refuses_a_new_block_it_did_not_need():
    """Fail loudly rather than splice a block over hours that were already honest."""
    base = pd.DataFrame(
        {
            ID: "unit_000",
            TIME: pd.date_range("2023-01-01", periods=N_HOURS, freq="h"),
            CHRONOS_COL: 1.0,
        }
    )
    with pytest.raises(AssertionError, match="needed none"):
        rehost_gapped(base, _meta(MATCHED), base, 3648, MATCHED, N_HOURS)


# --------------------------------------------------------------------------- frame identity
def test_frame_tag_separates_the_arms_from_their_controls():
    """The grid and the fill both live in the NAME, because both are silent when wrong."""
    assert frame_tag() == ""
    assert frame_tag("median", H) == ""
    assert frame_tag("interp", H) == "_interp"
    assert frame_tag("median", MATCHED) == "_h672"
    assert frame_tag("interp", MATCHED) == "_interp_h672"


def test_the_matched_member_reads_a_different_frame_from_its_control():
    """An ablation whose two arms share a training covariate is an ablation of nothing."""
    matched = member_train_csv("tft_cascade_matched", "", 3312)
    control = member_train_csv("tft_cascade", "", 3312)
    assert matched == "data/derived/train_chronos_h672_cut3312.csv"
    assert control == "data/derived/train_chronos_cut3312.csv"
    assert matched != control


def test_the_matched_member_declares_the_cascade_channel():
    """Without the declaration `run_member` never enters the fence, and never checks the sidecar."""
    spec = get_member("tft_cascade_matched")
    assert spec.cascade == ("chronos2_forecast",)
    assert spec.needs_gpu and spec.seedable
    assert "{cut}" in spec.train_csv  # per-window: honest for one cutoff only


# --------------------------------------------------------------------------- S4.3 / S7: the VSN
def test_neuralforecast_tft_still_exposes_feature_importances():
    """TRIPWIRE on an upstream internal, for the reason `get_activation_fn('GELU')` earned one.

    The VSN importance figure the exposé promises, and the measurement that would settle S3's open
    cascade hypothesis, both come from `TFT.feature_importances()` — a method on a vendored model,
    not an API we control. If a version bump renames or drops it, the failure should land here in
    seconds rather than after a GPU fan-out that produces cubes and no figure.
    """
    from neuralforecast.models import TFT

    assert hasattr(TFT, "feature_importances")
    assert hasattr(TFT, "attention_weights")


def test_vsn_importance_is_a_no_op_for_a_model_without_one():
    """LSTM/BiTCN have no variable-selection network, so the export must skip, not crash."""

    class _NoVSN:
        pass

    class _NF:
        models = [_NoVSN()]

    assert vsn_importance(_NF()) == {}


def test_write_vsn_importance_names_one_file_per_table(tmp_path):
    """One CSV per table, slugged — so a run's tables land beside its cube rather than merged."""

    class _Model:
        inference_windows_batch_size = 8

        def feature_importances(self):
            return {"Future variable importance over time": pd.DataFrame({CHRONOS_COL: [0.4, 0.6]})}

    class _NF:
        models = [_Model()]

        def predict(self, futr_df=None):
            return None

    assert write_vsn_importance(_NF(), tmp_path, "tft_cascade") == 1
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["tft_cascade_vsn_future_variable_importance_over_time.csv"]
    assert CHRONOS_COL in pd.read_csv(tmp_path / written[0]).columns
