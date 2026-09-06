"""Tests for src.eval.protocol — the shared grading rules.

Two jobs here. First, prove the pooling is what it claims (accumulate then divide, per-window
breakdown, regime selection). Second, prove ``validate_prediction_df`` actually *rejects* — a
validator that never fires is worse than none, because it reads like a guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.protocol import (
    CUT_BLK,
    add_block_index,
    compute_pooled_wape,
    error_correlation,
    evaluate_admission,
    metric_report,
    min_windows_required,
    paired_bootstrap_delta,
    select_horizon,
    select_regime,
    validate_prediction_df,
    window_sigma,
)

CUTOFFS = [2976, 3312, 3648]
SERIES = ["unit_000", "unit_001"]
BLOCK_LEN = 336


def make_cube(member: str = "lgbm", *, seed: int = 0, error_scale: float = 1.0) -> pd.DataFrame:
    """A synthetic three-window prediction cube with the schema the protocol expects."""
    rng = np.random.default_rng(seed)
    rows = []
    for cut in CUTOFFS:
        for sid in SERIES:
            y = rng.uniform(5.0, 50.0, BLOCK_LEN)
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": sid,
                        "ds": pd.date_range("2023-01-01", periods=BLOCK_LEN, freq="h"),
                        "cutoff": cut,
                        "y": y,
                        member: y + rng.normal(0.0, error_scale, BLOCK_LEN),
                    }
                )
            )
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["cutoff", "unique_id", "ds"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- pooling


def test_pooled_wape_accumulates_before_dividing():
    """Pooled = sum(ae) / sum(ya) over all windows, not the mean of per-window WAPEs."""
    cube = make_cube()
    rep = compute_pooled_wape(cube, "lgbm", "full")
    manual = float(np.abs(cube["y"] - cube["lgbm"]).sum() / np.abs(cube["y"]).sum())
    assert rep["pooled_wape"] == pytest.approx(manual)
    mean_of_windows = float(np.mean(list(rep["per_window_wape"].values())))
    assert rep["pooled_wape"] != pytest.approx(mean_of_windows, abs=1e-12)


def test_pooled_wape_reports_every_window():
    rep = compute_pooled_wape(make_cube(), "lgbm", "full")
    assert sorted(rep["per_window_wape"]) == CUTOFFS
    assert rep["n_rows"] == len(CUTOFFS) * len(SERIES) * BLOCK_LEN


def test_regime_late_keeps_only_the_final_112_hours_per_window():
    """`late` must select the tail of EACH window's block, not the tail of the concatenation."""
    cube = make_cube()
    late = select_regime(cube, "late")
    assert len(late) == len(CUTOFFS) * len(SERIES) * (BLOCK_LEN - CUT_BLK)
    # every window survives the filter — the bug this guards is all rows coming from one window
    assert sorted(late["cutoff"].unique()) == CUTOFFS
    per_window = late.groupby("cutoff").size()
    assert per_window.nunique() == 1


def test_block_index_restarts_per_window_not_per_series_alone():
    """blk must reset at each cutoff; a single running counter would break `late` selection."""
    blocked = add_block_index(make_cube())
    for (_, _), g in blocked.groupby(["cutoff", "unique_id"]):
        assert g["blk"].min() == 0
        assert g["blk"].max() == BLOCK_LEN - 1


def test_full_and_late_regimes_give_different_answers():
    """If these ever agree, the regime argument does nothing and the record stays ambiguous."""
    cube = make_cube()
    assert compute_pooled_wape(cube, "lgbm", "full")["pooled_wape"] != pytest.approx(
        compute_pooled_wape(cube, "lgbm", "late")["pooled_wape"]
    )


def test_regime_is_required_and_validated():
    with pytest.raises(ValueError, match="regime must be one of"):
        select_regime(make_cube(), "far")  # the ambiguous name that means two things in the repo
    with pytest.raises(TypeError):
        compute_pooled_wape(make_cube(), "lgbm")  # no default: must be stated at the call site


def test_unknown_member_column_raises():
    with pytest.raises(KeyError, match="xgboost"):
        compute_pooled_wape(make_cube(), "xgboost", "full")


def test_window_sigma_is_the_spread_of_per_window_scores():
    rep = compute_pooled_wape(make_cube(), "lgbm", "full")
    assert window_sigma(rep) == pytest.approx(np.std(list(rep["per_window_wape"].values())))


# --------------------------------------------------------------------------- validation


def test_validate_accepts_a_well_formed_frame():
    cube = make_cube()
    assert validate_prediction_df(cube, "lgbm") is cube


@pytest.mark.parametrize("missing", ["cutoff", "y", "unique_id", "ds"])
def test_validate_rejects_missing_required_column(missing):
    cube = make_cube().drop(columns=[missing])
    with pytest.raises(AssertionError, match="missing required column"):
        validate_prediction_df(cube, "lgbm")


def test_validate_rejects_missing_member_column():
    """The exact failure mode of today's artifacts: emitted before `cutoff` existed."""
    cube = make_cube().drop(columns=["lgbm"])
    with pytest.raises(AssertionError, match="missing required column"):
        validate_prediction_df(cube, "lgbm")


def test_validate_rejects_unsorted_rows():
    cube = make_cube().sample(frac=1.0, random_state=1).reset_index(drop=True)
    with pytest.raises(AssertionError, match="not sorted"):
        validate_prediction_df(cube, "lgbm")


@pytest.mark.parametrize("col", ["y", "lgbm"])
def test_validate_rejects_nan(col):
    cube = make_cube()
    cube.loc[5, col] = np.nan
    with pytest.raises(AssertionError, match="NaN in column"):
        validate_prediction_df(cube, "lgbm")


def test_validate_rejects_duplicate_rows():
    cube = make_cube()
    dupe = pd.concat([cube, cube.iloc[[0]]], ignore_index=True).sort_values(
        ["cutoff", "unique_id", "ds"]
    )
    with pytest.raises(AssertionError, match="duplicate"):
        validate_prediction_df(dupe.reset_index(drop=True), "lgbm")


def test_validate_rejects_empty_frame():
    with pytest.raises(AssertionError, match="empty"):
        validate_prediction_df(make_cube().iloc[0:0], "lgbm")


# --------------------------------------------------------------------------- admission


def test_error_correlation_is_one_for_identical_members():
    cube = make_cube()
    cube["clone"] = cube["lgbm"]
    assert error_correlation(cube, "lgbm", "clone") == pytest.approx(1.0)


def test_admission_reports_rather_than_drops():
    """A losing candidate still returns a full report — no member is ever dropped automatically."""
    cube = make_cube(error_scale=1.0)
    cube["weak"] = cube["y"] + np.random.default_rng(3).normal(0, 6.0, len(cube))
    rep = evaluate_admission(
        cube, cube, candidate="weak", baseline="lgbm", regime="full", incumbents=["lgbm"], n_boot=0
    )
    assert rep["checks"]["pooled_improves"]["pass"] is False
    assert set(rep["checks"]) == {
        "pooled_improves",
        "majority_windows_improve",
        "orthogonal",
    }
    assert "no member is dropped automatically" in rep["note"].lower()
    assert rep["checks_missed"]  # it failed something, and said so, and returned anyway


def test_admission_flags_a_genuine_improvement():
    cube = make_cube(error_scale=4.0)
    cube["strong"] = cube["y"] + np.random.default_rng(9).normal(0, 0.5, len(cube))
    rep = evaluate_admission(
        cube,
        cube,
        candidate="strong",
        baseline="lgbm",
        regime="full",
        incumbents=["lgbm"],
        n_boot=0,
    )
    assert rep["checks"]["pooled_improves"]["pass"] is True
    assert rep["checks"]["majority_windows_improve"]["pass"] is True
    assert rep["checks"]["majority_windows_improve"]["windows_total"] == len(CUTOFFS)


def test_admission_catches_a_redundant_candidate():
    """Beats the baseline on score but is a near-copy of an incumbent -> orthogonality must fail."""
    cube = make_cube(error_scale=2.0)
    rng = np.random.default_rng(11)
    cube["twin"] = cube["lgbm"] + rng.normal(0, 0.01, len(cube))
    rep = evaluate_admission(
        cube, cube, candidate="twin", baseline="lgbm", regime="full", incumbents=["lgbm"], n_boot=0
    )
    assert rep["checks"]["orthogonal"]["pass"] is False
    assert rep["checks"]["orthogonal"]["err_corr"]["lgbm"] > 0.95


# --------------------------------------------------------------------------- the 2/3 window rule


@pytest.mark.parametrize(("n", "want"), [(1, 1), (2, 2), (3, 2), (4, 3), (5, 4), (6, 4)])
def test_min_windows_required(n, want):
    """2/3 of the windows, rounded up. Three windows -> two."""
    assert min_windows_required(n) == want


def test_a_tiny_uniform_gain_is_admitted():
    """The requirement: 0.5 p.p. better in every window must pass. No sigma threshold to clear."""
    cube = make_cube(error_scale=3.0)
    base = compute_pooled_wape(cube, "lgbm", "full")["pooled_wape"]
    # shrink every residual slightly -> a small, uniform improvement
    cube["tuned"] = cube["y"] + 0.97 * (cube["lgbm"] - cube["y"])
    rep = evaluate_admission(
        cube, cube, candidate="tuned", baseline="lgbm", regime="full", n_boot=0
    )
    margin = rep["checks"]["pooled_improves"]["margin"]
    assert 0 < margin < 0.01, "fixture should be a small gain, not a landslide"
    assert margin < window_sigma(compute_pooled_wape(cube, "lgbm", "full")), (
        "fixture must sit BELOW the retired cross-window sigma bar — that is the whole point"
    )
    assert rep["checks"]["pooled_improves"]["pass"] is True
    assert rep["checks"]["majority_windows_improve"]["pass"] is True
    assert base > 0


def test_one_big_swing_is_rejected_by_the_majority_rule():
    """The pathology the 2/3 rule exists for: a huge win in one window, worse in the other two."""
    cube = make_cube(error_scale=2.0).copy()
    cube["swing"] = cube["lgbm"]
    first, *rest = CUTOFFS
    win = cube["cutoff"] == first
    cube.loc[win, "swing"] = cube.loc[win, "y"]  # perfect in one window
    for cut in rest:
        m = cube["cutoff"] == cut
        cube.loc[m, "swing"] = cube.loc[m, "y"] + 1.05 * (cube.loc[m, "lgbm"] - cube.loc[m, "y"])

    rep = evaluate_admission(
        cube, cube, candidate="swing", baseline="lgbm", regime="full", n_boot=0
    )
    assert rep["checks"]["pooled_improves"]["pass"] is True, "pooled is carried by the one window"
    assert rep["checks"]["majority_windows_improve"]["pass"] is False
    assert len(rep["checks"]["majority_windows_improve"]["windows_won"]) == 1
    assert rep["checks"]["majority_windows_improve"]["windows_required"] == 2


def test_two_of_three_windows_is_enough():
    """Losing one window is survivable — that is the difference from the old all-3 sign rule."""
    cube = make_cube(error_scale=2.0).copy()
    cube["mixed"] = cube["y"] + 0.9 * (cube["lgbm"] - cube["y"])
    loser = CUTOFFS[0]
    m = cube["cutoff"] == loser
    cube.loc[m, "mixed"] = cube.loc[m, "y"] + 1.1 * (cube.loc[m, "lgbm"] - cube.loc[m, "y"])
    rep = evaluate_admission(
        cube, cube, candidate="mixed", baseline="lgbm", regime="full", n_boot=0
    )
    assert len(rep["checks"]["majority_windows_improve"]["windows_won"]) == 2
    assert rep["checks"]["majority_windows_improve"]["pass"] is True


# --------------------------------------------------------------------------- paired bootstrap


def test_paired_bootstrap_uses_every_row_and_every_block():
    """It resamples blocks WITH REPLACEMENT — it never subsamples. The point estimate is exact."""
    cube = make_cube(error_scale=2.0)
    cube["cand"] = cube["y"] + 0.8 * (cube["lgbm"] - cube["y"])
    out = paired_bootstrap_delta(cube, candidate="cand", baseline="lgbm", regime="full", n_boot=200)
    assert out["n_blocks"] == len(CUTOFFS) * len(SERIES)
    exact = (
        compute_pooled_wape(cube, "lgbm", "full")["pooled_wape"]
        - compute_pooled_wape(cube, "cand", "full")["pooled_wape"]
    )
    assert out["delta"] == pytest.approx(exact), (
        "delta must be the full-sample value, not a resample"
    )
    assert out["se"] > 0


def test_paired_bootstrap_se_is_smaller_than_cross_window_sigma():
    """The claim the whole recalibration rests on: pairing cancels window difficulty."""
    cube = make_cube(error_scale=2.0)
    cube["cand"] = cube["y"] + 0.8 * (cube["lgbm"] - cube["y"])
    out = paired_bootstrap_delta(cube, candidate="cand", baseline="lgbm", regime="full", n_boot=400)
    sigma_w = window_sigma(compute_pooled_wape(cube, "lgbm", "full"))
    assert out["se"] < sigma_w


def test_paired_bootstrap_of_a_member_against_itself_is_zero():
    cube = make_cube()
    cube["clone"] = cube["lgbm"]
    out = paired_bootstrap_delta(cube, candidate="clone", baseline="lgbm", regime="full", n_boot=50)
    assert out["delta"] == pytest.approx(0.0)
    assert out["se"] == pytest.approx(0.0)


def test_bootstrap_is_evidence_not_a_gate():
    """A candidate whose CI spans zero still passes the checks — reported, never gated on.

    The fixture has to be genuinely ambiguous, which means better on some blocks and worse on
    others netting a small positive — not a uniform shrink of every residual, which improves every
    block and so is *not* ambiguous however small it is. That distinction is the whole reason a
    cluster bootstrap says something a point estimate cannot.
    """
    cube = make_cube(error_scale=3.0).copy()
    resid = cube["lgbm"] - cube["y"]
    # per-block scale factors straddling 1.0, mean just under it
    factors = dict(
        zip(
            sorted(cube.groupby(["cutoff", "unique_id"]).groups),
            [0.55, 1.45, 0.70, 1.30, 0.80, 1.14],
            strict=True,
        )
    )
    scale = pd.Series(
        [factors[(c, u)] for c, u in zip(cube["cutoff"], cube["unique_id"], strict=True)],
        index=cube.index,
    )
    cube["tiny"] = cube["y"] + scale * resid
    rep = evaluate_admission(
        cube, cube, candidate="tiny", baseline="lgbm", regime="full", n_boot=600
    )
    lo, hi = rep["evidence"]["paired_bootstrap"]["ci95"]
    assert lo < 0 < hi, "fixture should be statistically indistinguishable from no change"
    assert rep["checks"]["pooled_improves"]["pass"] is True
    assert "pooled_improves" not in rep["checks_missed"], "a wide CI must not fail a check"


# --------------------------------------------------------------------------- NEAR / FAR (#55)


def make_full_cube(member: str = "tft", *, seed: int = 0) -> pd.DataFrame:
    """A 672-step cube with a `step` column — the shape a full-horizon cube carries.

    The far half is deliberately noisier: a 336h covariate-absent gap sits in front of it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cut in CUTOFFS:
        for sid in SERIES:
            step = np.arange(1, 2 * BLOCK_LEN + 1)
            y = rng.uniform(5.0, 50.0, len(step))
            noise = np.where(step <= BLOCK_LEN, 1.0, 3.0)
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": sid,
                        "ds": pd.date_range("2023-01-01", periods=len(step), freq="h"),
                        "cutoff": cut,
                        "step": step,
                        "y": y,
                        member: y + rng.normal(0.0, 1.0, len(step)) * noise,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True).sort_values(["cutoff", "unique_id", "ds"])


def test_near_and_far_split_the_horizon_in_half():
    cube = make_full_cube()
    near, far = select_horizon(cube, "near"), select_horizon(cube, "far")
    assert len(near) == len(far) == len(cube) // 2
    assert near["step"].max() == BLOCK_LEN
    assert far["step"].min() == BLOCK_LEN + 1
    assert len(select_horizon(cube, "all")) == len(cube)


def test_horizon_is_validated_and_needs_a_step_column():
    with pytest.raises(ValueError, match="horizon must be one of"):
        select_horizon(make_full_cube(), "late")  # a regime name, not a horizon name
    with pytest.raises(KeyError, match="step"):
        select_horizon(make_cube(), "far")  # scored-block cube has no `step`


def test_metric_report_returns_all_six_metrics_per_horizon():
    """Issue #55: never WAPE alone, and never without the NEAR/FAR split."""
    rep = metric_report(make_full_cube(), "tft")
    assert set(rep["horizons"]) == {"near", "far", "all"}
    for h in ("near", "far", "all"):
        assert list(rep["horizons"][h]["pooled"]) == ["wape", "mae", "mse", "rmse", "mape", "smape"]
        assert sorted(rep["horizons"][h]["per_window"]) == CUTOFFS


def test_metric_report_distinguishes_the_two_halves():
    """If near and far ever agree, the split is doing nothing and the record stays FAR-only."""
    rep = metric_report(make_full_cube(), "tft")["horizons"]
    assert rep["far"]["pooled"]["wape"] > rep["near"]["pooled"]["wape"]
    assert rep["all"]["n_rows"] == rep["near"]["n_rows"] + rep["far"]["n_rows"]


def test_metric_report_rejects_an_unknown_member():
    with pytest.raises(KeyError, match="xgboost"):
        metric_report(make_full_cube(), "xgboost")


def test_evidence_reports_window_spread_without_gating_on_it():
    cube = make_cube(error_scale=2.0)
    cube["cand"] = cube["y"] + 0.9 * (cube["lgbm"] - cube["y"])
    rep = evaluate_admission(cube, cube, candidate="cand", baseline="lgbm", regime="full", n_boot=0)
    assert rep["evidence"]["per_window_delta_spread"] >= 0
    assert "baseline_window_sigma" in rep["evidence"]
    assert "window_sigma" not in str(rep["checks"]), "sigma must not be a gate any more"
