"""Numerical guardrails for fit_lgbm under sample weights, L1, and linear_tree.

The linear-leaf solver fits a Hessian-weighted ridge regression per leaf, so ``linear_tree`` is
where skewed sample weights can go ill-conditioned. ``linear_tree`` is the next lever we want for
the tree (trees cannot extrapolate, which is the July-drift risk in #54), so the guardrail lands
before the lever.

Writing these turned up the constraint that actually matters: **``linear_tree=True`` is unusable
with ``objective="regression_l1"``** in LightGBM 4.6.0. It does not fail loudly — it corrupts
in-range predictions, fitting a straight line with ~700x the in-sample MAE of the L2 fit, silently.
Since WAPE makes L1 the aligned objective, the two levers we most want simply do not compose. Both
halves of that are pinned below.

These tests also pin the claim in ``fit_lgbm``'s docstring: for this design matrix the *unweighted*
L1 fit is already the WAPE-aligned one, and MAD-style volume weights change the model without
helping it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.metrics import wape
from src.models.lgbm import fit_lgbm

lgb = pytest.importorskip("lightgbm")

N_SERIES = 12
N_ROWS_PER_SERIES = 200
FAST = {"num_leaves": 7, "min_child_samples": 5, "verbosity": -1, "seed": 42}


def make_panel(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Heterogeneous panel: per-series levels spanning ~3 orders of magnitude, like the real units.

    Returns ``(X, y, series_id)``. The volume spread is the point — it is what makes a weighted
    fit differ from an unweighted one at all.
    """
    rng = np.random.default_rng(seed)
    levels = np.geomspace(0.5, 500.0, N_SERIES)
    frames, ys, sids = [], [], []
    for s, level in enumerate(levels):
        n = N_ROWS_PER_SERIES
        hour = rng.integers(0, 24, n)
        step = rng.integers(1, 337, n)
        lag = level * (1.0 + 0.2 * rng.standard_normal(n))
        trend = np.linspace(0, 1, n)
        y = (
            level * (1.0 + 0.3 * np.sin(hour / 24 * 2 * np.pi))
            + 0.1 * lag
            + rng.normal(0, level * 0.05, n)
        )
        frames.append(
            pd.DataFrame(
                {"hour": hour, "horizon_step": step, "lag_1": lag, "trend": trend, "level": level}
            )
        )
        ys.append(y)
        sids.append(np.full(n, s))
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(ys),
        np.concatenate(sids),
    )


def mad_weights(y: np.ndarray, series_id: np.ndarray) -> np.ndarray:
    """Per-series MAD weights, mean-normalised — the add_volume_sample_weight recipe."""
    w = np.empty_like(y, dtype=float)
    for s in np.unique(series_id):
        m = series_id == s
        w[m] = max(float(np.median(np.abs(y[m] - np.median(y[m])))), 1e-6)
    return w / w.mean()


# --------------------------------------------------------------------------- plumbing


def test_sample_weight_reaches_the_booster_and_changes_the_model():
    """The plumbing must not silently no-op — otherwise every weighted ablation is a lie."""
    X, y, sid = make_panel()
    w = mad_weights(y, sid)
    assert w.std() > 0.1, "fixture is not skewed enough to distinguish weighted from unweighted"

    plain, _ = fit_lgbm(X, y, {**FAST}, num_boost_round=30)
    weighted, _ = fit_lgbm(X, y, {**FAST}, num_boost_round=30, sample_weight=w)

    p_plain = plain.predict(X)
    p_weighted = weighted.predict(X)
    assert not np.allclose(p_plain, p_weighted), "sample_weight had no effect on the fitted model"


def test_uniform_weights_are_equivalent_to_no_weights():
    """A constant weight vector must not change anything — bounds the test above."""
    X, y, _ = make_panel()
    plain, _ = fit_lgbm(X, y, {**FAST}, num_boost_round=25)
    ones, _ = fit_lgbm(X, y, {**FAST}, num_boost_round=25, sample_weight=np.ones(len(y)))
    assert np.allclose(plain.predict(X), ones.predict(X))


def test_mismatched_sample_weight_length_raises():
    X, y, _ = make_panel()
    with pytest.raises(ValueError, match="sample_weight has"):
        fit_lgbm(X, y, {**FAST}, num_boost_round=5, sample_weight=np.ones(len(y) - 1))


# --------------------------------------------------------------------------- numerical guardrails


def test_weighted_linear_tree_fits_without_warnings_or_nan():
    """Skewed weights + linear leaves under L2 — the combination Phase 4 would actually ship.

    (Not L1: see ``test_linear_tree_is_silently_broken_under_l1`` for why that pairing is unusable.)
    The linear-leaf solver runs a weighted ridge per leaf, which is where a skewed weight vector
    can go ill-conditioned.
    """
    X, y, sid = make_panel()
    w = mad_weights(y, sid)
    params = {**FAST, "objective": "regression", "linear_tree": True, "lambda_l2": 1.0}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        booster, feat = fit_lgbm(X, y, params, num_boost_round=40, sample_weight=w)
        preds = booster.predict(X)

    numeric = [
        str(x.message)
        for x in caught
        if any(
            t in str(x.message).lower() for t in ("nan", "inf", "singular", "diverg", "overflow")
        )
    ]
    assert not numeric, f"numerical instability warnings: {numeric}"
    assert np.all(np.isfinite(preds)), "linear_tree produced non-finite predictions"
    assert feat == list(X.columns)


def test_extreme_weight_spread_still_produces_finite_predictions():
    """Four orders of magnitude between the largest and smallest weight."""
    X, y, sid = make_panel()
    w = np.where(sid < N_SERIES // 2, 1e-3, 10.0)
    params = {**FAST, "objective": "regression", "linear_tree": True}
    booster, _ = fit_lgbm(X, y, params, num_boost_round=30, sample_weight=w)
    assert np.all(np.isfinite(booster.predict(X)))


def test_zero_weight_rows_are_ignored_not_fatal():
    """Half the rows carry zero weight — LightGBM must train on the rest rather than fail."""
    X, y, sid = make_panel()
    w = (sid % 2).astype(float)
    booster, _ = fit_lgbm(
        X, y, {**FAST, "objective": "regression_l1"}, num_boost_round=20, sample_weight=w
    )
    assert np.all(np.isfinite(booster.predict(X)))


# DEFAULT_PARAMS uses learning_rate 0.03, which needs many rounds to converge. The ramp tests are
# about the shape of the fit, not about tuning, so they use a faster rate and enough rounds to
# actually reach the target.
RAMP = {**FAST, "learning_rate": 0.15}
RAMP_ROUNDS = 200


def _ramp_fixture():
    """y = 3x on [0, 100], plus two query points far outside the training range."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 600)
    X = pd.DataFrame({"x": x, "noise": rng.normal(0, 0.01, 600)})
    y = 3.0 * x + rng.normal(0, 0.1, 600)
    beyond = pd.DataFrame({"x": [200.0, 300.0], "noise": [0.0, 0.0]})
    return X, y, beyond


def test_linear_tree_extrapolates_under_l2_where_plain_trees_flatline():
    """Why linear_tree is a Phase-4 lever at all: constant leaves cannot leave their range.

    Trained on a rising ramp, then asked about x well beyond anything seen. A plain tree returns
    its top-bin constant; linear leaves keep climbing. This is the answer to the macro-drift risk
    in #54 — but only under an L2 objective, see the next test.
    """
    X, y, beyond = _ramp_fixture()

    flat, _ = fit_lgbm(X, y, {**RAMP, "objective": "regression"}, num_boost_round=RAMP_ROUNDS)
    lin, _ = fit_lgbm(
        X, y, {**RAMP, "objective": "regression", "linear_tree": True}, num_boost_round=RAMP_ROUNDS
    )
    p_flat, p_lin = flat.predict(beyond), lin.predict(beyond)

    assert p_flat[0] == pytest.approx(p_flat[1], rel=1e-6), "plain tree should saturate"
    assert p_lin[1] > p_lin[0] * 1.2, "linear_tree should keep extrapolating"
    # and it should be roughly right, not merely increasing: true values are 600 and 900
    assert p_lin[0] == pytest.approx(600.0, rel=0.05)
    assert p_lin[1] == pytest.approx(900.0, rel=0.05)


def test_linear_tree_cannot_be_combined_with_l1():
    """``linear_tree=True`` + ``objective="regression_l1"`` is unusable at every LightGBM we ran.

    **The finding.** Through LightGBM 4.6.0 the combination did not merely fail to extrapolate — it
    corrupted **in-range** predictions, fitting a trivially linear target with ~700x the in-sample
    MAE of the L2 fit, **and LightGBM raised no warning whatsoever**. The linear-leaf solver runs a
    Hessian-weighted least squares per leaf, and L1's gradients (constant magnitude, sign only)
    carry no curvature information for it to use.

    **What changed.** LightGBM 4.7.0 rejects the combination outright:
    ``LightGBMError: Cannot use regression_l1 objective when fitting linear trees``. That is the
    correct fix — the failure moved from silent corruption to a loud refusal — and this test was
    written to notice exactly that, so it now accepts either behaviour and asserts the invariant the
    project actually depends on: **the two levers do not stack.**

    Why the invariant matters: this stack is L1-by-design, because WAPE is a sum of absolute errors
    and ``objective="regression_l1"`` is the aligned choice, while ``linear_tree`` is the standard
    fix for tree non-extrapolation. Anything wanting linear leaves has to accept an L2 objective and
    measure the alignment cost, rather than assume both levers compose.

    If a future LightGBM makes the combination genuinely *work*, this fails loudly and the
    constraint gets re-evaluated instead of being inherited forever.
    """
    X, y, _ = _ramp_fixture()
    l2, _ = fit_lgbm(
        X, y, {**RAMP, "objective": "regression", "linear_tree": True}, num_boost_round=RAMP_ROUNDS
    )
    mae_l2 = float(np.abs(y - l2.predict(X)).mean())
    assert mae_l2 < 1.0, f"L2 + linear_tree should fit a straight line almost exactly, got {mae_l2}"

    try:
        l1, _ = fit_lgbm(
            X,
            y,
            {**RAMP, "objective": "regression_l1", "linear_tree": True},
            num_boost_round=RAMP_ROUNDS,
        )
    except lgb.basic.LightGBMError as exc:  # >= 4.7.0 — refused, which is the better failure
        assert "linear" in str(exc).lower(), f"unexpected LightGBM error: {exc}"
        return

    # <= 4.6.0 — accepted, and silently wrong.
    mae_l1 = float(np.abs(y - l1.predict(X)).mean())
    assert mae_l1 > 20 * mae_l2, (
        "L1 + linear_tree appears to work now — LightGBM may have fixed this properly. "
        f"in-sample MAE: L1={mae_l1:.2f} vs L2={mae_l2:.2f}. Re-evaluate the constraint."
    )


# --------------------------------------------------------------------------- alignment claim


def test_unweighted_l1_is_the_wape_aligned_objective_for_the_tree():
    """Pins fit_lgbm's docstring: MAD weights do not improve pooled WAPE here, they cost it.

    Pooled WAPE is ``sum|y-yhat| / sum|y|`` with a model-independent denominator, so unweighted L1
    on the raw target already minimises it. MAD weighting exists to undo neuralforecast's per-series
    scaler; there is no such scaler in this design matrix, so applying it over-weights high-variance
    series beyond their share of the numerator.
    """
    X, y, sid = make_panel()
    w = mad_weights(y, sid)
    params = {**FAST, "objective": "regression_l1"}

    plain, _ = fit_lgbm(X, y, params, num_boost_round=60)
    weighted, _ = fit_lgbm(X, y, params, num_boost_round=60, sample_weight=w)

    assert wape(y, plain.predict(X)) <= wape(y, weighted.predict(X))
