"""Tests for src.metrics — the module every recorded number in this project depends on.

It had none until now. The three things that actually matter here:

  * ``wape`` is *exactly* ``sum|y-yhat| / sum|y|`` — not a mean of per-row ratios, not a mean of
    per-series WAPEs. It is volume-weighted by construction, which is why the loss we train with
    has to be volume-weighted too.
  * the zero-denominator branch returns ``nan`` rather than raising or silently yielding ``inf``.
  * **pooled WAPE != mean-of-folds WAPE.** Both conventions appear in ``results/*.json`` and they
    are not interchangeable; a test pins the difference so nobody quotes one as the other.

``wape`` returns unit scale (0.1463). The leaderboard reports ``100 x`` that (13.2 ~ 100 x 0.132).
That conversion is asserted explicitly below rather than being folded into ``wape`` itself —
changing the base unit would silently invalidate every number already on record.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import all_metrics, mae, mape, mse, rmse, smape, wape

LEADERBOARD_SCALE = 100.0


def test_wape_matches_hand_calculated_fixture():
    """sum|y-yhat| / sum|y| on numbers small enough to verify by hand."""
    y = [10.0, 20.0, 30.0, 40.0]
    yhat = [12.0, 18.0, 33.0, 39.0]
    # |errors| = 2 + 2 + 3 + 1 = 8 ; sum|y| = 100
    assert wape(y, yhat) == pytest.approx(8.0 / 100.0)


def test_wape_is_volume_weighted_not_a_mean_of_ratios():
    """The high-volume row dominates. A mean of per-row ratios would give a very different number.

    This is the whole reason the graded metric needs a volume-weighted training loss: equal-weight
    L1 optimises the mean-of-ratios quantity, which is not what we are scored on.
    """
    y = [1.0, 1000.0]
    yhat = [2.0, 1100.0]  # 100% error on the small row, 10% on the big one
    mean_of_ratios = np.mean([1.0 / 1.0, 100.0 / 1000.0])  # 0.55
    assert wape(y, yhat) == pytest.approx(101.0 / 1001.0)  # ~0.1009
    assert wape(y, yhat) != pytest.approx(mean_of_ratios)


def test_wape_zero_denominator_returns_nan():
    """All-zero actuals: undefined, must be nan — not inf, not 0.0, not an exception."""
    result = wape([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert np.isnan(result)


def test_wape_perfect_prediction_is_zero():
    y = [5.0, 7.5, 100.0]
    assert wape(y, y) == pytest.approx(0.0)


def test_wape_ignores_error_sign():
    """WAPE is built on absolute error, so over- and under-prediction cost the same."""
    y = [10.0, 10.0]
    assert wape(y, [12.0, 8.0]) == pytest.approx(wape(y, [8.0, 12.0]))


def test_pooled_wape_differs_from_mean_of_folds_on_skewed_data():
    """The convention pin. Two folds, wildly different volumes -> the two numbers diverge.

    Pooling accumulates numerators and denominators across folds *before* dividing, so a
    high-volume fold dominates. Mean-of-folds gives every fold equal say regardless of volume.
    ``results/*.json`` contains both; this test exists so they are never quoted interchangeably.
    """
    # fold A: small volume, poor accuracy. fold B: large volume, good accuracy.
    y_a, yhat_a = np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])  # wape 1.0
    y_b, yhat_b = np.array([1000.0, 1000.0]), np.array([1010.0, 1010.0])  # wape 0.01

    wape_a, wape_b = wape(y_a, yhat_a), wape(y_b, yhat_b)
    assert wape_a == pytest.approx(1.0)
    assert wape_b == pytest.approx(0.01)

    mean_of_folds = float(np.mean([wape_a, wape_b]))  # 0.505 — fold A gets half the say
    pooled = wape(np.concatenate([y_a, y_b]), np.concatenate([yhat_a, yhat_b]))
    # pooled = (3 + 20) / (3 + 2000) = 23/2003 ~ 0.01148 — fold B's volume dominates
    assert pooled == pytest.approx(23.0 / 2003.0)

    assert pooled != pytest.approx(mean_of_folds)
    assert mean_of_folds > 40 * pooled  # not a rounding difference; a different quantity


def test_pooled_equals_mean_of_folds_only_when_denominators_match():
    """The one case where the conventions agree — equal volume per fold. Bounds the claim above."""
    y_a, yhat_a = np.array([10.0, 10.0]), np.array([11.0, 11.0])
    y_b, yhat_b = np.array([10.0, 10.0]), np.array([13.0, 13.0])
    mean_of_folds = float(np.mean([wape(y_a, yhat_a), wape(y_b, yhat_b)]))
    pooled = wape(np.concatenate([y_a, y_b]), np.concatenate([yhat_a, yhat_b]))
    assert pooled == pytest.approx(mean_of_folds)


def test_leaderboard_form_is_exactly_100x_wape():
    """The HF Space reports 100 x WAPE. Pin the conversion instead of baking it into wape()."""
    rng = np.random.default_rng(42)
    y = rng.uniform(0.5, 50.0, size=500)
    yhat = y + rng.normal(0.0, 2.0, size=500)
    assert LEADERBOARD_SCALE * wape(y, yhat) == pytest.approx(100.0 * wape(y, yhat))
    # sanity: a realistic WAPE lands near the leaderboard's observed magnitude
    assert 0.0 < wape(y, yhat) < 1.0


def test_wape_is_scale_invariant():
    """Multiplying actuals and predictions by a constant leaves WAPE unchanged."""
    y = np.array([3.0, 8.0, 21.0])
    yhat = np.array([4.0, 7.0, 25.0])
    assert wape(10 * y, 10 * yhat) == pytest.approx(wape(y, yhat))


def test_wape_accepts_lists_series_and_arrays_alike():
    import pandas as pd

    y, yhat = [1.0, 2.0, 3.0], [1.5, 2.5, 2.0]
    expected = wape(y, yhat)
    assert wape(np.array(y), np.array(yhat)) == pytest.approx(expected)
    assert wape(pd.Series(y), pd.Series(yhat)) == pytest.approx(expected)


# --------------------------------------------------------------------------- the other five


def test_mae_mse_rmse_hand_calculated():
    y = [1.0, 2.0, 3.0]
    yhat = [2.0, 4.0, 6.0]  # errors 1, 2, 3
    assert mae(y, yhat) == pytest.approx(6.0 / 3.0)
    assert mse(y, yhat) == pytest.approx((1 + 4 + 9) / 3.0)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(14.0 / 3.0))


def test_rmse_is_sqrt_of_mse():
    rng = np.random.default_rng(0)
    y, yhat = rng.uniform(1, 100, 50), rng.uniform(1, 100, 50)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(mse(y, yhat)))


def test_mape_is_a_mean_of_ratios_and_so_is_not_wape():
    """MAPE weights every row equally; WAPE weights by volume. On skewed data they diverge hard."""
    y, yhat = [1.0, 1000.0], [2.0, 1100.0]
    assert mape(y, yhat) == pytest.approx(np.mean([1.0, 0.1]))
    assert mape(y, yhat) != pytest.approx(wape(y, yhat))


def test_smape_is_bounded_by_two():
    """Symmetric MAPE lives in [0, 2]; the worst case is prediction 0 against a positive actual."""
    assert smape([10.0], [0.0]) == pytest.approx(2.0)
    assert smape([10.0], [10.0]) == pytest.approx(0.0)
    rng = np.random.default_rng(7)
    y, yhat = rng.uniform(0.1, 50, 200), rng.uniform(0.1, 50, 200)
    assert 0.0 <= smape(y, yhat) <= 2.0


def test_all_metrics_returns_all_six_with_wape_first():
    """Issue #55 requires every run to report all six, never WAPE alone."""
    y, yhat = [1.0, 2.0, 3.0], [1.1, 2.2, 2.7]
    out = all_metrics(y, yhat)
    assert list(out) == ["wape", "mae", "mse", "rmse", "mape", "smape"]
    assert out["wape"] == pytest.approx(wape(y, yhat))
    assert out["rmse"] == pytest.approx(rmse(y, yhat))
    assert all(isinstance(v, float) for v in out.values())
