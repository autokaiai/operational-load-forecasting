"""Forecast accuracy metrics.

WAPE is the project's primary metric (and the reason we train with L1/MAE loss); the rest are
reported alongside it. All take 1-D array-likes of actuals and predictions.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def _arrays(y, yhat) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)


def wape(y, yhat) -> float:
    """Weighted absolute percentage error = sum|y-yhat| / sum|y| (primary metric)."""
    y, yhat = _arrays(y, yhat)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else float("nan")


def mae(y, yhat) -> float:
    y, yhat = _arrays(y, yhat)
    return float(np.abs(y - yhat).mean())


def mse(y, yhat) -> float:
    y, yhat = _arrays(y, yhat)
    return float(((y - yhat) ** 2).mean())


def rmse(y, yhat) -> float:
    return float(np.sqrt(mse(y, yhat)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error (target is strictly positive, so this is well-defined)."""
    y, yhat = _arrays(y, yhat)
    return float((np.abs(y - yhat) / np.maximum(np.abs(y), EPS)).mean())


def smape(y, yhat) -> float:
    """Symmetric MAPE in [0, 2]."""
    y, yhat = _arrays(y, yhat)
    denom = np.maximum(np.abs(y) + np.abs(yhat), EPS)
    return float((2.0 * np.abs(y - yhat) / denom).mean())


def all_metrics(y, yhat) -> dict[str, float]:
    """All six metrics as a dict, WAPE first."""
    return {
        "wape": wape(y, yhat),
        "mae": mae(y, yhat),
        "mse": mse(y, yhat),
        "rmse": rmse(y, yhat),
        "mape": mape(y, yhat),
        "smape": smape(y, yhat),
    }
