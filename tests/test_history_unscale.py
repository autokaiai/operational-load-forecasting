"""`history_from_bundle` must return RAW units even when the checkpoint was fitted with a scaler.

WHY THIS FILE EXISTS. `NeuralForecast(local_scaler_type=...)` stores the SCALED temporal array and
inverts only inside `predict()`. `history_from_bundle` reads `dataset.temporal` directly, so a
scaled checkpoint silently yields a normalised frame — and that frame is the Chronos cascade
context, the tree's lag features, and the source of `nominal_capacity` for A9's weights. On
2026-09-04 that produced y at mean 0.200 / min -3.119 and `nominal_capacity` exactly 0.000. Only
A9's NaN guard caught it; the Chronos and tree paths would have failed SILENTLY.

The load-bearing assertion is `test_scaled_history_round_trips_to_raw`: it fails on the code that
shipped before the fix, which is the property S9.1a asks of every regression test here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neuralforecast")

from src.models.cascade_inference import history_from_bundle  # noqa: E402


class _Scaler:
    """Stand-in for `coreforecast.scalers.LocalRobustScaler`: y' = (y - c) / s, per series."""

    def __init__(self, centre: float, scale: float):
        self.centre, self.scale = centre, scale

    def inverse_transform(self, ga):
        return np.asarray(ga.data if hasattr(ga, "data") else ga) * self.scale + self.centre


class _DS:
    def __init__(self, arr, cols, indptr):
        self.temporal, self.temporal_cols, self.indptr = arr, cols, indptr


class _NF:
    def __init__(self, arr, cols, indptr, uids, last, scalers):
        self.dataset = _DS(arr, cols, indptr)
        self.uids, self.last_dates, self.freq = uids, last, "h"
        self.scalers_ = scalers


def _bundle(scalers):
    n = 6
    raw = np.array(
        [[10.0, 70.0], [12.0, 70.0], [11.0, 70.0], [20.0, 80.0], [22.0, 80.0], [21.0, 80.0]]
    )
    stored = raw.copy()
    if scalers:  # emulate what NeuralForecast stores: the SCALED array
        stored[:, 0] = (raw[:, 0] - 15.0) / 5.0
        stored[:, 1] = (raw[:, 1] - 75.0) / 5.0
    return _NF(
        stored,
        ["y", "nominal_capacity"],
        np.array([0, 3, n]),
        ["unit_000", "unit_001"],
        [pd.Timestamp("2023-01-01 02:00"), pd.Timestamp("2023-01-01 02:00")],
        scalers,
    ), raw


def test_scaled_history_round_trips_to_raw():
    """THE REGRESSION. Without the inverse transform this returns the normalised array and both
    assertions below fail — `nominal_capacity` most visibly, since a per-series constant scales to
    exactly 0.0 and A9 then divides 0/0."""
    sc = {"y": _Scaler(15.0, 5.0), "nominal_capacity": _Scaler(75.0, 5.0)}
    nf, raw = _bundle(sc)
    h = history_from_bundle(nf)
    assert np.allclose(h["y"].to_numpy(), raw[:, 0])
    assert np.allclose(h["nominal_capacity"].to_numpy(), raw[:, 1])
    assert h["nominal_capacity"].min() > 0, "a zeroed capacity makes A9's weights 0/0"


def test_unscaled_checkpoint_is_untouched():
    """Every checkpoint shipped before the fix has an empty `scalers_`; those must pass through
    byte-identical, so this change cannot alter a previously-produced submission."""
    nf, raw = _bundle({})
    h = history_from_bundle(nf)
    assert np.allclose(h["y"].to_numpy(), raw[:, 0])
    assert np.allclose(h["nominal_capacity"].to_numpy(), raw[:, 1])


def test_a_column_without_a_scaler_is_left_alone():
    """`scalers_` covers only the columns NeuralForecast scaled; nothing else may be touched."""
    nf, raw = _bundle({"y": _Scaler(15.0, 5.0)})
    h = history_from_bundle(nf)
    assert np.allclose(h["y"].to_numpy(), raw[:, 0])
    # nominal_capacity was stored scaled but has no scaler -> stays as stored, not invented
    assert np.allclose(h["nominal_capacity"].to_numpy(), (raw[:, 1] - 75.0) / 5.0)
