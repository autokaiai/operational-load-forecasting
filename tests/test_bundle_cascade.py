"""Guards for the ship path's two silent-failure modes: a lying sidecar and a missing channel.

Both are the same class of defect the project keeps meeting — an artifact that describes itself
incorrectly, with no symptom until something downstream fails for an unrelated-looking reason.
The shipped `cascade_bag5.pt` records 29 future covariates while its models require 31, and the
first thing that noticed was `nf.predict` refusing to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import bundle
from src.data.features import MISSING_SUFFIX
from src.data.loader import NF_ID, NF_TIME
from src.models import cascade_inference as ci


def test_cascade_channels_are_derived_from_the_futr_list_not_declared():
    """The sidecar has no `cascade` key, so the channels must come from the model's own columns."""
    assert ci.CHRONOS_COL in bundle.CASCADE_FORECASTS_ALL
    base = ["hour_sin", "trend", "demand_forecast"]
    assert bundle.cascade_channels_of(base) == []
    assert bundle.cascade_channels_of([*base, ci.CHRONOS_COL]) == [ci.CHRONOS_COL]


def test_model_futr_exog_refuses_a_bag_whose_models_disagree():
    """A bag is one model five times; disagreement is corruption, not something to reconcile."""

    class _M:
        def __init__(self, cols):
            self.futr_exog_list = cols

    class _NF:
        def __init__(self, models):
            self.models = models

    assert bundle.model_futr_exog(_NF([_M(["a", "b"]), _M(["a", "b"])])) == ["a", "b"]
    with pytest.raises(ValueError, match="disagree"):
        bundle.model_futr_exog(_NF([_M(["a", "b"]), _M(["a"])]))


def _futr(n_series=3, h=4):
    ids = [f"unit_{i:03d}" for i in range(n_series)]
    ts = pd.date_range("2023-01-01", periods=h, freq="h")
    return pd.DataFrame(
        {
            NF_ID: np.repeat(ids, h),
            NF_TIME: np.tile(ts, n_series),
            "demand_forecast": np.arange(n_series * h, dtype=float),
        }
    )


def test_attach_channel_real_values_clear_the_missing_flag():
    futr = _futr()
    values = futr[[NF_ID, NF_TIME]].copy()
    values[ci.CHRONOS_COL] = 7.5

    out = ci.attach_channel(futr, values, fill_stats={}, channel=ci.CHRONOS_COL)

    assert (out[ci.CHRONOS_COL] == 7.5).all()
    assert (out[ci.CHRONOS_COL + MISSING_SUFFIX] == 0.0).all()
    assert len(out) == len(futr)


def test_attach_channel_degrades_to_the_stored_median_and_says_so():
    """The offline arm: per-series median + missing=1, which is the model's own warm-up state."""
    futr = _futr()
    stats = {ci.CHRONOS_COL: {"unit_000": 1.0, "unit_001": 2.0, "unit_002": 3.0}}

    with pytest.warns(UserWarning, match="OFFLINE arm"):
        out = ci.attach_channel(futr, None, fill_stats=stats, channel=ci.CHRONOS_COL)

    assert (out[ci.CHRONOS_COL + MISSING_SUFFIX] == 1.0).all()
    got = out.groupby(NF_ID)[ci.CHRONOS_COL].first().to_dict()
    assert got == {"unit_000": 1.0, "unit_001": 2.0, "unit_002": 3.0}


def test_attach_channel_refuses_a_forecast_that_misses_rows():
    """A short generated block must raise, never be silently back-filled to a plausible number."""
    futr = _futr()
    values = futr[[NF_ID, NF_TIME]].iloc[:5].copy()
    values[ci.CHRONOS_COL] = 1.0

    with pytest.raises(ValueError, match="got no chronos2_forecast"):
        ci.attach_channel(futr, values, fill_stats={}, channel=ci.CHRONOS_COL)


def test_degrading_without_a_stored_statistic_raises_rather_than_inventing_one():
    with pytest.raises(ValueError, match="no fill statistic"):
        ci.attach_channel(_futr(), None, fill_stats={}, channel=ci.CHRONOS_COL)


def test_the_pinned_revision_is_a_full_commit_sha():
    """A tag or a branch would let the covariate worth +8.3% move under us with no symptom."""
    assert len(ci.PINNED_REVISION) == 40
    assert set(ci.PINNED_REVISION) <= set("0123456789abcdef")
