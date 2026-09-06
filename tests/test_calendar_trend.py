"""The division of labour between S3's ``exact`` modifier and ``src.data.calendar``.

The point of this file is the BOUNDARY, not the arithmetic. ``src/data/calendar.py`` exists only
because five of the six ``TIME_ENCODINGS`` are already handled by machinery that ships, and the
sixth is not. If that ever stops being true — upstream changes, or the detector gets smarter — the
module is either redundant or insufficient, and both are silent failures on the submission path.

So the first test asserts the split itself. It is a tripwire in the same family as #58's
``get_activation_fn`` check: it fails loudly if the reason for this module's existence goes away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import TREND_COL, complete_trend, fit_trend
from src.data.features import ID, TIME, TIME_ENCODINGS
from src.data.gap_fill import exact_how_columns

PERIODIC = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


def _frame(n_hours: int = 24 * 21, n_series: int = 3, slope: float = 6.939306e-04):
    """A panel with the real dataset's calendar semantics: periodic encodings + a linear trend."""
    start = pd.Timestamp("2023-01-01")
    ts = pd.date_range(start, periods=n_hours, freq="h")
    rows = []
    for s in range(n_series):
        hours = np.arange(n_hours, dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    ID: f"unit_{s:03d}",
                    TIME: ts,
                    "hour_sin": np.sin(2 * np.pi * ts.hour / 24.0),
                    "hour_cos": np.cos(2 * np.pi * ts.hour / 24.0),
                    "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7.0),
                    "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7.0),
                    "is_weekend": (ts.dayofweek >= 5).astype(float),
                    TREND_COL: 1.2660764 + slope * hours,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_exact_covers_the_five_periodic_encodings_but_NOT_trend():
    """The reason ``src/data/calendar.py`` exists, asserted rather than assumed.

    S3's ``exact`` modifier reconstructs any column that is a deterministic function of
    (series, hour-of-week). Five of the six calendar encodings are; ``trend`` is monotone in
    ABSOLUTE time, so the hour-of-week detector cannot see it and a per-series median would fill
    it with a flat constant — plausible, smooth, and undetectably wrong across a 336h gap.
    """
    df = _frame()
    hits = exact_how_columns(df[TIME_ENCODINGS], df[ID], df[TIME])
    assert set(hits) == set(PERIODIC), (
        f"the exact/trend split moved: detected {sorted(hits)}. If `trend` is now detected this "
        "module is redundant; if a periodic column is not, it is insufficient."
    )
    assert TREND_COL not in hits


def test_trend_extrapolates_across_a_gap_it_never_saw():
    """The submission case: fit on supplied hours, produce values for hours nobody supplied."""
    df = _frame()
    observed = df[df[TIME] < df[TIME].max() - pd.Timedelta(hours=336)]
    target = df.copy()
    target.loc[target[TIME] >= observed[TIME].max(), TREND_COL] = np.nan
    assert target[TREND_COL].isna().any()

    out = complete_trend(target, reference=observed)
    assert not out[TREND_COL].isna().any()
    np.testing.assert_allclose(out[TREND_COL], df[TREND_COL], atol=1e-9)


def test_supplied_values_are_never_overwritten():
    """A row the harness handed us keeps its number; only absent rows are computed.

    The reference is passed explicitly here because the off-line value is deliberately corrupt —
    fitting on it would (correctly) trip the linearity guard, which is the next test.
    """
    clean = _frame()
    target = clean.copy()
    target.loc[0, TREND_COL] = 999.0  # deliberately off the line
    target.loc[5:10, TREND_COL] = np.nan
    out = complete_trend(target, reference=clean)
    assert out.loc[0, TREND_COL] == 999.0, "a supplied value was silently rewritten"
    assert not out[TREND_COL].isna().any()
    # the absent rows came from the line, not from the corrupt neighbour
    np.testing.assert_allclose(out.loc[5:10, TREND_COL], clean.loc[5:10, TREND_COL], atol=1e-9)


def test_a_corrupt_supplied_value_trips_the_guard_when_it_is_its_own_reference():
    """Fail closed: with no clean reference, one off-line row must stop the extrapolation rather
    than drag the fitted line and silently skew every gap hour."""
    target = _frame()
    target.loc[0, TREND_COL] = 999.0
    target.loc[5:10, TREND_COL] = np.nan
    with pytest.raises(ValueError, match="not linear in time"):
        complete_trend(target)


def test_a_nonlinear_trend_RAISES_rather_than_extrapolating():
    """The tripwire. A wrong `trend` over the gap is a smooth column of believable numbers, so it
    must fail loudly — the failure mode this project has been caught by six times."""
    df = _frame()
    df.loc[df.index[: len(df) // 2], TREND_COL] = df[TREND_COL] ** 2 + 3.0
    with pytest.raises(ValueError, match="not linear in time"):
        fit_trend(df[TIME], df[TREND_COL])


def test_fit_trend_recovers_the_real_datasets_slope_shape():
    """Sanity: the fitted line reproduces its own inputs to floating point, as measured on the
    real `validation_input.csv` (max residual 4.441e-16)."""
    df = _frame()
    slope, intercept, origin = fit_trend(df[TIME], df[TREND_COL])
    hours = (df[TIME] - origin).dt.total_seconds() / 3600.0
    np.testing.assert_allclose(slope * hours + intercept, df[TREND_COL], atol=1e-9)
    assert origin == df[TIME].min()


def test_too_few_observed_rows_raises():
    """One row cannot define a line."""
    df = _frame(n_hours=1, n_series=1)
    with pytest.raises(ValueError, match="need >= 2 observed"):
        fit_trend(df[TIME], df[TREND_COL])


def test_a_single_timestamp_across_many_series_raises():
    """Three series at ONE hour is three rows and still no slope — the degenerate case a bare
    row-count check would wave through."""
    df = _frame(n_hours=1, n_series=3)
    assert len(df) == 3
    with pytest.raises(ValueError, match="cannot fit a slope"):
        fit_trend(df[TIME], df[TREND_COL])
