"""Plan 4.2a — forecast-hour-anchored lags, and the boundary that makes them legal.

The existing ``LAGS`` are origin-anchored: ``y[o-(l-1)]``, one value per (series, origin), constant
across all 672 horizon hours. A weekly lag is a different animal — ``y[fc-l]``, which varies with
the forecast hour and therefore lands on the target's own hour-of-week.

Everything here turns on one inequality. ``fc - l = o + k - l`` is at or before the origin for every
``k <= horizon`` **iff** ``l >= horizon``. So 672 is not a tasteful choice, it is the minimum; 336
would read ``y[o+336]`` at the far end of the horizon. Nothing about that failure is loud — the
index is in range and the value is a real number — so it is checked rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.models.lgbm as L
from src.data.features import futr_exog_list, stat_exog_list

HORIZON = L.GAPPED_H  # 672
N = 2400


@pytest.fixture
def long_df():
    """Two series with a y that encodes its own hour index, so a lag is checkable by eye."""
    futr, stat = futr_exog_list(), stat_exog_list()
    frames = []
    for i, sid in enumerate(["u0", "u1"]):
        g = pd.DataFrame(
            {
                "unique_id": sid,
                "ds": pd.date_range("2023-01-01", periods=N, freq="h"),
                # y[t] = t + 1000*i — unique per (series, hour), so any lag is identifiable.
                "y": np.arange(N, dtype=float) + 1000.0 * i,
                "_hidx": np.arange(N),
            }
        )
        for c in futr:
            g[c] = 0.0
        for c in stat:
            g[c] = float(i)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _pairs(o, ks, sid="u0"):
    return pd.DataFrame({"unique_id": sid, "o": o, "k": list(ks)})


# --------------------------------------------------------------------------- the boundary


@pytest.mark.parametrize("lag", [1, 168, 336, 504, 671])
def test_a_lag_shorter_than_the_horizon_is_rejected(lag):
    with pytest.raises(ValueError, match="shorter than the horizon"):
        L.validate_fc_lags([lag])


@pytest.mark.parametrize("lag", L.WEEKLY_LAGS)
def test_the_documented_weekly_lags_are_all_legal(lag):
    assert L.validate_fc_lags([lag]) == [lag]
    assert lag % 168 == 0, "an hour-of-week-aligned lag must be a whole number of weeks"


def test_672_is_exactly_the_boundary_and_is_allowed():
    """At k=672 it reads y[o] — the origin itself, the last observed hour. Legal, barely."""
    assert L.validate_fc_lags([HORIZON]) == [HORIZON]
    with pytest.raises(ValueError):
        L.validate_fc_lags([HORIZON - 1])


def test_duplicate_lags_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        L.validate_fc_lags([672, 672])


def test_the_boundary_follows_a_shorter_horizon():
    """The rule is relative to the horizon in play, not to the constant 672."""
    assert L.validate_fc_lags([336], horizon=336) == [336]


# --------------------------------------------------------------------------- the values


def test_the_weekly_lag_reads_the_forecast_hour_minus_the_lag(long_df):
    o = 1500
    ks = [1, 337, 672]
    X, _ = L.build_design(long_df, _pairs(o, ks), fc_lags=[672])
    got = X["wlag_672"].to_numpy()
    assert list(got) == [float(o + k - 672) for k in ks]


def test_the_weekly_lag_never_reads_past_the_origin(long_df):
    """The leakage property itself, checked on values rather than argued from the index."""
    o = 1500
    X, _ = L.build_design(long_df, _pairs(o, range(1, HORIZON + 1)), fc_lags=L.WEEKLY_LAGS)
    for lag in L.WEEKLY_LAGS:
        vals = X[f"wlag_{lag}"].dropna().to_numpy()
        assert vals.max() <= o, f"wlag_{lag} read y[{vals.max():.0f}] from origin {o}"


def test_it_varies_across_the_horizon_where_the_origin_anchored_lag_does_not(long_df):
    """The whole point: one number per origin vs one number per forecast hour."""
    X, _ = L.build_design(long_df, _pairs(1500, range(1, 673)), fc_lags=[672])
    assert X["lag_168"].nunique() == 1, "origin-anchored lags are constant across the horizon"
    assert X["wlag_672"].nunique() == 672


def test_the_series_start_boundary_leaves_nan_not_a_wrapped_index(long_df):
    """`fc - lag < 0` must produce NaN, mirroring the origin-anchored guard — never y[-n]."""
    X, _ = L.build_design(long_df, _pairs(100, [1, 2]), fc_lags=[1008])
    assert X["wlag_1008"].isna().all()


def test_each_lag_lands_on_the_same_hour_of_week_as_its_target(long_df):
    o = 1500
    X, _ = L.build_design(long_df, _pairs(o, [400, 500, 672]), fc_lags=[840])
    fc = np.array([o + k for k in (400, 500, 672)])
    assert ((fc - X["wlag_840"].to_numpy()) % 168 == 0).all()


# --------------------------------------------------------------------------- wiring


def test_the_columns_appear_in_order_and_only_when_asked(long_df):
    futr, stat = futr_exog_list(), stat_exog_list()
    assert not any(c.startswith("wlag_") for c in L.feature_columns(futr, stat))
    cols = L.feature_columns(futr, stat, L.WEEKLY_LAGS)
    assert cols[len(L.LAGS) : len(L.LAGS) + 3] == ["wlag_672", "wlag_840", "wlag_1008"]


def test_the_default_design_is_byte_identical_to_before(long_df):
    """The recorded 0.1628 must not move because this landed."""
    p = _pairs(1500, range(1, 20))
    a, ya = L.build_design(long_df, p)
    b, yb = L.build_design(long_df, p, fc_lags=[])
    pd.testing.assert_frame_equal(a, b)
    assert np.array_equal(ya, yb, equal_nan=True)


def test_adding_weekly_lags_widens_the_matrix_without_disturbing_the_other_columns(long_df):
    """Column offsets are computed by hand in build_design; a stale one silently shifts features."""
    p = _pairs(1500, range(1, 20))
    base, _ = L.build_design(long_df, p)
    wide, _ = L.build_design(long_df, p, fc_lags=L.WEEKLY_LAGS)
    assert len(wide.columns) == len(base.columns) + 3
    for col in base.columns:
        pd.testing.assert_series_equal(wide[col], base[col], check_names=False)


def test_the_member_records_its_null_result_against_the_right_baseline():
    """A negative result is only useful if it stays findable and stays attached to its baseline."""
    from src.models import members as mem

    spec = mem.get_member("lgbm_wlag")
    assert spec.status == "measured", "it has been run; the registry must not still say untested"
    assert "lgbm_es" in spec.note, "the A/B baseline has to be stated where it can be seen"
    assert "NULL RESULT" in spec.note, "a wash must not read as a candidate"


# --------------------------------------------------------------------------- 4.2c: unit as a class


def test_the_unit_column_appears_only_when_asked(long_df):
    futr, stat = futr_exog_list(), stat_exog_list()
    assert L.UNIT_COL not in L.feature_columns(futr, stat)
    assert L.feature_columns(futr, stat, None, True)[-1] == L.UNIT_COL


def test_unit_codes_are_stable_and_dense(long_df):
    codes = L.unit_codes(long_df)
    assert codes == {"u0": 0, "u1": 1}
    assert L.unit_codes(long_df.iloc[::-1]) == codes, "codes must not depend on row order"


def test_each_row_carries_its_own_series_code(long_df):
    pairs = pd.concat([_pairs(1500, [1, 2], "u0"), _pairs(1500, [1, 2], "u1")], ignore_index=True)
    X, _ = L.build_design(long_df, pairs, categorical_unit=True)
    assert X.groupby("unique_id")[L.UNIT_COL].nunique().eq(1).all()
    assert set(X[L.UNIT_COL]) == {0.0, 1.0}


def test_the_unit_column_does_not_displace_the_statics(long_df):
    """`stat` was written into an open-ended slice; adding a trailing column would have eaten it."""
    p = _pairs(1500, range(1, 10))
    base, _ = L.build_design(long_df, p)
    wide, _ = L.build_design(long_df, p, categorical_unit=True)
    for col in stat_exog_list():
        pd.testing.assert_series_equal(wide[col], base[col], check_names=False)
    assert len(wide.columns) == len(base.columns) + 1


def test_the_two_feature_levers_compose(long_df):
    p = _pairs(1500, range(1, 10))
    X, _ = L.build_design(long_df, p, fc_lags=L.WEEKLY_LAGS, categorical_unit=True)
    assert "wlag_1008" in X.columns and L.UNIT_COL in X.columns
    assert X.columns[-1] == L.UNIT_COL


def test_lightgbm_is_told_the_column_is_categorical(long_df, monkeypatch):
    """Left numeric it would split with `<=`, imposing an ordering on 96 unrelated units."""
    import lightgbm as lgb

    seen = {}
    real = lgb.Dataset

    def spy(*a, **kw):
        seen.setdefault("cats", kw.get("categorical_feature"))
        return real(*a, **kw)

    monkeypatch.setattr(lgb, "Dataset", spy)
    X, y = L.build_design(long_df, _pairs(1500, range(1, 60)), categorical_unit=True)
    L.fit_lgbm(X, y, num_boost_round=2)
    assert seen["cats"] == [L.UNIT_COL]


def test_without_the_lever_lightgbm_is_left_on_auto(long_df, monkeypatch):
    import lightgbm as lgb

    seen = {}
    real = lgb.Dataset
    monkeypatch.setattr(
        lgb,
        "Dataset",
        lambda *a, **kw: (seen.setdefault("c", kw.get("categorical_feature")), real(*a, **kw))[1],
    )
    X, y = L.build_design(long_df, _pairs(1500, range(1, 60)))
    L.fit_lgbm(X, y, num_boost_round=2)
    assert seen["c"] == "auto"
