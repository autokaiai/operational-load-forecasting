"""Plan 4.6 — per-unit target normalisation, and the identity that keeps it WAPE-aligned.

The claim being implemented is that normalising and weighting are **separable concerns that
compose**, not a trade-off. With ``w = level``, ``z = y/level`` and ``zhat = yhat/level``::

    w * |z - zhat|  =  level * |y - yhat| / level  =  |y - yhat|

summed over rows, exactly the WAPE numerator. So the division fixes the tree's *representation* —
96 units spanning 0.16 to 53 stop burning early splits on level — while the weight restores the
metric's emphasis, and neither costs the other anything.

That identity is the load-bearing part, so it is tested as arithmetic rather than trusted as
prose. ``(y/level, unweighted)`` is a third, misaligned thing that flatters itself on anything but
pooled WAPE; it is named here so it cannot be mistaken for the normalisation result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.models.lgbm as L
from src.data.features import futr_exog_list, stat_exog_list

N = 1500
CUT = 1200


@pytest.fixture
def long_df():
    """Three units at deliberately unequal scales — the heterogeneity the lever targets."""
    futr, stat = futr_exog_list(), stat_exog_list()
    rng = np.random.default_rng(0)
    frames = []
    for i, (sid, scale) in enumerate([("u0", 1.0), ("u1", 20.0), ("u2", 0.5)]):
        base = scale * (2.0 + np.sin(np.arange(N) * 2 * np.pi / 168))
        g = pd.DataFrame(
            {
                "unique_id": sid,
                "ds": pd.date_range("2023-01-01", periods=N, freq="h"),
                "y": base + rng.normal(0, 0.05 * scale, N),
                "_hidx": np.arange(N),
            }
        )
        for c in futr:
            g[c] = rng.normal(size=N)
        for c in stat:
            g[c] = float(i)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- the identity


def test_the_weighted_normalised_loss_equals_the_wape_numerator():
    """The claim the whole lever rests on, as arithmetic."""
    y = np.array([10.0, 12.0, 100.0, 90.0])
    yhat = np.array([11.0, 11.0, 95.0, 99.0])
    level = np.array([10.0, 10.0, 100.0, 100.0])
    weighted_normalised = (level * np.abs(y / level - yhat / level)).sum()
    wape_numerator = np.abs(y - yhat).sum()
    assert weighted_normalised == pytest.approx(wape_numerator)


def test_normalising_without_the_weight_is_a_different_objective():
    """The misaligned third option — it de-weights exactly the units WAPE cares about most."""
    y = np.array([10.0, 100.0])
    yhat = np.array([11.0, 101.0])
    level = np.array([10.0, 100.0])
    unweighted = np.abs(y / level - yhat / level).sum()  # 0.1 + 0.01 — the big unit barely counts
    assert unweighted != pytest.approx(np.abs(y - yhat).sum())


def test_scaled_target_returns_both_halves_together(long_df):
    design = pd.DataFrame({"unique_id": ["u0", "u1"], "f": [0.0, 0.0]})
    levels = pd.Series({"u0": 2.0, "u1": 40.0})
    y = np.array([4.0, 80.0])
    z, w = L._scaled_target(design, y, levels)
    assert list(z) == [2.0, 2.0], "unequal units land on a common scale"
    assert list(w) == [2.0, 40.0], "the weight is the level"


def test_a_caller_supplied_weight_multiplies_through_rather_than_being_replaced():
    """4.5a recency weighting has to compose with this, not compete with it."""
    design = pd.DataFrame({"unique_id": ["u0", "u1"]})
    levels = pd.Series({"u0": 2.0, "u1": 4.0})
    _, w = L._scaled_target(design, np.array([1.0, 1.0]), levels, weight=np.array([0.5, 0.25]))
    assert list(w) == [1.0, 1.0]


# --------------------------------------------------------------------------- the level


def test_the_level_is_fitted_on_the_train_slice_only(long_df):
    """A level computed over the whole frame carries the scored block's mean into training."""
    spiked = long_df.copy()
    spiked.loc[spiked["_hidx"] >= CUT, "y"] *= 1000.0
    assert L.unit_levels(spiked, CUT).equals(L.unit_levels(long_df, CUT))


def test_the_level_is_a_median_not_a_mean(long_df):
    """Right-skewed target: one spike must not rescale a unit's entire history."""
    spiked = long_df.copy()
    spiked.loc[spiked.index[0], "y"] = 1e6
    lv = L.unit_levels(spiked, CUT)
    assert lv["u0"] < 10.0


def test_it_is_one_level_per_unit_and_it_tracks_the_scales(long_df):
    lv = L.unit_levels(long_df, CUT)
    assert set(lv.index) == {"u0", "u1", "u2"}
    assert lv["u1"] > lv["u0"] > lv["u2"]


def test_a_flat_zero_unit_falls_back_instead_of_dividing_by_zero(long_df):
    df = long_df.copy()
    df.loc[df["unique_id"] == "u2", "y"] = 0.0
    lv = L.unit_levels(df, CUT)
    assert lv["u2"] > 0 and np.isfinite(lv["u2"])


# --------------------------------------------------------------------------- end to end


def test_predictions_come_back_on_the_original_scale(long_df):
    """The member's output column must stay in the target's units, or the merge lands nonsense."""
    out = L.predict_gapped(
        long_df,
        CUT,
        horizon=48,
        score_len=24,
        origin_stride=48,
        num_boost_round=5,
        normalise_level=True,
    )
    big = out[out["unique_id"] == "u1"]["lgbm"]
    small = out[out["unique_id"] == "u2"]["lgbm"]
    assert big.mean() > 10 * small.mean(), "unit scale was not restored after prediction"


def test_the_lever_actually_changes_the_fit(long_df):
    """Plumbing that silently no-ops would read as 'no effect' rather than 'not applied'."""
    kw = {"horizon": 48, "score_len": 24, "origin_stride": 48, "num_boost_round": 5}
    plain = L.predict_gapped(long_df, CUT, **kw)
    normed = L.predict_gapped(long_df, CUT, normalise_level=True, **kw)
    assert not np.allclose(plain["lgbm"], normed["lgbm"])


def test_the_scored_rows_are_identical_to_the_plain_member(long_df):
    """Both arms must land on the same rows or the paired A/B is not paired."""
    kw = {"horizon": 48, "score_len": 24, "origin_stride": 48, "num_boost_round": 5}
    plain = L.predict_gapped(long_df, CUT, **kw)
    normed = L.predict_gapped(long_df, CUT, normalise_level=True, **kw)
    pd.testing.assert_frame_equal(plain[["unique_id", "ds", "y"]], normed[["unique_id", "ds", "y"]])


def test_the_early_stopping_fold_is_scored_on_the_same_scale_as_the_refit(long_df):
    """Otherwise best_iteration is tuned against a different objective from the one that runs."""
    report: dict = {}
    L.predict_gapped(
        long_df,
        CUT,
        horizon=48,
        score_len=24,
        origin_stride=48,
        early_stopping_rounds=5,
        max_boost_round=40,
        tuning_report=report,
        normalise_level=True,
    )
    plain: dict = {}
    L.predict_gapped(
        long_df,
        CUT,
        horizon=48,
        score_len=24,
        origin_stride=48,
        early_stopping_rounds=5,
        max_boost_round=40,
        tuning_report=plain,
    )
    # The normalised fold's L1 is on z, so it must differ from the raw-scale fold's L1.
    assert report["best_score"] != pytest.approx(plain["best_score"])


def test_the_member_is_registered_and_states_what_the_comparison_isolates():
    from src.models import members as mem

    spec = mem.get_member("lgbm_norm")
    assert spec.status == "untested"
    assert "lgbm_es" in spec.note


# --------------------------------------------------------------------------- 4.5a: recency


def test_the_origin_is_recovered_by_merge_not_by_row_order(long_df):
    """build_design groups by series, so a parallel array would line up wrong. Merge instead."""
    pairs = pd.concat(
        [
            pd.DataFrame({"unique_id": s, "o": o, "k": [1, 5, 9]})
            for s in ("u0", "u1", "u2")
            for o in (400, 900)
        ],
        ignore_index=True,
    )
    X, _ = L.build_design(long_df, pairs)
    got = L.origin_hidx(X, long_df)
    assert set(np.unique(got)) == {400.0, 900.0}


def test_recency_weight_halves_every_half_life(long_df):
    pairs = pd.DataFrame({"unique_id": "u0", "o": [CUT - 1, CUT - 1 - 100, CUT - 1 - 200], "k": 1})
    X, _ = L.build_design(long_df, pairs)
    w = L.recency_weights(X, long_df, CUT, halflife=100)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(0.25)


def test_recency_is_anchored_to_the_origin_not_the_forecast_hour(long_df):
    """Every row from one origin shares a vintage; it is the vintage that goes stale."""
    pairs = pd.DataFrame({"unique_id": "u0", "o": CUT - 1 - 200, "k": [1, 20, 40]})
    X, _ = L.build_design(long_df, pairs)
    w = L.recency_weights(X, long_df, CUT, halflife=100)
    assert len(set(np.round(w, 12))) == 1


def test_weighting_schemes_multiply_rather_than_replace():
    assert L._combine(None, np.array([0.5, 0.25])).tolist() == [0.5, 0.25]
    assert L._combine(np.array([2.0, 4.0]), np.array([0.5, 0.25])).tolist() == [1.0, 1.0]


def test_recency_and_normalisation_compose_end_to_end(long_df):
    """The two levers are meant to stack; a silent override would look like 'no effect'."""
    kw = {"horizon": 48, "score_len": 24, "origin_stride": 24, "num_boost_round": 5}
    plain = L.predict_gapped(long_df, CUT, **kw)
    rec = L.predict_gapped(long_df, CUT, recency_halflife=200, **kw)
    both = L.predict_gapped(long_df, CUT, recency_halflife=200, normalise_level=True, **kw)
    assert not np.allclose(plain["lgbm"], rec["lgbm"])
    assert not np.allclose(rec["lgbm"], both["lgbm"])


def test_a_zero_half_life_leaves_the_member_untouched(long_df):
    """Off by default has to mean bit-identical, or every recorded number moves."""
    kw = {"horizon": 48, "score_len": 24, "origin_stride": 24, "num_boost_round": 5}
    pd.testing.assert_frame_equal(
        L.predict_gapped(long_df, CUT, **kw),
        L.predict_gapped(long_df, CUT, recency_halflife=0, **kw),
    )


def test_the_recency_member_is_registered():
    from src.models import members as mem

    spec = mem.get_member("lgbm_recency")
    assert spec.status == "untested"
    assert "lgbm_es" in spec.note
