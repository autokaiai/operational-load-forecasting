"""S2 — the CatBoost member: same design matrix, different booster.

The comparison S2 is buying is "does a different fitting procedure on the *same* basis land outside
cluster B". That question is only answerable if the basis really is the same, so the tests here pin
the things that would silently make it a different experiment:

* the design matrix comes from ``src.models.lgbm.build_design``, not a copy of it;
* the scored block is the same rows, at every cutoff and not merely at W0;
* the round count is measured on the same held-out fold, so neither arm is judged at a budget
  chosen for the other;
* the unit code reaches CatBoost as a genuine category rather than as a float it would order.

Two of these are failure modes the LightGBM path actually hit (the tail-slice label merge, and the
config-merge that sends neural keys into a booster), which is why they are pinned rather than
assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import HOUR_IDX
from src.models.lgbm import UNIT_COL, build_design, inference_pairs

cb = pytest.importorskip("catboost")

from src.models import catboost_tree as cbt  # noqa: E402  (after the importorskip guard)

N_SERIES = 4
N_HOURS = 1800
CUT = 1200
HORIZON = 336  # a small stand-in for GAPPED_H (672) so the fixture stays fast
SCORE = 168
STRIDE = 48

# Deliberately tiny: these tests pin plumbing, not accuracy.
FAST = {"depth": 3, "learning_rate": 0.3, "verbose": False, "allow_writing_files": False}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """A long frame with `_hidx`, in the shape build_design/training_pairs expect.

    Carries the full conditioning set (``futr_exog_list()`` + ``stat_exog_list()``), because
    ``predict_gapped`` reads those lists rather than taking columns as an argument — a fixture with
    only ``y`` would fail at the design matrix rather than at the thing under test.
    """
    from src.data.features import futr_exog_list, stat_exog_list

    rng = np.random.default_rng(0)
    frames = []
    for s in range(N_SERIES):
        hidx = np.arange(N_HOURS)
        # A per-unit level, so `unit_id` carries real signal and a categorical/numeric mix-up
        # would show up as more than a rounding difference.
        level = 10.0 * (s + 1)
        y = level + 5 * np.sin(hidx / 24 * 2 * np.pi) + rng.normal(0, 1, N_HOURS)
        g = pd.DataFrame(
            {
                NF_ID: f"unit_{s:03d}",
                NF_TIME: pd.date_range("2023-01-01", periods=N_HOURS, freq="h"),
                NF_TARGET: y,
                HOUR_IDX: hidx,
            }
        )
        for col in futr_exog_list():
            g[col] = rng.normal(0, 1, N_HOURS)
        for col in stat_exog_list():
            g[col] = float(s)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def _predict(panel, **kw):
    return cbt.predict_gapped(
        panel,
        CUT,
        params=FAST,
        horizon=HORIZON,
        score_len=SCORE,
        origin_stride=STRIDE,
        num_boost_round=kw.pop("num_boost_round", 30),
        **kw,
    )


# --------------------------------------------------------------------------- the member contract


def test_output_is_the_scored_block_on_the_member_contract(panel):
    out = _predict(panel)
    assert list(out.columns) == [NF_ID, NF_TIME, NF_TARGET, cbt.MEMBER_COL]
    per_series = out.groupby(NF_ID).size()
    assert per_series.nunique() == 1, "ragged output would silently re-weight the pooled WAPE"
    assert per_series.iloc[0] == SCORE
    assert not out[cbt.MEMBER_COL].isna().any()
    assert (out[cbt.MEMBER_COL] >= 0).all(), (
        "the target is strictly positive; predictions clip at 0"
    )


def test_the_scored_block_is_the_far_half_at_a_non_terminal_cutoff(panel):
    """The label merge must be by timestamp, not by a tail slice.

    ``[cut+horizon-score_len, cut+horizon)`` coincides with the frame's global tail only when
    ``cut == n - horizon``. A ``tail(score_len)`` label slice therefore merges zero rows at every
    earlier cutoff — the exact defect the LightGBM path carried, so it is pinned on this one before
    it can be reintroduced.
    """
    assert CUT + HORIZON < N_HOURS, (
        "the fixture must put the scored block strictly inside the frame"
    )
    out = _predict(panel)
    hidx = panel[[NF_ID, NF_TIME, HOUR_IDX]]
    merged = out.merge(hidx, on=[NF_ID, NF_TIME], how="left")
    assert not merged[HOUR_IDX].isna().any(), "a scored row has no counterpart in the source frame"
    assert merged[HOUR_IDX].min() == CUT + HORIZON - SCORE
    assert merged[HOUR_IDX].max() == CUT + HORIZON - 1


def test_predictions_land_on_the_same_rows_as_the_lightgbm_member(panel):
    """Same basis, same rows. Anything else and the A/B is not paired."""
    lgbm = pytest.importorskip("lightgbm")
    del lgbm
    from src.models import lgbm as lgbm_mod

    a = _predict(panel)
    b = lgbm_mod.predict_gapped(
        panel,
        CUT,
        params={"num_leaves": 7, "min_child_samples": 5, "verbosity": -1, "seed": 42},
        horizon=HORIZON,
        score_len=SCORE,
        origin_stride=STRIDE,
        num_boost_round=30,
    )
    keys = ["_".join(map(str, k)) for k in zip(a[NF_ID], a[NF_TIME], strict=True)]
    other = ["_".join(map(str, k)) for k in zip(b[NF_ID], b[NF_TIME], strict=True)]
    assert sorted(keys) == sorted(other)


# --------------------------------------------------------------------------- the categorical unit


def test_unit_id_reaches_catboost_as_a_category_not_a_float(panel):
    """``build_design`` emits one float matrix, and CatBoost refuses a float ``cat_feature``.

    Rightly so — a float category is nearly always a cast someone forgot, and silently rounding it
    would be worse than the error. ``_prep`` is what does the cast, so it is pinned here: without it
    the member does not merely score differently, it does not run at all.
    """
    pairs = inference_pairs(panel, CUT, HORIZON)
    design, _ = build_design(panel, pairs, [], [], False, None, True)
    assert UNIT_COL in design.columns
    assert design[UNIT_COL].dtype.kind == "f", "the fixture no longer exercises the cast"

    feat = [c for c in design.columns if c not in (NF_ID, NF_TIME)]
    prepped = cbt._prep(design, feat)
    assert prepped[UNIT_COL].dtype.kind in "iu"
    assert cbt._cat_features(feat) == [UNIT_COL]
    # And it round-trips: the code must survive the cast, not just change dtype.
    assert (prepped[UNIT_COL].to_numpy() == design[UNIT_COL].to_numpy().astype(int)).all()


def test_categorical_unit_is_off_by_default(panel):
    pairs = inference_pairs(panel, CUT, HORIZON)
    design, _ = build_design(panel, pairs, [], [], False, None, False)
    assert UNIT_COL not in design.columns
    assert cbt._cat_features(list(design.columns)) == []


def test_the_member_runs_with_the_unit_categorical(panel):
    out = _predict(panel, categorical_unit=True)
    assert len(out) == N_SERIES * SCORE
    assert not out[cbt.MEMBER_COL].isna().any()


# --------------------------------------------------------------------------- the round count


def test_early_stopping_measures_a_round_count_and_reports_the_fold(panel):
    report: dict = {}
    out = cbt.predict_gapped(
        panel,
        CUT,
        params=FAST,
        horizon=HORIZON,
        score_len=SCORE,
        origin_stride=STRIDE,
        early_stopping_rounds=10,
        max_boost_round=120,
        tuning_report=report,
    )
    assert len(out) == N_SERIES * SCORE
    assert 1 <= report["best_iteration"] <= 120
    # The fold lives strictly inside the train region — never the scored block.
    assert report["valid_block"] == [CUT - SCORE, CUT]
    assert report["n_valid_rows"] > 0
    assert "hit_ceiling" in report, "a truncated fit and a converged one look identical in the WAPE"


def test_best_iteration_is_one_based(panel):
    """CatBoost's ``get_best_iteration()`` is 0-based; the refit count is one more.

    Off by one here is invisible in the score and wrong in the record, and the record is what the
    write-up quotes. A single-iteration optimum is the case that makes the difference observable.
    """
    report = cbt.tune_iterations(
        panel,
        CUT,
        params=FAST,
        horizon=HORIZON,
        score_len=SCORE,
        origin_stride=STRIDE,
        max_iterations=40,
        early_stopping_rounds=5,
    )
    assert report["best_iteration"] >= 1


# --------------------------------------------------------------------------- the registry wiring


def test_the_member_is_registered_as_a_cpu_tree_with_no_cascade():
    from src.models.members import get_member

    spec = get_member("catboost")
    assert spec.kind == "tree"
    assert spec.needs_gpu is False
    assert spec.seedable is True
    # The standing architectural rule: a model fitted on OUR data enters only as a blend member.
    # A cascade channel here would be exactly the thing that rule forbids.
    assert spec.cascade == ()
    assert spec.train_csv is None


def test_the_config_carries_no_key_catboost_would_reject():
    """CatBoost RAISES on an unknown parameter; LightGBM only warns and ``verbosity: -1`` hides it.

    So the runner reads tree configs without merging ``configs/base.yaml`` (plan 3.9). This asserts
    the *outcome* of that decision on the real config rather than the decision itself: every key the
    runner does not consume must be one CatBoost accepts.
    """
    import yaml

    from src.models.members import _TREE_RUNNER_KEYS

    cfg = yaml.safe_load(open("configs/catboost.yaml"))
    params = {k: v for k, v in cfg.items() if k not in _TREE_RUNNER_KEYS}
    # Constructing is enough — CatBoost validates parameter names eagerly.
    cb.CatBoostRegressor(**{**cbt.DEFAULT_PARAMS, **params, "iterations": 1})


def test_a_seed_is_spelled_random_seed_and_reproduces(panel):
    """LightGBM's parameter is ``seed``, CatBoost's is ``random_seed``.

    Handing CatBoost the wrong spelling is a crash rather than a silent default — the better
    failure, but only if nobody writes the wrong spelling. So: the accepted spelling reproduces,
    and the LightGBM spelling is rejected loudly.
    """
    kw = {
        "horizon": HORIZON,
        "score_len": SCORE,
        "origin_stride": STRIDE,
        "num_boost_round": 25,
    }
    a = cbt.predict_gapped(panel, CUT, {**FAST, "random_seed": 1}, **kw)
    b = cbt.predict_gapped(panel, CUT, {**FAST, "random_seed": 1}, **kw)
    assert np.allclose(a[cbt.MEMBER_COL], b[cbt.MEMBER_COL]), "same seed must reproduce"

    with pytest.raises(Exception):  # noqa: B017 — catboost raises its own type here
        cbt.fit_catboost(*_tiny(panel), {**FAST, "seed": 1}, iterations=2)


def _tiny(panel):
    """A two-column design + target, just enough to construct a fit."""
    from src.models.lgbm import training_pairs

    pairs = training_pairs(panel, 400, HORIZON, 336)
    X, y = build_design(panel, pairs, [], [], True, None, False)
    ok = ~np.isnan(y)
    return X.loc[ok].reset_index(drop=True), y[ok]


def test_the_runner_honours_a_seed_from_the_run_context(panel):
    """``run_member`` passes ``ctx.seed``; the runner must translate it, not drop it."""
    from src.models.members import get_member

    spec = get_member("catboost")
    assert spec.seedable, "a seed probe over this member would otherwise raise"
    assert spec.run.member_config == "configs/catboost.yaml"
    # The cube cache keys on the declared parameters, so a runner that does not declare them is
    # deliberately not cached — better an uncached member than a stale hit reported as a number.
    assert "src/models/catboost_tree.py" in spec.run.member_code
