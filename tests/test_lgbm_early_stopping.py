"""Phase 4.1a — the round count is measured, not assumed.

``num_boost_round: 600`` was hand-picked with no validation set anywhere in the pipeline. It is a
property of the current feature set, objective and weighting, so every Phase-4 lever that changes
any of those would have been measured against a stale round count — confounding the lever with the
truncation. This pins the fold that fixes it.

What actually needs guarding is not "does early stopping run" but **does the fold leak**. A random
split over a rolling-origin design does: the same forecast hour recurs under many origins, so a
held-out row almost always has a near-duplicate in train, and early stopping then runs far past the
honest optimum. The fold is therefore built by time, and the tests below pin that shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import HOUR_IDX
from src.models.lgbm import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    GAPPED_H,
    early_stopping_pairs,
    fit_lgbm,
    training_pairs,
)

lgb = pytest.importorskip("lightgbm")

N_SERIES = 4
N_HOURS = 2400
CUT = 1800
HORIZON = 336  # a small stand-in for GAPPED_H so the fixture stays fast
SCORE = 168
STRIDE = 48
FAST = {"num_leaves": 7, "min_child_samples": 5, "verbosity": -1, "seed": 42, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """A long frame with `_hidx`, in the shape build_design/training_pairs expect."""
    rng = np.random.default_rng(0)
    frames = []
    for s in range(N_SERIES):
        hidx = np.arange(N_HOURS)
        y = 50.0 + 10 * np.sin(hidx / 24 * 2 * np.pi) + rng.normal(0, 2, N_HOURS)
        frames.append(
            pd.DataFrame(
                {
                    NF_ID: f"unit_{s:03d}",
                    NF_TIME: pd.date_range("2023-01-01", periods=N_HOURS, freq="h"),
                    NF_TARGET: y,
                    HOUR_IDX: hidx,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- the fold's shape


def test_the_fold_never_touches_the_scored_block(panel):
    """Everything — train origins, valid origins, and every target — lives at _hidx < cut."""
    tr, va = early_stopping_pairs(panel, CUT, HORIZON, SCORE, STRIDE)
    for grid in (tr, va):
        assert (grid["o"] + grid["k"]).max() < CUT, "a fold target reached into the scored block"
        assert grid["o"].max() < CUT


def test_validation_is_the_336h_block_immediately_before_the_cut(panel):
    """The fold mirrors the graded task: the last `score_len` hours of the train region."""
    _, va = early_stopping_pairs(panel, CUT, HORIZON, SCORE, STRIDE)
    fc = (va["o"] + va["k"]).to_numpy()
    assert fc.min() == CUT - SCORE
    assert fc.max() == CUT - 1
    assert len(np.unique(va["o"])) == 1, "one validation origin per series, not a scatter"
    assert va["o"].iloc[0] == CUT - 1 - HORIZON, "forecast from a full horizon back, across the gap"


def test_training_targets_stop_before_the_validation_block_begins(panel):
    """The leakage check that matters. No training target may land inside the validation window."""
    tr, va = early_stopping_pairs(panel, CUT, HORIZON, SCORE, STRIDE)
    val_start = (va["o"] + va["k"]).min()
    assert (tr["o"] + tr["k"]).max() < val_start


def test_validation_features_predate_the_validation_targets(panel):
    """The other leakage direction: the origin's lags must not read the answer.

    The validation origin is `cut - 1 - horizon`, so its features read y no later than that —
    comfortably before the targets begin at `cut - score_len`.
    """
    _, va = early_stopping_pairs(panel, CUT, HORIZON, SCORE, STRIDE)
    assert va["o"].max() < (va["o"] + va["k"]).min()


def test_the_fold_is_a_time_split_not_a_random_one(panel):
    """No (series, forecast-hour) appears in both halves — the failure mode a random split has."""
    tr, va = early_stopping_pairs(panel, CUT, HORIZON, SCORE, STRIDE)
    key = lambda g: set(zip(g[NF_ID], g["o"] + g["k"], strict=True))  # noqa: E731
    assert not (key(tr) & key(va))


def test_training_grid_matches_the_ordinary_one_pulled_back(panel):
    """The train half is the ordinary rolling grid with the cutoff moved, not a bespoke one."""
    tr, _ = early_stopping_pairs(panel, CUT, HORIZON, SCORE, STRIDE)
    expected = training_pairs(panel, CUT - SCORE, HORIZON, STRIDE)
    pd.testing.assert_frame_equal(tr.reset_index(drop=True), expected.reset_index(drop=True))


def test_default_horizon_is_the_gapped_one(panel):
    _, va = early_stopping_pairs(panel, CUT, stride=STRIDE)
    assert va["o"].iloc[0] == CUT - 1 - GAPPED_H


# --------------------------------------------------------------------------- fit_lgbm behaviour


def make_xy(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 3 * X["a"].to_numpy() - 2 * X["b"].to_numpy() + rng.normal(0, 0.5, n)
    return X, y


def test_early_stopping_sets_best_iteration_below_the_ceiling():
    Xtr, ytr = make_xy(seed=0)
    Xva, yva = make_xy(seed=1)
    booster, _ = fit_lgbm(
        Xtr,
        ytr,
        {**FAST},
        num_boost_round=2000,
        valid_data=(Xva, yva),
        early_stopping_rounds=20,
    )
    assert 0 < booster.best_iteration < 2000


def test_without_a_valid_set_nothing_stops_early():
    Xtr, ytr = make_xy()
    booster, _ = fit_lgbm(Xtr, ytr, {**FAST}, num_boost_round=40)
    assert booster.num_trees() == 40


def test_an_explicit_round_count_beats_one_left_in_params():
    """The bug this found: lgb.train treats a round count in `params` as authoritative.

    `fit_lgbm` used to pop it only when the caller passed 0, so an explicit count was silently
    discarded whenever the config also set one — and configs/lgbm.yaml sets 600. Harmless while
    every caller passed 0; not harmless once early stopping passes a measured best_iteration.
    """
    Xtr, ytr = make_xy()
    booster, _ = fit_lgbm(Xtr, ytr, {**FAST, "num_boost_round": 600}, num_boost_round=17)
    assert booster.num_trees() == 17


def test_a_round_count_in_params_is_still_honoured_when_no_argument_is_given():
    """The pre-existing path must not move — this is how every recorded number was produced."""
    Xtr, ytr = make_xy()
    booster, _ = fit_lgbm(Xtr, ytr, {**FAST, "num_boost_round": 23})
    assert booster.num_trees() == 23


def test_sample_weights_reach_the_validation_set_too():
    """A weighted objective needs a weighted validation score, or early stopping optimises the
    wrong quantity — it would stop where the *unweighted* loss plateaus."""
    Xtr, ytr = make_xy(seed=0)
    Xva, yva = make_xy(seed=1)
    w = np.linspace(0.1, 10.0, len(yva))
    a, _ = fit_lgbm(
        Xtr, ytr, {**FAST}, num_boost_round=200, valid_data=(Xva, yva), early_stopping_rounds=10
    )
    b, _ = fit_lgbm(
        Xtr, ytr, {**FAST}, num_boost_round=200, valid_data=(Xva, yva, w), early_stopping_rounds=10
    )
    assert a.best_score["valid"]["l1"] != b.best_score["valid"]["l1"]


def test_the_default_patience_is_declared_not_magic():
    assert DEFAULT_EARLY_STOPPING_ROUNDS > 0
