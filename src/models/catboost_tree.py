"""Direct-multistep CatBoost member — the SAME design matrix, a different fitting procedure.

Why this exists, and what it is a test of ----------------------------------------- The
error-correlation matrix has two clusters (plan S2.0): ``{tft, tft_cascade}`` on one side and
``{chronos_ft, lgbm, lstm, bitcn, lgbm_s24_unitcat}`` on the other, with a minimum of 0.916 *within*
either block and a maximum of 0.813 *across* them. Five architectures that share nothing but a
training set are, as far as the errors go, one member. So the prior for a second GBDT is
emphatically cluster B — ``lgbm|lgbm_s24_unitcat`` is already 0.976, and that is the *same library*
at two parameter settings; a different library on the same 41-feature basis should not be far off.

That prior is cheap to *measure* rather than assert (3 CPU containers), and the measurement is worth
having either way: if CatBoost lands at err-corr ~0.95 against the frozen tree it is evidence FOR
the two-cluster reading, which the write-up wants; if it lands materially lower, the reading is
wrong and S2 found a member without spending a GPU minute.

The comparison is deliberately narrow: **only the booster changes.** Same design matrix
(``src.models.lgbm.build_design``), same rolling-origin grid, same ``origin_stride``, same
early-stopping fold, same unit-as-categorical lever — which is why this module imports lgbm's
*design* helpers (pure functions, no fitting) instead of copying them. What differs is ordered
boosting, symmetric trees, and native categorical handling.

What deliberately does NOT change --------------------------------- ``loss_function="MAE"``. Plan
4.1 established that pooled WAPE is ``sum|y-yhat| / sum|y|`` with a model-independent denominator,
so minimising ``sum|y-yhat|`` *is* minimising WAPE, and L1 on the raw target minimises exactly that.
An earlier note in the stacker work reached for ``MultiRMSE``; that is superseded and would misalign
this member against the metric it is scored on.

A trap this module is written around
------------------------------------
**CatBoost RAISES on an unknown parameter where LightGBM only warns.** Plan 3.9 found that tree
configs were being merged with ``configs/base.yaml``, so every neural-runner key (``h``,
``input_size``, ``max_steps``, ``accelerator``, …) was reaching the booster; LightGBM ignored them
with a warning that ``verbosity: -1`` then hid. The registry now reads tree configs *without* the
base merge, which is what keeps this member from failing on arrival — the fix landed before the
member that needed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import futr_exog_list, stat_exog_list
from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import BLOCK_REGIMES, SCORE_LEN, add_hour_index
from src.models.lgbm import (
    GAPPED_H,
    UNIT_COL,
    _combine,
    _level_of,
    _scaled_target,
    build_design,
    early_stopping_pairs,
    inference_pairs,
    recency_weights,
    training_pairs,
    unit_levels,
)

MEMBER_COL = "catboost"

# CatBoost at close to its own defaults, with two deliberate choices.
#
# `loss_function: MAE` — the aligned objective (see the module docstring).
#
# `depth: 6` — CatBoost grows *symmetric* (oblivious) trees, so depth d means exactly 2**d leaves.
# 6 gives 64, matching LightGBM's `num_leaves: 63` closely enough that the comparison is about the
# boosting procedure rather than about capacity. Deeper symmetric trees are also where CatBoost gets
# expensive, and this member is CPU-bound at origin_stride 24.
DEFAULT_PARAMS = {
    "loss_function": "MAE",
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,  # otherwise every container litters a catboost_info/ directory
}
DEFAULT_ITERATIONS = 600
DEFAULT_EARLY_STOPPING_ROUNDS = 50


def _prep(X: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    """Feature columns only, with the unit code as a genuine integer category.

    ``build_design`` materialises one float matrix, so the unit code arrives as ``12.0``. CatBoost
    refuses a float column in ``cat_features`` — and rightly: a float category is almost always a
    number someone forgot to cast, and silently rounding it would be worse than the error.
    """
    out = X[feat].copy()
    if UNIT_COL in out.columns:
        out[UNIT_COL] = out[UNIT_COL].astype(int)
    return out


def _cat_features(feat: list[str]) -> list[str]:
    return [UNIT_COL] if UNIT_COL in feat else []


def fit_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict | None = None,
    iterations: int = 0,
    sample_weight: np.ndarray | None = None,
    valid_data: tuple | None = None,
    early_stopping_rounds: int = 0,
):
    """Train a CatBoost regressor on the design matrix (feature columns only).

    Mirrors :func:`src.models.lgbm.fit_lgbm` in contract and in the two-step early-stopping
    protocol, so the A/B against the frozen tree is paired on procedure as well as on rows.

    ``sample_weight`` is here for the same reason it is on the LightGBM side and carries the same
    warning: the tree needs **no** weighting to match the graded metric. MAE on the raw target is
    already the WAPE numerator. The hook exists for per-unit normalisation (plan 4.6, measured and
    rejected) and for ablations, not as a default to reach for.
    """
    from catboost import CatBoostRegressor, Pool

    p = {**DEFAULT_PARAMS, **(params or {})}
    # Pop unconditionally, like the LightGBM path: a round count living in `params` and one passed
    # as an argument are two authorities over the same number, and the argument must win when the
    # early-stopping fold has measured it.
    from_params = p.pop("iterations", None)
    fallback = from_params if from_params is not None else DEFAULT_ITERATIONS
    p["iterations"] = int(iterations or fallback)

    feat = [c for c in X.columns if c not in (NF_ID, NF_TIME)]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(sample_weight) != len(X):
            raise ValueError(
                f"sample_weight has {len(sample_weight)} rows, design matrix has {len(X)}"
            )
    cats = _cat_features(feat)
    train_pool = Pool(_prep(X, feat), label=y, weight=sample_weight, cat_features=cats)

    fit_kwargs: dict = {}
    if valid_data is not None and early_stopping_rounds > 0:
        Xva, yva = valid_data[0], valid_data[1]
        wva = valid_data[2] if len(valid_data) > 2 else None
        fit_kwargs["eval_set"] = Pool(_prep(Xva, feat), label=yva, weight=wva, cat_features=cats)
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        fit_kwargs["use_best_model"] = True

    model = CatBoostRegressor(**p)
    model.fit(train_pool, **fit_kwargs)
    return model, feat


def tune_iterations(
    long_df: pd.DataFrame,
    cut_idx: int,
    params: dict | None = None,
    horizon: int = GAPPED_H,
    score_len: int = SCORE_LEN,
    origin_stride: int = 168,
    max_iterations: int = 3000,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    sample_weight_fn=None,
    fc_lags: list[int] | None = None,
    categorical_unit: bool = False,
    normalise_level: bool = False,
    recency_halflife: float = 0.0,
) -> dict:
    """Measure the honest iteration count on the same held-out fold the tree uses.

    The fold comes from :func:`src.models.lgbm.early_stopping_pairs` verbatim — it mirrors the
    graded task (one origin per series at ``cut-1-horizon``, scored on the last 336 steps) rather
    than being a random split, because a random split over a rolling-origin design leaks badly: the
    same forecast hour recurs under many origins, so a held-out row almost always has
    near-duplicates in train and early stopping runs far past the honest optimum.

    **Check ``hit_ceiling``.** If boosting ran out of iterations before the validation MAE
    plateaued, ``best_iteration`` is a floor rather than an optimum.
    """
    long_df = add_hour_index(long_df)
    futr_cols, stat_cols = futr_exog_list(), stat_exog_list()

    tr_pairs, va_pairs = early_stopping_pairs(long_df, cut_idx, horizon, score_len, origin_stride)
    Xtr, ytr = build_design(
        long_df, tr_pairs, futr_cols, stat_cols, True, fc_lags, categorical_unit
    )
    Xva, yva = build_design(
        long_df, va_pairs, futr_cols, stat_cols, True, fc_lags, categorical_unit
    )
    ok_tr, ok_va = ~np.isnan(ytr), ~np.isnan(yva)
    Xtr, ytr = Xtr.loc[ok_tr].reset_index(drop=True), ytr[ok_tr]
    Xva, yva = Xva.loc[ok_va].reset_index(drop=True), yva[ok_va]
    if not len(Xva):
        raise ValueError(
            f"early-stopping fold is empty at cut_idx={cut_idx}: the validation block "
            f"[{cut_idx - score_len}, {cut_idx}) has no observed targets."
        )

    wtr = sample_weight_fn(Xtr, ytr) if sample_weight_fn is not None else None
    wva = sample_weight_fn(Xva, yva) if sample_weight_fn is not None else None
    if recency_halflife > 0:
        wtr = _combine(wtr, recency_weights(Xtr, long_df, cut_idx, recency_halflife))
        wva = _combine(wva, recency_weights(Xva, long_df, cut_idx, recency_halflife))
    if normalise_level:
        levels = unit_levels(long_df, cut_idx)
        ytr, wtr = _scaled_target(Xtr, ytr, levels, wtr)
        yva, wva = _scaled_target(Xva, yva, levels, wva)

    model, _ = fit_catboost(
        Xtr,
        ytr,
        params,
        iterations=max_iterations,
        sample_weight=wtr,
        valid_data=(Xva, yva, wva),
        early_stopping_rounds=early_stopping_rounds,
    )
    # CatBoost's best_iteration_ is 0-based; the count of trees to refit is one more.
    best = int(model.get_best_iteration() or 0) + 1
    scores = model.get_best_score() or {}
    valid = scores.get("validation", {})
    return {
        "best_iteration": best,
        "best_score": float(next(iter(valid.values()), float("nan"))) if valid else float("nan"),
        "max_boost_round": int(max_iterations),
        "early_stopping_rounds": int(early_stopping_rounds),
        "n_train_rows": int(len(Xtr)),
        "n_valid_rows": int(len(Xva)),
        "valid_block": [int(cut_idx - score_len), int(cut_idx)],
        "hit_ceiling": bool(best >= max_iterations - early_stopping_rounds),
    }


def predict_gapped(
    long_df: pd.DataFrame,
    cut_idx: int,
    params: dict | None = None,
    horizon: int = GAPPED_H,
    score_len: int = SCORE_LEN,
    origin_stride: int = 168,
    num_boost_round: int = 0,
    sample_weight_fn=None,
    early_stopping_rounds: int = 0,
    max_boost_round: int = 3000,
    tuning_report: dict | None = None,
    fc_lags: list[int] | None = None,
    categorical_unit: bool = False,
    normalise_level: bool = False,
    recency_halflife: float = 0.0,
    regime: str = "far",
) -> pd.DataFrame:
    """Fit on the train region (``_hidx < cut``) and forecast the scored block.

    Returns ``[unique_id, ds, y, catboost]`` for the scored block only — the final ``score_len``
    steps of the ``horizon``-hour forecast — so this member lands on the exact same rows as every
    other one. Signature deliberately mirrors :func:`src.models.lgbm.predict_gapped`; the member
    runner in ``src.models.members`` calls both through the same shape.
    """
    if regime not in BLOCK_REGIMES:
        raise ValueError(f"regime must be one of {BLOCK_REGIMES}, got {regime!r}")
    long_df = add_hour_index(long_df)
    futr_cols, stat_cols = futr_exog_list(), stat_exog_list()

    if early_stopping_rounds > 0:
        report = tune_iterations(
            long_df,
            cut_idx,
            params=params,
            horizon=horizon,
            score_len=score_len,
            origin_stride=origin_stride,
            max_iterations=max_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            sample_weight_fn=sample_weight_fn,
            fc_lags=fc_lags,
            categorical_unit=categorical_unit,
            normalise_level=normalise_level,
            recency_halflife=recency_halflife,
        )
        num_boost_round = report["best_iteration"]
        if tuning_report is not None:
            tuning_report.update(report)

    tr_pairs = training_pairs(long_df, cut_idx, horizon, origin_stride)
    Xtr, ytr = build_design(
        long_df, tr_pairs, futr_cols, stat_cols, True, fc_lags, categorical_unit
    )
    ok = ~np.isnan(ytr)
    Xtr, ytr = Xtr.loc[ok].reset_index(drop=True), ytr[ok]
    weights = sample_weight_fn(Xtr, ytr) if sample_weight_fn is not None else None
    if recency_halflife > 0:
        weights = _combine(weights, recency_weights(Xtr, long_df, cut_idx, recency_halflife))
    levels = unit_levels(long_df, cut_idx) if normalise_level else None
    if normalise_level:
        ytr, weights = _scaled_target(Xtr, ytr, levels, weights)
    model, feat = fit_catboost(Xtr, ytr, params, iterations=num_boost_round, sample_weight=weights)

    inf_pairs = inference_pairs(long_df, cut_idx, horizon)
    Xinf, _ = build_design(
        long_df, inf_pairs, futr_cols, stat_cols, False, fc_lags, categorical_unit
    )
    preds = model.predict(_prep(Xinf, feat))
    if normalise_level:
        preds = preds * _level_of(Xinf, levels)
    Xinf = Xinf.assign(**{MEMBER_COL: np.clip(preds, 0.0, None)})

    # Scored block = the final `score_len` steps of the horizon. Labels are matched on the scored
    # rows' own timestamps rather than by a tail slice: [cut+horizon-score_len, cut+horizon) only
    # coincides with the global tail at W0, and a tail() slice would silently merge 0 rows at every
    # earlier cutoff. Same reasoning, same fix, as the LightGBM path.
    # ``regime`` picks which half of the horizon is returned — ``far`` (the last ``score_len``
    # steps, the graded private-test block and the default that reproduces every recorded number)
    # or ``near`` (the first ``score_len``). One booster, one inference grid, both halves; the
    # signature mirrors ``lgbm.predict_gapped`` because the member runner calls both through the
    # same shape and a divergence there is a silent mis-slice.
    keep = (
        Xinf["horizon_step"] > horizon - score_len
        if regime == "far"
        else Xinf["horizon_step"] <= score_len
    )
    scored = Xinf[keep].copy()
    labels = long_df[[NF_ID, NF_TIME, NF_TARGET]]
    out = scored[[NF_ID, NF_TIME, MEMBER_COL]].merge(labels, on=[NF_ID, NF_TIME], how="inner")
    return out[[NF_ID, NF_TIME, NF_TARGET, MEMBER_COL]]
