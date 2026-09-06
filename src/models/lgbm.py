"""Direct-multistep LightGBM member (and the shared feature core for the residual/stacker arms).

A single LightGBM regressor forecasts the whole 672h gapped horizon directly: the lag features are
*anchored at the forecast origin* (the last observed hour before the gap) and held constant across
the horizon, while a ``horizon_step`` feature and the known-future covariates carry the time-varying
signal. Direct-multistep (one model, step as a feature) is preferred over recursive rollout here
because recursion compounds error over 672 steps and cannot use future-known covariates cleanly.

Leakage discipline:
  * lags / rolling means read only ``y`` at ``_hidx <= origin`` (origin < cut at inference) — never
    a future target;
  * calendar + ``*_forecast`` known-future covariates are legitimately available at every horizon
    hour and are read at the forecast time;
  * the caller is responsible for fitting imputation stats on the window's own train slice (see
    ``scripts/member_preds_window.py``) so this module never peeks past the cutoff.

The same ``build_design`` core (origin/step → feature row) backs the standalone member, the residual
corrector (``src.models.residual_lgbm``, target = y − TFT_oof) and the tree-stacker (Arm F).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import futr_exog_list, stat_exog_list
from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import BLOCK_REGIMES, HOUR_IDX, SCORE_LEN, add_hour_index

# Origin-anchored target features. Lag ``l`` reads y[origin-(l-1)] (lag 1 = the origin itself);
# rolling means summarise the last ``w`` observed hours. {1,24,48,168,336,504} spans the recent
# value, the daily and 2-day echoes, the weekly and bi-/tri-weekly echoes.
LAGS = [1, 24, 48, 168, 336, 504]
ROLLS = [24, 168]
GAPPED_H = 2 * SCORE_LEN  # 672 — gap (336) + scored block (336), matches src.eval.cv

# Forecast-hour-anchored ("weekly-aligned") lags — plan 4.2a, a DIFFERENT feature family from
# ``LAGS`` above, not more entries in it.
#
# The origin-anchored lags read ``y[origin-(l-1)]``: one value per (series, origin), constant across
# the whole 672h horizon. They tell the tree where the series *was* when the forecast was issued,
# and nothing about which hour it is forecasting.
#
# A forecast-hour-anchored lag reads ``y[fc - l]`` and therefore *varies across the horizon*. Being
# exact multiples of 168 these land on the same hour-of-week as the target, so they carry that
# unit's own level, drift and idiosyncrasy at the matching phase — what a shared 168-bin profile
# averages away.
#
# **Availability, not staleness, is the criterion, and it fixes the minimum.** For a forecast hour
# ``fc = o + k`` with ``k <= horizon``, ``fc - l = o + k - l``, which is at or before the origin iff
# ``l >= horizon``. So 672 (= 4 weeks = the gapped horizon exactly, worst case ``k=672`` reading
# ``y[o]``) is the smallest weekly multiple that is observable at *every* step; 840 (5w) and 1008
# (6w) follow. Anything shorter — 336, say — would read ``y[o+336]`` at ``k=672``, straight from the
# future. :func:`validate_fc_lags` enforces this rather than trusting the constant.
WEEKLY_LAGS = [672, 840, 1008]

# Moderately-regularised MAE-aligned defaults; a config (configs/lgbm.yaml) may override any key.
DEFAULT_PARAMS = {
    "objective": "regression_l1",  # L1 == the WAPE training proxy
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "num_threads": 0,
    "seed": 42,
    "verbosity": -1,
}
DEFAULT_NUM_BOOST_ROUND = 600
# Patience for the early-stopping fold. 50 rounds at lr=0.03 is ~1.5 effective units of learning
# rate — enough to ride out the noise in a 32k-row validation block without waiting out a genuine
# plateau.
DEFAULT_EARLY_STOPPING_ROUNDS = 50


def _lag_col(lag: int) -> str:
    return f"lag_{lag}"


def _roll_col(win: int) -> str:
    return f"roll_{win}"


def _wlag_col(lag: int) -> str:
    """Distinct prefix from ``lag_``: same integer, different anchor, so the names must differ."""
    return f"wlag_{lag}"


def validate_fc_lags(fc_lags, horizon: int = GAPPED_H) -> list[int]:
    """Reject a forecast-hour-anchored lag that would read the future. See ``WEEKLY_LAGS``.

    ``fc - l <= o`` for every ``k <= horizon`` iff ``l >= horizon``. A shorter lag does not fail
    loudly on its own — ``build_design`` would happily read an in-range index past the origin and
    the leak would surface only as an implausibly good CV score, so it is checked here.
    """
    out = [int(x) for x in (fc_lags or ())]
    bad = [x for x in out if x < horizon]
    if bad:
        raise ValueError(
            f"forecast-hour-anchored lag(s) {bad} are shorter than the horizon ({horizon}); at "
            f"step k they would read y[o + k - lag] > y[o] — a future target. Use lags >= {horizon}"
            f" (weekly multiples: {WEEKLY_LAGS})."
        )
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate forecast-hour-anchored lags: {out}")
    return out


UNIT_COL = "unit_id"


def feature_columns(
    futr_cols: list[str],
    stat_cols: list[str],
    fc_lags: list[int] | None = None,
    categorical_unit: bool = False,
) -> list[str]:
    """Feature order: lags, weekly lags, rolls, horizon_step, futr, statics, [unit_id]."""
    return (
        [_lag_col(lag) for lag in LAGS]
        + [_wlag_col(lag) for lag in (fc_lags or ())]
        + [_roll_col(w) for w in ROLLS]
        + ["horizon_step"]
        + list(futr_cols)
        + list(stat_cols)
        + ([UNIT_COL] if categorical_unit else [])
    )


def unit_codes(long_df: pd.DataFrame) -> dict:
    """Stable ``series id -> integer code`` map for the categorical unit feature.

    Sorted, so the same window's train and inference designs agree and two runs of the same window
    produce the same codes. The code is arbitrary and LightGBM never orders it — that is the point
    of declaring it categorical rather than leaving it as a number the tree would split on with
    ``<=``, which would impose a meaningless ordering on 96 unrelated units.
    """
    return {sid: i for i, sid in enumerate(sorted(long_df[NF_ID].unique()))}


def unit_levels(long_df: pd.DataFrame, cut_idx: int) -> pd.Series:
    """Per-unit level (train-slice median of ``y``) for plan 4.6's normalise-and-weight scheme.

    **Fitted on the window's own train slice** (``_hidx < cut_idx``), like the imputation statistics
    — a level computed over the whole frame would carry the scored block's mean into training.

    **Constant per unit, deliberately.** The whole scheme rests on the level cancelling::

        w * |z - zhat|  =  level * |y - yhat| / level  =  |y - yhat|

    so a time-varying level would not cancel and the objective would drift off WAPE. The median
    rather than the mean because the target is right-skewed and a single spike would then rescale
    that unit's entire history.

    A unit whose train median is 0 or missing falls back to the cross-unit median: dividing by it
    would produce inf, and a unit that is flat-zero over the train slice has no scale to speak of.
    """
    tr = long_df[long_df[HOUR_IDX] < cut_idx] if HOUR_IDX in long_df else long_df
    lv = tr.groupby(NF_ID)[NF_TARGET].median()
    lv = lv.where(lv > 0)
    fallback = lv.median()
    return lv.fillna(fallback if fallback and fallback > 0 else 1.0)


def _level_of(design: pd.DataFrame, levels: pd.Series) -> np.ndarray:
    """Each design row's unit level, aligned by ``unique_id``."""
    return design[NF_ID].map(levels).to_numpy(dtype=float)


def origin_hidx(design: pd.DataFrame, long_df: pd.DataFrame) -> np.ndarray:
    """Each design row's forecast ORIGIN as an ``_hidx``: the forecast hour minus ``horizon_step``.

    Recovered by merging rather than threaded through ``build_design``, because the design's row
    order is "grouped by series, then pairs order within a group" — reproducing that by hand to
    line up a parallel array is the kind of implicit coupling that breaks silently. A merge on
    ``(unique_id, ds)`` is order-independent.
    """
    idx = long_df[[NF_ID, NF_TIME, HOUR_IDX]]
    merged = design[[NF_ID, NF_TIME]].merge(idx, on=[NF_ID, NF_TIME], how="left")
    return merged[HOUR_IDX].to_numpy(dtype=float) - design["horizon_step"].to_numpy(dtype=float)


def recency_weights(
    design: pd.DataFrame, long_df: pd.DataFrame, cut_idx: int, halflife: float
) -> np.ndarray:
    """Plan 4.5a: exponential decay in the AGE OF THE ORIGIN, so recent regime info dominates.

    ``w = 0.5 ** (age / halflife)`` with ``age = (cut - 1) - origin``. Anchored to the origin, not
    the forecast hour: every row from one origin describes the same "what did the world look like
    when this forecast was issued", and it is that vintage which goes stale under macro drift
    (the July shift).

    This is the more direct answer to drift than ``linear_tree`` was — and unlike ``linear_tree``
    it composes with the L1 objective cleanly, because it is a sample weight rather than a leaf
    model. **It matters more once 4.4 lands**: denser origins pull in proportionally more *old*
    data, so stride and half-life want tuning together rather than in sequence.

    A row whose origin cannot be resolved (its forecast hour is off the end of the frame) keeps
    weight 1.0 rather than silently dropping out of the objective.
    """
    age = (cut_idx - 1) - origin_hidx(design, long_df)
    w = np.power(0.5, age / float(halflife))
    return np.where(np.isfinite(w), w, 1.0)


def _combine(base, extra):
    """Multiply weights together. Weighting schemes stack; none of them replaces another."""
    return extra if base is None else np.asarray(base, dtype=float) * extra


def _scaled_target(design, y, levels, weight=None):
    """``(y/level, weight*level)`` — plan 4.6's normalise-AND-weight pair, applied together.

    They are returned together because applying either alone is a different experiment. Dividing
    the target without the weight de-weights exactly the high-volume units WAPE cares about most;
    the weight without the division is the identity. Any caller-supplied weight multiplies through,
    so recency weighting (4.5a) composes with this rather than competing with it.
    """
    lv = _level_of(design, levels)
    w = lv if weight is None else np.asarray(weight, dtype=float) * lv
    return y / lv, w


def _series_views(long_df: pd.DataFrame, futr_cols: list[str], stat_cols: list[str]) -> dict:
    """Per-series numpy views keyed by series id (``_hidx`` is a 0-based contiguous index)."""
    views: dict[str, dict] = {}
    for sid, g in long_df.groupby(NF_ID, sort=False):
        g = g.sort_values(HOUR_IDX)
        views[sid] = {
            "y": g[NF_TARGET].to_numpy(dtype=float) if NF_TARGET in g else None,
            "ds": g[NF_TIME].to_numpy(),
            "futr": g[futr_cols].to_numpy(dtype=float),
            "stat": g[stat_cols].iloc[0].to_numpy(dtype=float) if stat_cols else np.empty(0),
            "n": len(g),
        }
    return views


def build_design(
    long_df: pd.DataFrame,
    pairs: pd.DataFrame,
    futr_cols: list[str] | None = None,
    stat_cols: list[str] | None = None,
    with_target: bool = True,
    fc_lags: list[int] | None = None,
    categorical_unit: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Materialise feature rows for explicit (series, origin, step) triples.

    ``pairs`` columns: ``unique_id``, ``o`` (origin ``_hidx``), ``k`` (step >= 1). The forecast
    hour is ``o + k``. Returns ``(design_df, y)`` where ``design_df`` carries the id/time columns
    plus ``feature_columns(...)`` and ``y`` is the target at the forecast hour (or None).

    ``fc_lags`` adds forecast-hour-anchored lags ``y[fc - l]`` alongside the origin-anchored ones
    (see ``WEEKLY_LAGS``); empty by default, so the recorded numbers are untouched.

    ``long_df`` must already be imputed and carry ``_hidx`` (added here if missing).
    """
    futr_cols = futr_cols if futr_cols is not None else futr_exog_list()
    stat_cols = stat_cols if stat_cols is not None else stat_exog_list()
    fc_lags = validate_fc_lags(fc_lags)
    if HOUR_IDX not in long_df.columns:
        long_df = add_hour_index(long_df)
    views = _series_views(long_df, futr_cols, stat_cols)
    codes = unit_codes(long_df) if categorical_unit else {}

    rows, id_out, ds_out, y_out = [], [], [], []
    n_lag, n_wlag, n_roll = len(LAGS), len(fc_lags), len(ROLLS)
    n_futr, n_stat = len(futr_cols), len(stat_cols)
    width = n_lag + n_wlag + n_roll + 1 + n_futr + n_stat + (1 if categorical_unit else 0)

    for sid, grp in pairs.groupby(NF_ID, sort=False):
        v = views.get(sid)
        if v is None:
            continue
        y, futr, stat, n = v["y"], v["futr"], v["stat"], v["n"]
        o = grp["o"].to_numpy(dtype=int)
        k = grp["k"].to_numpy(dtype=int)
        fc = o + k  # forecast hour index
        m = len(grp)
        block = np.full((m, width), np.nan, dtype=float)

        # origin-anchored lags: y[o-(lag-1)] when in range
        for j, lag in enumerate(LAGS):
            src = o - (lag - 1)
            ok = src >= 0
            block[ok, j] = y[src[ok]]
        # forecast-hour-anchored ("weekly-aligned") lags: y[fc-lag], varying across the horizon.
        # `validate_fc_lags` has already established lag >= horizon, so `src <= o` and no value
        # here can come from after the origin; the `src >= 0` guard is the series-start boundary,
        # mirroring the origin-anchored block above.
        for j, lag in enumerate(fc_lags):
            src = fc - lag
            ok = src >= 0
            block[ok, n_lag + j] = y[src[ok]]
        # rolling means over the last `win` observed hours up to and including the origin
        for j, win in enumerate(ROLLS):
            col = n_lag + n_wlag + j
            for r in range(m):
                lo = max(0, o[r] - win + 1)
                hi = o[r] + 1
                if hi > lo:
                    block[r, col] = float(np.mean(y[lo:hi]))
        # horizon step
        block[:, n_lag + n_wlag + n_roll] = k
        # known-future covariates read at the forecast hour
        fc_ok = fc < n
        futr_start = n_lag + n_wlag + n_roll + 1
        if n_futr:
            block[fc_ok, futr_start : futr_start + n_futr] = futr[fc[fc_ok]]
        # statics (constant per series)
        if n_stat:
            block[:, futr_start + n_futr : futr_start + n_futr + n_stat] = stat
        if categorical_unit:
            block[:, -1] = float(codes[sid])

        rows.append(block)
        id_out.append(np.full(m, sid, dtype=object))
        ds = np.full(m, np.datetime64("NaT"), dtype=v["ds"].dtype)
        ds[fc_ok] = v["ds"][fc[fc_ok]]
        ds_out.append(ds)
        if with_target and y is not None:
            yt = np.full(m, np.nan)
            yt[fc_ok] = y[fc[fc_ok]]
            y_out.append(yt)

    if not rows:
        cols = [NF_ID, NF_TIME, *feature_columns(futr_cols, stat_cols, fc_lags, categorical_unit)]
        return pd.DataFrame(columns=cols), (None if not with_target else np.array([]))

    X = np.vstack(rows)
    design = pd.DataFrame(
        X, columns=feature_columns(futr_cols, stat_cols, fc_lags, categorical_unit)
    )
    design.insert(0, NF_TIME, np.concatenate(ds_out))
    design.insert(0, NF_ID, np.concatenate(id_out))
    y_arr = np.concatenate(y_out) if (with_target and y_out) else None
    return design, y_arr


def training_pairs(long_df: pd.DataFrame, cut_idx: int, horizon: int, stride: int) -> pd.DataFrame:
    """Rolling-origin training grid: origins strided back from ``cut-1``; steps 1..min(h, cut-1-o).

    Only forecast hours strictly inside the train region (``o+k <= cut-1``) become samples, so the
    target is always observed and leakage-free. Striding the origin keeps the row count tractable
    while every origin still spans up to ``horizon`` consecutive hours (full weekly cycles).
    """
    rows = []
    for sid, g in long_df.groupby(NF_ID, sort=False):
        n = int(g[HOUR_IDX].max()) + 1 if HOUR_IDX in g else len(g)
        last = min(cut_idx - 1, n - 1)
        origins = range(last, 0, -stride)
        for o in origins:
            kmax = min(horizon, (cut_idx - 1) - o)
            if kmax < 1:
                continue
            ks = np.arange(1, kmax + 1)
            rows.append(pd.DataFrame({NF_ID: sid, "o": o, "k": ks}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[NF_ID, "o", "k"])


def inference_pairs(
    long_df: pd.DataFrame, cut_idx: int, horizon: int, first_step: int = 1
) -> pd.DataFrame:
    """Inference grid: one origin ``cut-1`` per series, steps ``first_step``..horizon.

    ``first_step`` exists for the early-stopping fold, which wants only the scored-block tail of the
    horizon (steps 337-672) rather than the gap as well.
    """
    rows = []
    for sid in long_df[NF_ID].unique():
        ks = np.arange(first_step, horizon + 1)
        rows.append(pd.DataFrame({NF_ID: sid, "o": cut_idx - 1, "k": ks}))
    return pd.concat(rows, ignore_index=True)


def early_stopping_pairs(
    long_df: pd.DataFrame,
    cut_idx: int,
    horizon: int = GAPPED_H,
    score_len: int = SCORE_LEN,
    stride: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/validation origin grids for an honest early-stopping fold, entirely inside train.

    ``num_boost_round: 600`` was hand-picked with no validation set at all. That number is a
    property of the *current* feature set, objective and weighting — change any of them and it is
    stale, so carrying it unchanged into every Phase-4 A/B would confound each lever with a wrong
    round count. This fold gives the round count something to be measured against.

    The fold **mirrors the graded task rather than being a random split**, because a random split
    over a rolling-origin design leaks: the same forecast hour appears under many origins, so a
    randomly held-out row almost always has near-duplicates in train and early stopping would run
    far past the honest optimum. Instead:

      * **validation** = one origin per series at ``cut - 1 - horizon``, evaluated only on steps
        ``horizon - score_len + 1 .. horizon`` — i.e. the 336 hours ``[cut-336, cut)``, forecast
        from 672h back across a gap. That is exactly the shape of the real scored block.
      * **training** = the ordinary rolling grid with the cutoff pulled back to ``cut - score_len``,
        so every training target lands strictly before the validation block.

    Leakage check on the features: the validation origin is ``cut-1-horizon``, so its lags and
    rolling means read ``y`` no later than ``cut-673`` — well before the validation targets begin at
    ``cut-336``. Nothing in the fold sees its own answer.

    Costs 336 hours of training targets, which is why the caller refits on the full train region
    once the round count is known.
    """
    val_boundary = cut_idx - score_len
    tr = training_pairs(long_df, val_boundary, horizon, stride)
    va = inference_pairs(long_df, cut_idx - horizon, horizon, first_step=horizon - score_len + 1)
    return tr, va


def fit_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict | None = None,
    num_boost_round: int = 0,
    sample_weight: np.ndarray | None = None,
    valid_data: tuple | None = None,
    early_stopping_rounds: int = 0,
):
    """Train a LightGBM regressor on the design matrix (feature columns only).

    ``sample_weight`` is passed straight to ``lgb.Dataset(weight=...)``, which scales each row's
    contribution to the objective.

    ``valid_data`` is ``(X_valid, y_valid[, weight_valid])``. Supplying it together with
    ``early_stopping_rounds > 0`` stops boosting when the validation L1 has not improved for that
    many rounds and leaves the round count on ``booster.best_iteration``. Build the fold with
    :func:`early_stopping_pairs` — a random split over a rolling-origin design leaks badly, since
    the same forecast hour recurs under many origins.

    **Read this before reaching for it.** Unlike the neuralforecast members, the tree needs no
    weighting to match the graded metric. Pooled WAPE is ``sum|y-yhat| / sum|y|`` and the
    denominator does not depend on the model, so minimising ``sum|y-yhat|`` *is* minimising WAPE.
    ``objective="regression_l1"`` on the raw target minimises exactly that, and this design matrix
    carries no per-series scaler to undo (every series contributes equally many rows, so the
    training distribution already matches the scoring distribution). The tree's loss is aligned
    out of the box.

    The neural members differ only because neuralforecast trains inside a per-series robust
    (MAD) scaler, so its residuals are implicitly divided by MAD and plain MAE equalises series
    regardless of volume — which is what ``add_volume_sample_weight`` exists to undo.
    Applying those same MAD weights here would *break* an already-correct objective by
    over-weighting high-variance series beyond their share of the WAPE numerator.

    The hook is here for the cases where weighting genuinely *is* needed:

    **Per-unit target normalisation — normalise the target AND weight by level.** These are
    separable concerns that fix different problems, and together they are the correct
    parameterisation rather than a compromise.

    *The representation problem.* 96 units span wildly different scales. Trees split on absolute
    thresholds, so an unnormalised target makes the ensemble burn its early splits separating units
    by level instead of learning temporal structure. Modelling ``z = y / level`` pools every unit
    onto a common scale, so one tree learns shared shape and every unit's rows contribute to every
    pattern — standard M5 practice for heterogeneous panels.

    *The metric problem, and why it is not a trade-off.* Normalising alone changes what is
    optimised: L1 on ``z`` minimises ``sum (1/level) * |y - yhat|``, de-weighting exactly the
    high-volume units WAPE cares about most. Weighting by level cancels it exactly — with
    ``w = level``, ``z = y/level`` and ``zhat = yhat/level``::

        w * |z - zhat|  =  level * |y - yhat| / level  =  |y - yhat|

    Summed over rows that is precisely the WAPE numerator. So the normalisation fixes the tree's
    representation while the weight restores the metric's emphasis, and neither costs the other
    anything. Requires ``level`` constant per unit (a train-slice statistic, e.g. the per-unit
    median) — a time-varying level would not cancel.

    Measure ``(raw target, unweighted)`` against ``(y/level, sample_weight=level)``: both are
    WAPE-aligned, so the comparison isolates the representation gain. ``(y/level, unweighted)`` is
    a third, misaligned thing that will flatter itself on anything but pooled WAPE — worth naming
    so it is not mistaken for the normalisation result.

    **Log transform: don't.** L1 on ``log y`` minimises ``sum|log y - log yhat|``, which is a
    relative-error criterion (MAPE-like), not WAPE. Restoring alignment would need weights
    proportional to ``y`` itself — unstable across a 0.16-53 target range — so there is no clean
    correction. The strictly-positive target makes the transform *available*, not advisable.

    Also fine: horizon or recency weighting, and ablations that quantify the above rather than
    assuming it.
    """
    import lightgbm as lgb

    p = {**DEFAULT_PARAMS, **(params or {})}
    # Pop UNCONDITIONALLY, before choosing. `lgb.train` treats a round count left in `params` as
    # authoritative and silently ignores its own `num_boost_round` argument — so the old
    # short-circuit (`num_boost_round or p.pop(...)`) meant an explicit round count was discarded
    # whenever the config also set one, which configs/lgbm.yaml does (600). Harmless while every
    # caller passed 0; not harmless now that early stopping passes a measured best_iteration.
    from_params = p.pop("num_boost_round", None)
    fallback = from_params if from_params is not None else DEFAULT_NUM_BOOST_ROUND
    nbr = num_boost_round or int(fallback)
    feat = [c for c in X.columns if c not in (NF_ID, NF_TIME)]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(sample_weight) != len(X):
            raise ValueError(
                f"sample_weight has {len(sample_weight)} rows, design matrix has {len(X)}"
            )
    # `params` must also reach the Dataset: several options (notably `linear_tree`, `max_bin` and
    # the binning controls) are consumed at construction time, not at train time. Passing them only
    # to `lgb.train` leaves the dataset built under defaults and the option partially applied.
    # A unit code is a label, not a magnitude. Left numeric, LightGBM would split it with `<=` and
    # impose an ordering on 96 unrelated units; declared categorical, it partitions the set instead.
    cats = [UNIT_COL] if UNIT_COL in feat else "auto"
    dset = lgb.Dataset(
        X[feat],
        label=y,
        weight=sample_weight,
        params=p,
        free_raw_data=False,
        categorical_feature=cats,
    )

    kwargs: dict = {}
    if valid_data is not None and early_stopping_rounds > 0:
        Xva, yva = valid_data[0], valid_data[1]
        wva = valid_data[2] if len(valid_data) > 2 else None
        kwargs["valid_sets"] = [
            lgb.Dataset(
                Xva[feat],
                label=yva,
                weight=wva,
                params=p,
                reference=dset,
                free_raw_data=False,
                categorical_feature=cats,
            )
        ]
        kwargs["valid_names"] = ["valid"]
        kwargs["callbacks"] = [lgb.early_stopping(int(early_stopping_rounds), verbose=False)]

    return lgb.train(p, dset, num_boost_round=nbr, **kwargs), feat


def tune_num_boost_round(
    long_df: pd.DataFrame,
    cut_idx: int,
    params: dict | None = None,
    horizon: int = GAPPED_H,
    score_len: int = SCORE_LEN,
    origin_stride: int = 168,
    max_boost_round: int = 3000,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    sample_weight_fn=None,
    fc_lags: list[int] | None = None,
    categorical_unit: bool = False,
    normalise_level: bool = False,
    recency_halflife: float = 0.0,
) -> dict:
    """Find the honest round count on a held-out fold inside the train region.

    Returns a report dict — ``best_iteration`` plus the diagnostics needed to tell a real optimum
    from a truncated one. **Check ``hit_ceiling``**: if boosting ran out of rounds before the
    validation L1 plateaued, ``best_iteration`` is a floor, not an optimum, and raising
    ``max_boost_round`` is the fix.

    Never touches the scored block: the whole fold lives at ``_hidx < cut_idx``.
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
        # Same reference point (cut_idx) in both halves of the fold and in the refit, so the fold
        # ranks round counts under the weighting the refit will actually use.
        wtr = _combine(wtr, recency_weights(Xtr, long_df, cut_idx, recency_halflife))
        wva = _combine(wva, recency_weights(Xva, long_df, cut_idx, recency_halflife))
    if normalise_level:
        # The fold has to be scored on the SAME scale the refit will be, or `best_iteration` is
        # tuned against a different objective from the one that finally runs.
        levels = unit_levels(long_df, cut_idx)
        ytr, wtr = _scaled_target(Xtr, ytr, levels, wtr)
        yva, wva = _scaled_target(Xva, yva, levels, wva)
    booster, _ = fit_lgbm(
        Xtr,
        ytr,
        params,
        num_boost_round=max_boost_round,
        sample_weight=wtr,
        valid_data=(Xva, yva, wva),
        early_stopping_rounds=early_stopping_rounds,
    )
    best = int(booster.best_iteration or max_boost_round)
    return {
        "best_iteration": best,
        "best_score": float(booster.best_score["valid"]["l1"])
        if booster.best_score
        else float("nan"),
        "max_boost_round": int(max_boost_round),
        "early_stopping_rounds": int(early_stopping_rounds),
        "n_train_rows": int(len(Xtr)),
        "n_valid_rows": int(len(Xva)),
        "valid_block": [int(cut_idx - score_len), int(cut_idx)],
        # True = boosting ran out of rounds before the validation loss plateaued, so
        # `best_iteration` is a floor rather than an optimum. Raise max_boost_round and re-run.
        "hit_ceiling": bool(best >= max_boost_round - early_stopping_rounds),
    }


def fit_gapped(
    long_df: pd.DataFrame,
    cut_idx: int,
    params: dict | None = None,
    horizon: int = GAPPED_H,
    origin_stride: int = 168,
    num_boost_round: int = 0,
    sample_weight_fn=None,
    fc_lags: list[int] | None = None,
    categorical_unit: bool = False,
    normalise_level: bool = False,
    recency_halflife: float = 0.0,
):
    """Fit the booster on the train region. Returns ``(booster, feature_names, levels)``.

    Split out of :func:`predict_gapped` for the **submission path**, which must fit once now and
    forecast later from a persisted booster: the graded run gets an input dir with covariates and a
    forecast index and **no training data**, so there is nothing to refit from at inference.

    `predict_gapped` is now this plus :func:`forecast_gapped`, so the CV numbers cannot drift away
    from what ships — there is one implementation, not two.
    """
    long_df = add_hour_index(long_df)
    futr_cols, stat_cols = futr_exog_list(), stat_exog_list()

    tr_pairs = training_pairs(long_df, cut_idx, horizon, origin_stride)
    Xtr, ytr = build_design(
        long_df, tr_pairs, futr_cols, stat_cols, True, fc_lags, categorical_unit
    )
    ok = ~np.isnan(ytr)
    Xtr, ytr = Xtr.loc[ok].reset_index(drop=True), ytr[ok]
    # ``sample_weight_fn(Xtr, ytr) -> weights``. Default None = unweighted, which is already the
    # WAPE-aligned objective for the tree (see fit_lgbm's docstring); the hook exists for ablations
    # and for variants that normalise the target per unit.
    weights = sample_weight_fn(Xtr, ytr) if sample_weight_fn is not None else None
    if recency_halflife > 0:
        weights = _combine(weights, recency_weights(Xtr, long_df, cut_idx, recency_halflife))
    levels = unit_levels(long_df, cut_idx) if normalise_level else None
    if normalise_level:
        ytr, weights = _scaled_target(Xtr, ytr, levels, weights)
    booster, feat = fit_lgbm(Xtr, ytr, params, num_boost_round, sample_weight=weights)
    return booster, feat, levels


def forecast_gapped(
    long_df: pd.DataFrame,
    cut_idx: int,
    booster,
    feat: list[str],
    horizon: int = GAPPED_H,
    fc_lags: list[int] | None = None,
    categorical_unit: bool = False,
    levels=None,
) -> pd.DataFrame:
    """Forecast the whole ``horizon`` from a fitted booster. Returns the design + an ``lgbm`` col.

    **This is a DIRECT multi-horizon forecast, not a recursive roll** — one ``booster.predict`` over
    every (origin, horizon_step) pair at once. The lags are origin-anchored (``y[o-(lag-1)]``), so
    every feature is available from the observed history and nothing needs a previous prediction
    fed back in. That is what makes the tree cheap to ship: at inference it is a design matrix and
    a single predict call.
    """
    long_df = add_hour_index(long_df)
    futr_cols, stat_cols = futr_exog_list(), stat_exog_list()

    inf_pairs = inference_pairs(long_df, cut_idx, horizon)
    Xinf, _ = build_design(
        long_df, inf_pairs, futr_cols, stat_cols, False, fc_lags, categorical_unit
    )
    preds = booster.predict(Xinf[feat])
    if levels is not None:
        # Back to the target's own scale. The model predicted z = y/level, so the member's output
        # column stays in the same units as every other member and the merge below is unaffected.
        preds = preds * _level_of(Xinf, levels)
    return Xinf.assign(lgbm=np.clip(preds, 0.0, None))


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
    """Fit on the train region (``_hidx < cut``) and forecast the scored block (last ``score_len``).

    Returns ``[unique_id, ds, y, lgbm]`` for the scored block only (final ``score_len`` of the
    ``horizon``-hour forecast), mirroring ``src.eval.cv.run_gapped_eval`` so the LGBM member
    lands on the exact same rows as the neuralforecast members.

    With ``early_stopping_rounds > 0`` the round count is **measured rather than assumed**: a
    held-out fold inside the train region (see :func:`early_stopping_pairs`) picks
    ``best_iteration``, then the model is **refit on the full train region** for that many rounds.
    The refit matters — the fold costs 336 hours of the most recent training targets, and on a
    drifting series those are the ones worth keeping. Pass a dict as ``tuning_report`` to receive
    the fold's diagnostics; check its ``hit_ceiling`` flag before trusting the count.

    ``normalise_level`` is plan 4.6: model ``z = y/level``, weight by ``level``, multiply back. The
    two halves are applied together by :func:`_scaled_target` because either alone is a different
    experiment — see :func:`unit_levels` for why the level must be constant per unit.

    ``regime`` picks which half of the ``horizon``-hour forecast is returned: ``far`` (the last
    ``score_len`` steps — the graded private-test block, and the default that reproduces every
    recorded number) or ``near`` (the first ``score_len``, the leaderboard's scenario). It is a
    slice of the SAME booster and the SAME inference grid, so a near cube costs no extra fit.

    Default is off for every lever, which reproduces the recorded numbers exactly.
    """
    if regime not in BLOCK_REGIMES:
        raise ValueError(f"regime must be one of {BLOCK_REGIMES}, got {regime!r}")
    long_df = add_hour_index(long_df)

    if early_stopping_rounds > 0:
        report = tune_num_boost_round(
            long_df,
            cut_idx,
            params=params,
            horizon=horizon,
            score_len=score_len,
            origin_stride=origin_stride,
            max_boost_round=max_boost_round,
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

    booster, feat, levels = fit_gapped(
        long_df,
        cut_idx,
        params=params,
        horizon=horizon,
        origin_stride=origin_stride,
        num_boost_round=num_boost_round,
        sample_weight_fn=sample_weight_fn,
        fc_lags=fc_lags,
        categorical_unit=categorical_unit,
        normalise_level=normalise_level,
        recency_halflife=recency_halflife,
    )
    Xinf = forecast_gapped(
        long_df,
        cut_idx,
        booster,
        feat,
        horizon=horizon,
        fc_lags=fc_lags,
        categorical_unit=categorical_unit,
        levels=levels,
    )

    # scored block = the final `score_len` steps of the horizon (horizon_step carries k). Match
    # labels by the scored rows' own `ds` against the FULL label frame — the scored block is
    # [cut+horizon-score_len, cut+horizon), which only coincides with the global tail when
    # cut == n-horizon (W0). A `tail(score_len)` label slice silently drops every earlier cutoff
    # to 0 merged rows; merging on the real timestamps is cutoff-agnostic.
    keep = (
        Xinf["horizon_step"] > horizon - score_len
        if regime == "far"
        else Xinf["horizon_step"] <= score_len
    )
    scored = Xinf[keep].copy()
    labels = long_df[[NF_ID, NF_TIME, NF_TARGET]]
    out = scored[[NF_ID, NF_TIME, "lgbm"]].merge(labels, on=[NF_ID, NF_TIME], how="inner")
    return out[[NF_ID, NF_TIME, NF_TARGET, "lgbm"]]
