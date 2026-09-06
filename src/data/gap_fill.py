"""Named, pluggable reconstruction strategies for the TWO covariate-imputation surfaces.

The project has been treating "imputation" as one thing. It is two, with different code,
different best strategies and a different blast radius (plan S3.0, Finding 2):

===================  ==========================================  =========================
surface              where                                       who it reaches
===================  ==========================================  =========================
the **336h gap**     ``chronos2_eval._withhold_gap_covariates``   nf members + ``chronos_ft``
the **scattered      ``src.data.impute.apply_fill``               EVERY member, incl. the tree
NaNs** (~4.5%)
===================  ==========================================  =========================

They are genuinely different problems and the measured winners differ. Scattered NaNs are
isolated single hours (run length: mean 1.05, median 1, p95 1, max 4) with observed neighbours
on *both* sides, so linear interpolation wins (+57.1% reconstruction WAPE vs the flat median).
The gap is 336 consecutive unobserved hours with no right-hand neighbour at all, so
interpolation is not merely worse there — it is structurally unavailable, and the winner is the
per-series hour-of-week profile (``how168``, +31.2%). Same word, different problem; a strategy
therefore declares which surfaces it supports and asking for one it does not is an error, not a
silent degradation to something adjacent.

The measured negative control is worth keeping in view: ``snaive168`` (tile the last observed
week forward) is *worse* than the flat median (-7.2%). A single week carries its own noise and
the median over ~20 weeks denoises it. The lever is the weekly **profile**, not weekly
persistence.

THE CONTRACT every strategy honours
-----------------------------------
``fn(values, ctx) -> DataFrame`` where ``values`` has already been NaN-masked down to the rows
the strategy is *allowed to learn from* (``ctx.fit``), and the return carries a value for every
row in ``ctx.fill``. Splitting "may learn from" out of "must reconstruct" rather than treating
them as complements is what keeps the gap surface honest: there, the rows after the gap (the
scored block) carry real covariates that must be preserved and must NOT be visible to the
strategy fitting the gap. A local operation like ``ffill`` cannot peek at what it is meant to
reconstruct because the values simply are not there.

``+exact`` — a modifier, not a strategy
---------------------------------------
``workload_intensity`` is an EXACT function of (series, hour-of-week): ``max|delta|`` at lag 168
is ``0.000e+00`` over all 96 series x 4320 hours, exactly 48 distinct values per series, exactly
one per (series, hour-of-week). It carries 0% NaN, so it is not a ``NAN_COL`` and has no
``*_missing`` flag — yet ``_withhold_gap_covariates`` overwrites all 13 ``KNOWN_FUTURE_SIGNALS``
over the gap and flags only the 10 that have flags. For 336 hours the model is handed a wrong
value for a perfectly knowable covariate and is not told. **That is a defect, not a lever**, so
the fix is adopted whatever the WAPE gates say, and it is expressed as a suffix
(``"median+exact"``, ``"how168+exact"``) so the A/B can separate the defect fix from the
strategy change instead of confounding them.

Detection is MEASURED on the fit rows, never hardcoded to a column name: a covariate qualifies
only if every (series, hour-of-week) bin holds exactly one distinct value over enough
observations to mean it. A column that merely *looks* constant on a thin slice does not qualify.
Flags are deliberately left alone — a ``*_missing`` flag says "this row was reconstructed", and
that stays true.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.features import ID, TIME

GAP = "gap"
SCATTERED = "scattered"
EXACT_SUFFIX = "+exact"

# A (series, hour-of-week) bin needs at least this many observations before we are willing to
# call a column deterministic. The CV train slices are 2976h+ (~17 weeks), so a genuine profile
# clears it comfortably while a thin slice cannot manufacture a false positive.
EXACT_MIN_OBS = 3


@dataclass(frozen=True)
class FillContext:
    """Everything a strategy may condition on, aligned row-for-row with ``values``.

    ``fit`` and ``fill`` are NOT complements. On the gap surface ``fit`` is the train slice and
    ``fill`` is the gap block, while the scored block after the gap is neither: its covariates
    are real, must survive untouched, and must stay invisible to the strategy.
    """

    ids: pd.Series  # series id per row
    ts: pd.Series  # timestamp per row
    fit: pd.Series  # rows the strategy may learn from
    fill: pd.Series  # rows the strategy must reconstruct
    stats: dict | None = None  # stored fill stats, for a global fallback at inference
    cut_idx: int | None = None  # the window's cutoff — required by any frame-backed strategy

    def hour_of_week(self) -> pd.Series:
        return self.ts.dt.dayofweek * 24 + self.ts.dt.hour

    def hour_of_day(self) -> pd.Series:
        return self.ts.dt.hour


StrategyFn = Callable[[pd.DataFrame, FillContext], pd.DataFrame]


@dataclass(frozen=True)
class FillStrategy:
    name: str
    fn: StrategyFn
    surfaces: frozenset[str]
    note: str = ""


_REGISTRY: dict[str, FillStrategy] = {}


def register(strategy: FillStrategy) -> FillStrategy:
    if strategy.name in _REGISTRY:
        raise ValueError(f"fill strategy {strategy.name!r} is already registered")
    unknown = strategy.surfaces - {GAP, SCATTERED}
    if unknown:
        raise ValueError(f"{strategy.name!r}: unknown surface(s) {sorted(unknown)}")
    _REGISTRY[strategy.name] = strategy
    return strategy


def infer_gap_len(last_observed, first_forecast, freq_hours: int = 1) -> int:
    """Hours between the last observed row and the first forecast row — DERIVED, never assumed.

    The CV windows use a 336h gap because that is how they are constructed, but the graded run
    must not bake that in: the specification's *"the timeframe might differ"* is the whole reason
    the imputation route exists at all. So the submission path reads the gap off the two files it is
    given rather than off a constant, and every consumer of this module takes the length as an
    argument.

    Returns 0 for a contiguous horizon (the validation scenario, first forecast hour = last
    observed + 1), which is the case a 336-shaped assumption also gets wrong.
    """
    delta = pd.Timestamp(first_forecast) - pd.Timestamp(last_observed)
    hours = delta.total_seconds() / 3600.0
    gap = round(hours / freq_hours) - 1
    if gap < 0:
        raise ValueError(
            f"first forecast {first_forecast} is not after last observed {last_observed} "
            f"(implied gap {gap})"
        )
    return int(gap)


def parse_strategy(spec: str) -> tuple[str, set[str]]:
    """``"chronos2+exact+guard"`` -> ``("chronos2", {"exact", "guard"})``.

    Modifiers are per-COLUMN overrides layered on top of a strategy, and both are decided from
    TRAIN-side properties alone — never from the gap truth, which would be an oracle and would
    not survive to the private test.

    * ``exact`` — a covariate that is a deterministic function of (series, hour-of-week) is looked
      up rather than forecast. Do not predict what you can compute.
    * ``guard`` — a sparse event flag falls back to the plain median. Measured: Chronos-2 scores
      **-34.2%** against the median on ``maintenance_known``, because a forecaster asked for a
      column that is almost always exactly 0 will hallucinate small non-zero values, while the
      median is 0 and is therefore right almost everywhere. Sparsity is visible in the train slice,
      so this needs no knowledge of the block being reconstructed.
    """
    parts = spec.split("+")
    return parts[0], set(parts[1:])


def get_strategy(spec: str, surface: str) -> FillStrategy:
    """Look up a strategy and REFUSE one that does not support ``surface``.

    Refusing matters: ``interp`` on the gap surface would not error, it would quietly become a
    forward-fill (there is no right-hand anchor across 336 unobserved hours), and a silently
    degraded arm reports a number for a strategy that never ran.
    """
    name, _mods = parse_strategy(spec)
    if name not in _REGISTRY:
        raise KeyError(f"unknown fill strategy {name!r}; registered: {available_strategies()}")
    strat = _REGISTRY[name]
    if surface not in strat.surfaces:
        raise ValueError(
            f"fill strategy {name!r} does not support the {surface!r} surface "
            f"(supports {sorted(strat.surfaces)}). {strat.note}"
        )
    return strat


def available_strategies(surface: str | None = None) -> list[str]:
    return sorted(n for n, s in _REGISTRY.items() if surface is None or surface in s.surfaces)


# --------------------------------------------------------------------------- helpers


def _global_median(values: pd.DataFrame, ctx: FillContext) -> pd.Series:
    """Per-column global median of the fit rows, with the stored stats as a last resort.

    At inference the frame is a bare future block, so its own rows can be entirely NaN for a
    column; ``ctx.stats`` is the train-fitted table that travels inside ``checkpoint.pt``.
    """
    gmed = values.median()
    if ctx.stats:
        for col in values.columns:
            if pd.isna(gmed.get(col, np.nan)):
                stored = ctx.stats.get(col, {}).get("__global__")
                if stored is not None:
                    gmed[col] = float(stored)
    return gmed.fillna(0.0)


def _per_series_median(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    """Broadcast the per-series median of the fit rows back over every row."""
    med = values.groupby(ctx.ids).transform("median")
    gmed = _global_median(values, ctx)
    return med.fillna(gmed)


def _profile_fill(values: pd.DataFrame, ctx: FillContext, bins: pd.Series) -> pd.DataFrame:
    """Per-(series, bin) median of the fit rows, falling back to per-series then global.

    The whole ``how168`` / ``hod24`` family is this function at two bin widths.
    """
    out = values.groupby([ctx.ids, bins]).transform("median")
    return out.fillna(_per_series_median(values, ctx))


# --------------------------------------------------------------------------- strategies


def _median(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    return _per_series_median(values, ctx)


def _how168(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    return _profile_fill(values, ctx, ctx.hour_of_week())


def _hod24(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    return _profile_fill(values, ctx, ctx.hour_of_day())


def _how168_lvl(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    """``how168`` rescaled by the ratio of the recent level to the profile's own level.

    The idea is that a weekly shape is stable while its level drifts. Measured, it buys +0.4%
    over the flat median against ``how168``'s +31.2% — the shape is the signal and the level
    correction is noise. Kept because a measured near-null is a result the write-up can use.
    """
    profile = _how168(values, ctx)
    # The last observed week per series. Rank over the FIT rows only: ranking over every row puts
    # the (masked, all-NaN) target rows at the top, the mean comes back NaN, the ratio collapses
    # to 1.0 and the whole correction silently becomes a no-op that reports how168's number under
    # a different name. Measured that failure once — it is why this ranks `ts.where(ctx.fit)`.
    fit_ts = ctx.ts.where(ctx.fit)
    order = fit_ts.groupby(ctx.ids).rank(method="first", ascending=False)
    recent = values.where(ctx.fit & (order <= 168), other=np.nan)
    lvl = recent.groupby(ctx.ids).transform("mean")
    base = profile.where(ctx.fit).groupby(ctx.ids).transform("mean")
    ratio = (lvl / base.replace(0.0, np.nan)).fillna(1.0).clip(0.5, 2.0)
    return profile * ratio


def _ffill_decay(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    """Last observed value relaxing toward the per-series median with a 48h half-life.

    A persistence prior that admits it decays. Measured at -1.5% against the flat median: over a
    336h block the last value is stale almost immediately, so this is persistence's fair test
    and it fails it.
    """
    half_life = 48.0
    med = _per_series_median(values, ctx)
    last = values.groupby(ctx.ids).ffill()
    # Hours since the most recent observed row, per series.
    obs_ts = ctx.ts.where(ctx.fit & values.notna().any(axis=1))
    last_ts = obs_ts.groupby(ctx.ids).ffill()
    age = (ctx.ts - last_ts).dt.total_seconds() / 3600.0
    w = np.power(0.5, age.clip(lower=0.0) / half_life)
    w = w.fillna(0.0)
    return med.add((last - med).mul(w, axis=0), fill_value=0.0).fillna(med)


def _snaive168(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    """Tile the last observed week forward — the measured NEGATIVE control.

    Worse than the flat median (-7.2%), which is the point: one week carries its own noise and
    the median over ~20 weeks denoises it. Keeping a losing arm in the registry is what lets the
    write-up say the lever is the weekly *profile* rather than weekly persistence.
    """
    hours = (ctx.ts.astype("int64") // 3_600_000_000_000).astype("int64")
    key = pd.DataFrame({"_id": ctx.ids.to_numpy(), "_h": hours.to_numpy()}, index=values.index)
    obs = values.where(ctx.fit)
    # Latest observed hour per series -> the week that gets tiled.
    last_h = key["_h"].where(obs.notna().any(axis=1)).groupby(key["_id"]).transform("max")
    src = pd.DataFrame(index=values.index, columns=values.columns, dtype="float64")
    lookup = obs.copy()
    lookup["_id"] = key["_id"]
    lookup["_h"] = key["_h"]
    table = lookup.dropna(subset=["_h"]).set_index(["_id", "_h"])
    for col in values.columns:
        ser = table[col].dropna()
        # Walk back in whole weeks until the source hour lands inside the observed region.
        offset = ((key["_h"] - last_h) // 168 + 1).clip(lower=1) * 168
        want = pd.MultiIndex.from_arrays([key["_id"], key["_h"] - offset])
        src[col] = ser.reindex(want).to_numpy()
    return src.fillna(_per_series_median(values, ctx))


def _interp(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    """Linear interpolation between observed neighbours, within a series only.

    ``limit_area="inside"`` is the load-bearing argument: it refuses to extrapolate past the
    first or last observation, so an edge NaN falls through to the stored median rather than
    being invented from one side. Real NaN runs are isolated single hours with observed
    neighbours on both sides (mean run 1.05, max 4), which is exactly the regime this wins in
    (+57.1%) — and exactly what the 336h gap does not offer.
    """
    obs = values.where(ctx.fit)
    return obs.groupby(ctx.ids).transform(
        lambda s: s.interpolate(method="linear", limit_area="inside")
    )


def _ffill(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
    """Carry the last observed value forward within a series (+41.9% on scattered NaNs)."""
    return values.where(ctx.fit).groupby(ctx.ids).ffill()


register(
    FillStrategy(
        "median",
        _median,
        frozenset({GAP, SCATTERED}),
        "The incumbent. Per-series median, global fallback.",
    )
)
register(
    FillStrategy(
        "how168",
        _how168,
        frozenset({GAP, SCATTERED}),
        "Per-series x hour-of-week median. Screen A winner (+31.2%).",
    )
)
register(
    FillStrategy(
        "hod24",
        _hod24,
        frozenset({GAP, SCATTERED}),
        "Per-series x hour-of-day median (+28.6%).",
    )
)
register(
    FillStrategy(
        "how168_lvl",
        _how168_lvl,
        frozenset({GAP}),
        "how168 rescaled by the recent level (+0.4% — the level correction is noise).",
    )
)
register(
    FillStrategy(
        "ffill_decay",
        _ffill_decay,
        frozenset({GAP}),
        "Last value decaying to the median, 48h half-life (-1.5%).",
    )
)
register(
    FillStrategy(
        "snaive168",
        _snaive168,
        frozenset({GAP}),
        "Tile the last observed week — the measured negative control (-7.2%).",
    )
)


def _frame_backed(source: str, template: str):
    """Build a strategy that reads a precomputed per-window forecast of the covariates.

    The values come from a derived frame (``src.models.gap_covariate_oof``) rather than from a
    statistic of the train slice, because producing them needs a GPU and a model. Three guards,
    all fail-closed, because a covariate frame that is wrong scores *better* and so fails in the
    flattering direction:

    * the cutoff must be known — resolving ``{cut}`` without one would hand this window another
      window's forecast, a leak by aliasing and invisible in exactly the way ``member_train_csv``
      was hardened against;
    * the frame's provenance sidecar must certify gap honesty at THIS cutoff;
    * anything the frame does not cover falls back to ``how168``, never to silence.
    """

    def fn(values: pd.DataFrame, ctx: FillContext) -> pd.DataFrame:
        import os

        from src.models import cascade_provenance as cp

        if ctx.cut_idx is None:
            raise ValueError(
                f"the {source!r} fill strategy needs the window's cut_idx to resolve its frame "
                f"({template!r}). Defaulting would silently serve another window's forecast."
            )
        # Same escape hatch shape as CASCADE_CHANNELS: an env var, so a subprocess launch (Modal,
        # the CLI) can point at a scratch directory without a signature change.
        tmpl = os.environ.get("GAP_COVARIATE_FRAME", template)
        path = tmpl.format(cut=int(ctx.cut_idx))
        frame = pd.read_csv(path)
        frame[TIME] = pd.to_datetime(frame[TIME])
        cols = [c for c in values.columns if c in frame.columns]
        horizon = int(ctx.fill.sum() // max(1, ctx.ids[ctx.fill].nunique()))
        for col in cols:
            cp.check_gap_honest(path, col, int(ctx.cut_idx), horizon)

        key = pd.DataFrame({ID: ctx.ids.to_numpy(), TIME: ctx.ts.to_numpy()})
        merged = key.merge(frame[[ID, TIME, *cols]], on=[ID, TIME], how="left")
        out = pd.DataFrame(index=values.index, columns=values.columns, dtype="float64")
        for col in cols:
            out[col] = merged[col].to_numpy()
        return out.fillna(_how168(values, ctx))

    return fn


register(
    FillStrategy(
        "chronos2",
        _frame_backed("chronos2", "data/derived/gapcov_chronos_cut{cut}.csv"),
        frozenset({GAP}),
        "Zero-shot Chronos-2 forecast of each planning signal, 1248 pseudo-series, anchored at "
        "the cutoff (src.models.gap_covariate_oof). Falls back to how168 off-frame.",
    )
)
register(
    FillStrategy(
        "interp",
        _interp,
        frozenset({SCATTERED}),
        "Linear interpolation between observed neighbours. Screen B winner (+57.1%). "
        "Unavailable across the 336h gap: there is no right-hand anchor to interpolate to.",
    )
)
register(
    FillStrategy(
        "ffill",
        _ffill,
        frozenset({SCATTERED}),
        "Forward-fill within a series (+41.9%). Degenerate over a 336h block.",
    )
)


# --------------------------------------------------------------------------- the +exact modifier


def exact_how_columns(
    values: pd.DataFrame,
    ids: pd.Series,
    ts: pd.Series,
    *,
    min_obs: int = EXACT_MIN_OBS,
) -> list[str]:
    """Columns that are an EXACT function of (series, hour-of-week) on the rows given.

    Measured, not asserted: every (series, hour-of-week) bin must hold exactly one distinct
    value, over at least ``min_obs`` observations, with no NaN. A column that merely looks
    constant on a thin slice cannot qualify, which is what keeps this from firing on a genuine
    covariate that happens to repeat.
    """
    how = ts.dt.dayofweek * 24 + ts.dt.hour
    out = []
    for col in values.columns:
        if values[col].isna().any():
            continue
        grp = values[col].groupby([ids, how])
        if grp.nunique().max() == 1 and grp.size().min() >= min_obs:
            out.append(col)
    return out


def sparse_columns(values: pd.DataFrame, ids: pd.Series, mask: pd.Series) -> list[str]:
    """Columns whose PER-SERIES MEDIAN is zero for every series on the fit rows.

    Deliberately a criterion rather than a threshold, and it is the criterion that matches what
    the fallback actually does: when the per-series median is 0 the median fill emits a constant
    zero, so pooled reconstruction WAPE is ``sum|true - 0| / sum|true|`` = **exactly 1.0000**.
    That is the measured signature of the project's two irreducible signals, and it is a ceiling
    a forecaster can only reach by predicting the spikes themselves.

    Measured on the real train slice, this separates cleanly with nothing to tune:

        promotion_intensity   95.1% zeros, per-series median 0 for 100% of series
        maintenance_known     85.8% zeros, per-series median 0 for 100% of series
        the other eleven      per-series median non-zero for EVERY series

    So the guard is a decision to accept the 1.0000 floor on those two rather than risk what
    Chronos-2 actually does to ``maintenance_known``, which is **-34.2%** — it emits small
    non-zero values at nearly every hour, adding error everywhere without recovering the spikes.
    Sparsity is a train-side property, so this survives to the private test unchanged.
    """
    fit = values[mask]
    if fit.empty:
        return []
    med = fit.groupby(ids[mask]).median()
    return [c for c in values.columns if c in med and (med[c] == 0).all()]


def _exact_lookup(values: pd.DataFrame, ctx: FillContext, cols: Sequence[str]) -> pd.DataFrame:
    """Reconstruct ``cols`` exactly from their (series, hour-of-week) value."""
    how = ctx.hour_of_week()
    return values[list(cols)].groupby([ctx.ids, how]).transform("first")


# --------------------------------------------------------------------------- the two hooks


def reconstruct(
    values: pd.DataFrame,
    ctx: FillContext,
    strategy: str,
    surface: str,
) -> pd.DataFrame:
    """Run ``strategy`` over ``values`` and return a frame of fills aligned to ``values``.

    Only rows in ``ctx.fill`` are meaningful in the result. The strategy sees the fit rows'
    values and nothing else — everything outside ``ctx.fit`` is NaN before it is called, so an
    operation that walks the frame cannot read what it is supposed to reconstruct.
    """
    strat = get_strategy(strategy, surface)
    _, mods = parse_strategy(strategy)
    unknown = mods - {"exact", "guard"}
    if unknown:
        raise ValueError(f"unknown modifier(s) {sorted(unknown)} in {strategy!r}")
    masked = values.where(ctx.fit)
    out = strat.fn(masked, ctx)
    out = out.reindex(columns=values.columns)
    # Order matters: `guard` hands a column back to the median, `exact` then computes it outright.
    # A column cannot be both sparse-and-deterministic in practice, but if it were, computed beats
    # fallen-back and `exact` should win.
    if "guard" in mods:
        cols = sparse_columns(masked, ctx.ids, ctx.fit)
        if cols:
            out[cols] = _per_series_median(masked, ctx)[cols]
    if "exact" in mods:
        cols = exact_how_columns(masked[ctx.fit], ctx.ids[ctx.fit], ctx.ts[ctx.fit])
        if cols:
            out[cols] = _exact_lookup(masked, ctx, cols)
    return out.fillna(_global_median(masked, ctx))


def reconstruct_block(
    target: pd.DataFrame,
    history: pd.DataFrame,
    cols: Sequence[str],
    fill_mask: pd.Series,
    *,
    strategy: str = "median",
    id_col: str,
    time_col: str,
    cut_idx: int | None = None,
) -> pd.DataFrame:
    """The GAP surface. Reconstruct ``cols`` over ``fill_mask`` rows of ``target``.

    ``history`` supplies the observed rows (the window's train slice, where covariates are
    real). Rows of ``target`` outside ``fill_mask`` are neither fitted on nor filled: on this
    surface they are the scored block, whose covariates are genuine and must not leak into the
    statistic that reconstructs the gap.
    """
    hist = history[[id_col, time_col, *cols]].copy()
    tgt = target[[id_col, time_col, *cols]].copy()
    frame = pd.concat([hist, tgt], ignore_index=True)
    fit = pd.Series([True] * len(hist) + [False] * len(tgt), index=frame.index)
    fill = pd.Series([False] * len(hist) + list(fill_mask.to_numpy()), index=frame.index)
    ctx = FillContext(
        ids=frame[id_col],
        ts=pd.to_datetime(frame[time_col]),
        fit=fit,
        fill=fill,
        cut_idx=cut_idx,
    )
    out = reconstruct(frame[list(cols)], ctx, strategy, GAP)
    return out.iloc[len(hist) :].set_axis(target.index)


def fill_scattered(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    strategy: str,
    id_col: str,
    time_col: str,
    stats: dict | None = None,
) -> pd.DataFrame:
    """The SCATTERED surface. Fill isolated NaNs in ``cols`` from the frame's own observed rows.

    Returns ``df`` with ``cols`` filled where the strategy could reach; anything it could not
    reach (an edge NaN under ``interp``, an all-NaN series) is left NaN for the caller's stored
    median to back-stop. That split is deliberate — the local operation handles the interior,
    the train-fitted table in ``checkpoint.pt`` handles the edges, and inference needs no
    history to do the first part.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    ordered = out.sort_values([id_col, time_col])
    values = ordered[cols]
    ctx = FillContext(
        ids=ordered[id_col],
        ts=pd.to_datetime(ordered[time_col]),
        fit=values.notna().any(axis=1) | True,  # every row may be learnt from; NaNs mask themselves
        fill=values.isna().any(axis=1),
        stats=stats,
    )
    strat = get_strategy(strategy, SCATTERED)
    _, mods = parse_strategy(strategy)
    filled = strat.fn(values, ctx).reindex(columns=cols)
    if "exact" in mods:
        ex = exact_how_columns(
            values.dropna(), ctx.ids[values.notna().all(axis=1)], ctx.ts[values.notna().all(axis=1)]
        )
        if ex:
            filled[ex] = _exact_lookup(values, ctx, ex)
    for col in cols:
        out.loc[ordered.index, col] = ordered[col].fillna(filled[col])
    return out
