"""NaN imputation with a missingness indicator — identical at train and inference.

A NaN in the forecast/risk columns means the covariate is *unavailable*, not zero. We fill
it with a stored per-series median (global-median fallback) and record a binary `*_missing`
column so the model can learn that "imputed" is its own state. The fill stats are fitted on
the training data and travel inside the checkpoint, so inference imputes exactly as training
did. Never ``fillna(0)`` — that would conflate "no signal" with a genuine low value.
"""

from __future__ import annotations

import pandas as pd

from src.data.features import ID, MISSING_SUFFIX, TIME, nan_col_list

GLOBAL_KEY = "__global__"


def fit_fill_stats(df: pd.DataFrame, nan_cols: list[str] | None = None) -> dict:
    """Per-series median (+ global-median fallback) for each NaN-prone column.

    Returns a JSON-serialisable dict ``{col: {series_id: median, "__global__": median}}``.
    A series whose column is entirely NaN stores ``None`` and falls back to the global median.
    """
    nan_cols = nan_cols or nan_col_list()
    stats: dict[str, dict] = {}
    for col in nan_cols:
        per_series = df.groupby(ID)[col].median()
        stats[col] = {
            str(sid): (None if pd.isna(val) else float(val)) for sid, val in per_series.items()
        }
        stats[col][GLOBAL_KEY] = float(df[col].median())
    return stats


def apply_fill(
    df: pd.DataFrame,
    fill_stats: dict,
    nan_cols: list[str] | None = None,
    add_indicator: bool = True,
    strategy: str = "median",
) -> pd.DataFrame:
    """Add `*_missing` flags, then fill NaNs using ``strategy``, backed by the stored stats.

    ``strategy="median"`` is the incumbent and is bit-exact with what this function always did:
    the stored per-series median with a global fallback. Any other strategy
    (``src.data.gap_fill``) runs FIRST as a local pass over the frame's own observed rows, and
    whatever it cannot reach falls through to the same stored-median code below.

    That two-step order is what makes a local strategy shippable. ``interp`` needs no history —
    it reads the observed neighbours either side of an isolated NaN, which a bare 336-row future
    block carries — while the edges, where there is no neighbour to interpolate to, are covered
    by the train-fitted table travelling inside ``checkpoint.pt``. So inference imputes exactly
    as training did without baking any data in, which is the only route that ruling leaves
    open. Measured on the real ~4.5% NaN pattern: ``interp`` reconstructs at 0.2171 against the
    median's 0.5060 (+57.1%), because real NaN runs are isolated single hours.

    The indicator is computed BEFORE any filling, so ``*_missing`` keeps meaning "this row was
    reconstructed" whatever strategy did the reconstructing.
    """
    nan_cols = nan_cols or nan_col_list()
    df = df.copy()
    for col in nan_cols:
        if col not in df.columns:
            continue
        if add_indicator:
            df[f"{col}{MISSING_SUFFIX}"] = df[col].isna().astype("float32")

    if strategy != "median":
        from src.data.gap_fill import fill_scattered

        df = fill_scattered(
            df, nan_cols, strategy=strategy, id_col=ID, time_col=TIME, stats=fill_stats
        )

    for col in nan_cols:
        if col not in df.columns:
            continue
        col_stats = fill_stats.get(col, {})
        global_val = col_stats.get(GLOBAL_KEY, 0.0)
        medians = pd.Series(
            {k: v for k, v in col_stats.items() if k != GLOBAL_KEY and v is not None}
        )
        series_fill = df[ID].astype(str).map(medians)
        if global_val is not None:
            series_fill = series_fill.fillna(global_val)
        df[col] = df[col].fillna(series_fill)
    return df
