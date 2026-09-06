"""Build the ACTIVE cross-series aggregate columns onto an already-imputed frame.

ONE implementation, imported by everything that needs it — `predict.py` (the graded archive),
`scripts/submission_cascade.py` (the shipped neural bag) and `scripts/fit_submission_tree.py` (the
shipped tree). A second copy is how the submission quietly stops constructing the feature the way
the member was fitted, and nothing would raise.

TWO RULES, both of which fail SILENTLY if broken:

1. **After imputation, never before.** `data/raw/validation_input.csv` carries 4.4-4.7% NaN in
   exactly the covariates being aggregated, and pandas' `skipna=True` default would average ~92
   series at inference against 96 in training without raising.
2. **On a complete series rectangle.** Every aggregate is a within-hour statistic over all 96
   series, so a frame missing one series at one hour yields a mean over 95 — numerically fine,
   silently different from training, invisible in any output. Asserted below.
"""

from __future__ import annotations

import pandas as pd

from src.data.features import active_aggregate_columns
from src.data.loader import NF_ID, NF_TIME


def attach_active_aggregates(df: pd.DataFrame, *, where: str = "frame") -> pd.DataFrame:
    """Materialise whatever `active_aggregate_columns()` declares. No-op when nothing is active."""
    want = active_aggregate_columns()
    if not want:
        return df

    # IDEMPOTENT, and this is not defensive dressing. `cascade_inference.history_from_bundle`
    # returns the frame the models were FITTED on, and the aggregates were in `futr_exog_list()`
    # at fit time — so they are already baked into it. Rebuilding them there made the `_attach`
    # merge collide with itself and pandas silently produced `xs_cap_*_x` / `xs_cap_*_y`, leaving
    # the requested names absent. The horizon frame genuinely lacks them and still gets built.
    if all(c in df.columns for c in want):
        return df

    from src.data.xs_blocks import BLOCKS, NEEDS_CUT, _zone_labels
    from src.eval.splits import add_hour_index

    counts = df.groupby(NF_TIME)[NF_ID].nunique()
    n_series = int(df[NF_ID].nunique())
    if not (counts == n_series).all():
        bad = counts[counts != n_series]
        raise SystemExit(
            f"[xs] {where} is not a complete {n_series}-series rectangle: {len(bad)} hour(s) carry "
            f"{sorted(bad.unique())} series. Cross-series aggregates cannot be built from it."
        )

    out = add_hour_index(df)
    zones = _zone_labels(out)
    for key, (_, fn) in BLOCKS.items():
        if key in NEEDS_CUT:
            continue  # fitted blocks (PCA, clusters) cannot be rebuilt at submission time
        built, names = fn(out, zones)
        if not set(want).issubset(names):
            continue
        missing = [c for c in want if c not in built.columns]
        if missing:
            raise SystemExit(f"[xs] block {key} did not materialise {missing} on {where}")
        if built[want].isna().any().any():
            raise SystemExit(f"[xs] {key} produced NaN on {where}; aggregate AFTER imputation")
        print(f"[xs] {key}: {len(want)} aggregate column(s) on {where} ({len(built):,} rows)")
        return built
    raise SystemExit(f"[xs] no rebuildable block produces {want}")
