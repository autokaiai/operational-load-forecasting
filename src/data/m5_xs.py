"""The M5 analogue of A9 — the SINGLE definition, in `src/` so the panel and the screen share it.

MOVED HERE FROM the M5 screening script for the same reason `xs_blocks.py` moved out of
`scripts/`: two copies of a feature builder drift, and the number on record then describes neither.
`src/models/addl_panel.py` (the graded M5 panel) and the cheap tree screen both import from here.

THE MAPPING, FIXED BEFORE ANY RUN AND MEASURED ON THE REAL BUNDLE. A9 as shipped is the
capacity-weighted cross-sectional mean of three of OUR known-future covariates. None exist on M5, so
the transferable DEFINITION is "capacity-weighted cross-sectional mean of the known-future
covariates". Applying it to M5 exposes a structural difference rather than an obstacle:

    sell_price     cross-series spread 19.9700   varies per series
    snap_CA/TX/WI  0.0000               DEGENERATE — published per DATE, not per series
    wday/month/is_weekend/event_active  0.0000   DEGENERATE

SEVEN OF EIGHT carry no cross-sectional variation, so their cross-series mean IS the column the
model already has — a guaranteed null. Two channels survive, and they are the M5 A9. On our own
panel three covariates vary per unit per hour; that count is the explanatory variable for the
result (measured: A9 is -0.00611 on the M5 tree, i.e. it does NOT transfer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.models.addl_panel import gapped_split

XS_COLS = ["xs_price_cap", "xs_snap_cap"]


def build_m5_a9(bundle) -> tuple[pd.DataFrame, list[str]]:
    """Attach the M5 analogue of A9. Returns (long_df with the columns, their names)."""
    d = bundle.long_df.copy()
    stat = bundle.static_df.set_index(NF_ID)

    # --- snap_own: the SNAP flag each series actually faces, keyed by its own state.
    # `state_id` is label-encoded in the bundle, so recover the mapping from the raw codes rather
    # than assuming an order. Each state's column is global, so one lookup per (series, date).
    state = stat["state_id"]
    codes = sorted(state.unique())
    snap_cols = ["snap_CA", "snap_TX", "snap_WI"]
    # A SUBSET OF STATES IS LEGAL, and refusing one breaks every smoke. The panel's plumbing smoke
    # draws 12 series, which need not span all three states; the aggregate is still well defined
    # over whichever states are present. What is NOT legal is more states than SNAP columns, which
    # would mean the label encoding does not line up with the calendar and the mapping is wrong.
    if len(codes) > len(snap_cols):
        raise SystemExit(f"[m5-a9] {len(codes)} state codes but only {len(snap_cols)} SNAP columns")
    per_state = {
        c: d.groupby(NF_TIME)[col].first() for c, col in zip(codes, snap_cols, strict=False)
    }
    d["_state"] = d[NF_ID].map(state)
    d["snap_own"] = np.select(
        [d["_state"] == c for c in codes],
        [d[NF_TIME].map(per_state[c]).to_numpy() for c in codes],
        default=np.nan,
    )
    if d["snap_own"].isna().any():
        raise SystemExit("[m5-a9] snap_own has NaN — the state mapping is incomplete")

    # --- capacity = each series' TRAIN-REGION mean of y, fitted strictly `< cut`.
    split = gapped_split(bundle.long_df, bundle.horizon)
    cap = split.train.groupby(NF_ID)[NF_TARGET].mean()
    if (cap <= 0).any():
        cap = cap.clip(lower=1e-6)  # a series with no sales in train must not zero the weights

    n = d[NF_ID].nunique()
    counts = d.groupby(NF_TIME)[NF_ID].nunique()
    if not (counts == n).all():
        raise SystemExit(f"[m5-a9] panel is not a complete {n}-series rectangle")

    names = []
    for src, out in (("sell_price", "xs_price_cap"), ("snap_own", "xs_snap_cap")):
        w = d[NF_ID].map(cap).to_numpy(float)
        v = d[src].to_numpy(float)
        ok = ~np.isnan(v)
        num = pd.Series(np.where(ok, v * w, 0.0)).groupby(d[NF_TIME].to_numpy()).sum()
        den = pd.Series(np.where(ok, w, 0.0)).groupby(d[NF_TIME].to_numpy()).sum()
        cov = (num / den).rename(out)
        # coverage: if the observed share drifts, the column is partly an availability signal
        share = pd.Series(ok.astype(float)).groupby(d[NF_TIME].to_numpy()).mean()
        print(f"[m5-a9] {out}: coverage min {share.min():.3f} max {share.max():.3f}", flush=True)
        d[out] = d[NF_TIME].map(cov)
        names.append(out)
    d = d.drop(columns=["_state", "snap_own"])
    if d[names].isna().any().any():
        raise SystemExit("[m5-a9] aggregate columns contain NaN")
    return d, names
