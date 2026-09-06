"""Load the M5 (Walmart) competition data into the cascade's long format.

This is the **primary** external-dataset experiment for the exposé: M5 is covariate-rich
(SNAP benefits, promotions/price, calendar events) so it exercises the same known-future
covariate fusion our TFT/Chronos cascade relies on. The goal is to run the in-repo
architecture (cascade, LSTM, LightGBM, zero-shot Chronos-2) on a genuinely external dataset,
not merely cite one — a hard PDF requirement.

Long format produced (identical contract to ``src.data.loader``):
- ``unique_id`` (str)  -- one series per (item, store)
- ``ds``        (datetime) -- daily timestamp
- ``y``         (float) -- unit sales
- known-future exog (``futr_exog_list``): calendar one-hots + SNAP flags + ``sell_price`` +
  ``event_*`` flags. All are genuinely known ahead of time (the M5 calendar is published for
  the full horizon), mirroring our ``*_forecast`` planning signals.
- static exog (``stat_exog_list``): categorical ids (dept/cat/store/state), label-encoded.

Frequency decision
------------------
M5 is **daily** (1941 train days, 28-day forecast horizon ``h=28``). Rather than upsample to
hourly (which would fabricate a sub-daily pattern M5 does not have), we run M5 **natively at
daily frequency** with a horizon analogous to ours: our task forecasts ``2 x seasonal_period``
(2 x 168h weekly = 336h) at a ``+1 x horizon`` test gap. The daily analogue is ``h=28`` (the
official M5 horizon = 4 weeks = 2 x 14-day cycles) with a 28-day gap, so the gapped split below
trains to ``T - 56``, leaves a 28-day gap unobserved, and scores the final 28 days. This keeps
the "forecast a multiple of the dominant season, offset by one horizon" structure intact.
``freq="D"`` is threaded into the run config so neuralforecast builds daily windows.

Data source
-----------
The staged Parquet written by ``tools/modal_addl_data.py`` from the raw Kaggle M5 release:
``sales_train_evaluation`` melted to long, joined to ``calendar`` (SNAP / events / weekday) and
``sell_prices`` (on store/item/wm_yr_wk). That work is done once, on CPU, at staging time — a
GPU run just reads the Parquet (59,181,090 rows in 85.6 MB, hive-partitioned by ``store_id``).

Resolved via ``src.data.external_paths`` -> ``/data`` on Modal (volume ``tsf-addl-data``) or
``data/external/`` locally. A tiny synthetic fixture exists for CI, but only as an explicit
``synthetic=True`` opt-in — there is no silent fallback (see plan S8.0).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.external_paths import resolve

# neuralforecast long-format column names (match src.data.loader.NF_ID/NF_TIME/NF_TARGET).
NF_ID, NF_TIME, NF_TARGET = "unique_id", "ds", "y"

# M5 daily horizon analogue (official competition horizon = 28 days = 4 weeks).
M5_HORIZON = 28
M5_FREQ = "D"
M5_SEASONALITY = 7  # weekly

# Known-future calendar/price/SNAP covariates. These are all published for the full horizon in
# the M5 calendar, so they are legitimately "known future" exactly like our planning signals.
M5_FUTR_NUMERIC = ["sell_price", "snap_CA", "snap_TX", "snap_WI"]
M5_FUTR_CALENDAR = ["wday", "month", "is_weekend"]
M5_FUTR_EVENT = ["event_active"]  # binary: a named calendar event falls on this day
# Static per-series categoricals (label-encoded to ints for neuralforecast stat_exog).
M5_STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class M5Bundle:
    """Everything a run needs: long target frame, statics, and the exog column lists."""

    long_df: pd.DataFrame
    static_df: pd.DataFrame
    futr_exog: list[str]
    stat_exog: list[str]
    horizon: int = M5_HORIZON
    freq: str = M5_FREQ
    seasonality: int = M5_SEASONALITY
    synthetic: bool = False
    meta: dict = field(default_factory=dict)


def m5_futr_exog_list() -> list[str]:
    """Known-future covariate columns M5 models condition on."""
    return [*M5_FUTR_NUMERIC, *M5_FUTR_CALENDAR, *M5_FUTR_EVENT]


def m5_stat_exog_list() -> list[str]:
    """Static per-series covariate columns (label-encoded)."""
    return list(M5_STATIC_COLS)


def _label_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Turn categorical id columns into contiguous integer codes (neuralforecast stat_exog)."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("category").cat.codes.astype("int64")
    return out


M5_LONG_PARQUET = "m5/derived/m5_long.parquet"
M5_STATIC_PARQUET = "m5/derived/m5_static.parquet"
M5_SUBSET_PARQUET = "m5/derived/m5_long_s2000.parquet"  # precomputed cheap panel
M5_SUBSET_N = 2000
M5_STRATA = ["dept_id", "store_id"]  # 7 departments x 10 stores = 70 cells


def stratified_ids(
    static_df: pd.DataFrame, n_series: int, seed: int = 42, strata: list[str] | None = None
) -> list[str]:
    """Draw ``n_series`` ids spread proportionally across (department, store).

    **The precomputed subset file cannot do this and the difference is not cosmetic.** It was
    built as ``sorted(unique_id)[:2000]``, and M5 ids sort department-first, so those 2000 series
    are 2000 of the 2160 ``FOODS_1`` series — one of *seven* departments. A panel run on it
    supports the claim "we ran the architecture on FOODS_1", not "on M5", and FOODS_1 is not a
    neutral choice: it is the most intermittent department in the corpus (63% zero-sale days
    against the full set's ~68% but at a mean of 1.18 units), so the whole panel would be scored
    on near-degenerate series.

    Proportional allocation with largest-remainder, so the draw matches the corpus mix rather
    than giving every cell an equal share (departments differ ~5.5x in size). Seeded and sorted
    before sampling, so the same ``(n_series, seed)`` always yields the same panel — the subset
    is part of the experiment's identity, exactly like a cutoff or a frozen seed.
    """
    cols = list(strata or M5_STRATA)
    missing = [c for c in cols if c not in static_df.columns]
    if missing:
        raise ValueError(f"static frame lacks strata columns {missing}")
    rng = np.random.default_rng(seed)
    cells = static_df.groupby(cols, sort=True)[NF_ID].apply(lambda s: sorted(s.astype(str)))
    sizes = cells.map(len)
    total = int(sizes.sum())
    if n_series >= total:
        return sorted(static_df[NF_ID].astype(str))

    exact = sizes * (n_series / total)
    take = np.floor(exact).astype(int)
    # Largest remainder, capped by each cell's population, until the quota is met exactly.
    order = np.argsort(-(exact - take).to_numpy(), kind="stable")
    i = 0
    while int(take.sum()) < n_series:
        j = order[i % len(order)]
        if take.iloc[j] < sizes.iloc[j]:
            take.iloc[j] += 1
        i += 1

    picked: list[str] = []
    for cell_ids, k in zip(cells, take, strict=True):
        if k:
            idx = rng.choice(len(cell_ids), size=int(k), replace=False)
            picked.extend(cell_ids[t] for t in sorted(idx))
    return sorted(picked)


def _build_from_parquet(
    n_series: int | None,
    stratify: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Real M5 from the staged Parquet (see plan S8.0 and ``tools/modal_addl_data.py``).

    The Parquet already carries exactly :func:`m5_futr_exog_list` — the melt of the wide sales
    frame plus the calendar and ``sell_prices`` joins were done once, on CPU, at staging time.

    ``stratify`` selects the series across (department, store) via :func:`stratified_ids` and
    therefore has to read the FULL corpus, since the cheap subset file holds only ``FOODS_1``.
    The read is filtered at the pyarrow level so only the drawn series are ever materialised.

    Returns (long_df with futr exog merged, static_df label-encoded, meta). Raises if the corpus
    is missing — there is no synthetic fallback.
    """
    futr_cols = m5_futr_exog_list()
    static_raw = pd.read_parquet(resolve(M5_STATIC_PARQUET))
    static_raw[NF_ID] = static_raw[NF_ID].astype(str)

    keep: list[str] | None = None
    if stratify and n_series is not None:
        keep = stratified_ids(static_raw, n_series, seed=seed)
        src = resolve(M5_LONG_PARQUET)
    elif n_series is not None and n_series <= M5_SUBSET_N:
        try:
            src = resolve(M5_SUBSET_PARQUET)
        except FileNotFoundError:
            src = resolve(M5_LONG_PARQUET)
    else:
        src = resolve(M5_LONG_PARQUET)

    read_kwargs = {"columns": [NF_ID, NF_TIME, NF_TARGET, *futr_cols]}
    if keep is not None:
        read_kwargs["filters"] = [(NF_ID, "in", set(keep))]
    long_df = pd.read_parquet(src, **read_kwargs)
    long_df[NF_TIME] = pd.to_datetime(long_df[NF_TIME])
    long_df[NF_ID] = long_df[NF_ID].astype(str)

    if keep is not None:
        got = set(long_df[NF_ID].unique())
        if got != set(keep):
            raise ValueError(
                f"stratified draw asked for {len(keep)} series, corpus returned {len(got)}"
            )
    elif n_series is not None:
        sel = pd.Index(sorted(long_df[NF_ID].unique()))[:n_series]
        long_df = long_df[long_df[NF_ID].isin(sel)].copy()

    long_df = long_df.sort_values([NF_ID, NF_TIME]).reset_index(drop=True)
    # PER-SERIES fill. `sell_price` is NaN for every week before an item was first stocked in a
    # store, and a frame-wide ffill on a frame sorted by (id, ds) carries the last price of one
    # series into the opening rows of the next — a cross-series leak that is invisible because
    # the result is a plausible price. groupby-ffill keeps each series' gap inside that series;
    # a series that starts NaN has nothing to carry forward and falls through to bfill, then 0.
    grp = long_df.groupby(NF_ID, sort=False)[futr_cols]
    long_df[futr_cols] = grp.ffill().groupby(long_df[NF_ID], sort=False).bfill().fillna(0.0)

    stat_cols = [c for c in M5_STATIC_COLS if c in static_raw.columns]
    static_df = static_raw[static_raw[NF_ID].isin(long_df[NF_ID].unique())].copy()
    composition = {
        c: static_df[c].value_counts().sort_index().to_dict() for c in M5_STRATA if c in static_df
    }
    static_df = static_df[[NF_ID, *stat_cols]]
    static_df = _label_encode(static_df, stat_cols).reset_index(drop=True)

    meta = {
        "source": f"parquet:{src}",
        "selection": "stratified(dept_id,store_id)" if keep is not None else "alphabetical-prefix",
        "seed": seed if keep is not None else None,
        "n_series": int(long_df[NF_ID].nunique()),
        "n_rows": int(len(long_df)),
        "composition": composition,
    }
    return long_df, static_df, meta


def _build_synthetic(n_series: int = 8, n_days: int = 400, seed: int = 42) -> M5Bundle:
    """Tiny offline fixture with M5-shaped structure (weekly season + SNAP + price + events).

    NOT real M5 data — for plumbing/CI only. Flagged ``synthetic=True`` so metrics derived from
    it can never be mistaken for the real benchmark.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2011-01-29")  # M5's real start date, for realism
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for i in range(n_series):
        wday = dates.dayofweek.to_numpy()
        weekly = 5.0 + 3.0 * np.sin(2 * np.pi * wday / 7.0)
        snap = (dates.day <= 10).astype(float)  # SNAP roughly early-month
        price = 3.0 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 90.0)
        event = (rng.random(n_days) < 0.03).astype(float)
        base = weekly + 1.5 * snap - 0.8 * (price - 3.0) + 2.0 * event
        y = np.clip(base + rng.normal(0, 0.8, n_days) + 0.5 * i, 0.0, None)
        df = pd.DataFrame(
            {
                NF_ID: f"synth_{i:03d}",
                NF_TIME: dates,
                NF_TARGET: y,
                "sell_price": price,
                "snap_CA": snap,
                "snap_TX": snap,
                "snap_WI": snap,
                "wday": wday,
                "month": dates.month.to_numpy(),
                "is_weekend": (wday >= 5).astype(int),
                "event_active": event.astype(int),
            }
        )
        rows.append(df)
    long_df = pd.concat(rows, ignore_index=True)
    static_df = pd.DataFrame(
        {
            NF_ID: [f"synth_{i:03d}" for i in range(n_series)],
            "item_id": list(range(n_series)),
            "dept_id": [i % 3 for i in range(n_series)],
            "cat_id": [i % 2 for i in range(n_series)],
            "store_id": [i % 4 for i in range(n_series)],
            "state_id": [i % 3 for i in range(n_series)],
        }
    )
    return M5Bundle(
        long_df=long_df,
        static_df=static_df,
        futr_exog=m5_futr_exog_list(),
        stat_exog=m5_stat_exog_list(),
        synthetic=True,
        meta={"source": "synthetic", "n_series": n_series, "n_days": n_days},
    )


def load_m5_long(
    n_series: int | None = None,
    synthetic: bool = False,
    stratify: bool = False,
    seed: int = 42,
) -> M5Bundle:
    """Return an :class:`M5Bundle` ready for the run panel.

    Parameters
    ----------
    n_series : cap the number of (item, store) series (None = all 30490). For the brief exposé
        panel pass a subset (e.g. 500-2000) to keep the GPU run cheap; requests of <=2000 are
        served from the precomputed subset file *unless* ``stratify`` is set.
    synthetic : **opt-in only** offline fixture, for CI/plumbing. There is deliberately no
        automatic fallback: if the real corpus is missing this raises, because a graded
        deliverable that silently degrades to a toy fixture is worse than one that fails
        loudly (plan S8.0).
    stratify : draw the subset across (department, store) instead of taking the alphabetical
        prefix. **The panel should always set this** — see :func:`stratified_ids` for why the
        prefix is ``FOODS_1``-only and therefore not "M5".
    seed : the stratified draw's seed. Part of the experiment's identity, like a cutoff.
    """
    if synthetic:
        return _build_synthetic()
    long_df, static_df, meta = _build_from_parquet(n_series, stratify=stratify, seed=seed)
    return M5Bundle(
        long_df=long_df,
        static_df=static_df,
        futr_exog=m5_futr_exog_list(),
        stat_exog=m5_stat_exog_list(),
        meta=meta,
    )


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Smoke the M5 loader (prints shape + head).")
    ap.add_argument("--n-series", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    b = load_m5_long(n_series=args.n_series, synthetic=args.synthetic)
    print(f"synthetic={b.synthetic} source={b.meta.get('source')}")
    print(
        f"long_df: {b.long_df.shape}  series={b.long_df[NF_ID].nunique()}  "
        f"freq={b.freq} h={b.horizon}"
    )
    print(f"futr_exog={b.futr_exog}")
    print(f"stat_exog={b.stat_exog}")
    print(b.long_df.head())


if __name__ == "__main__":
    _cli()
