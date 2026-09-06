"""Load raw CSVs into neuralforecast long format and build the inference future table.

neuralforecast wants long format with columns ``unique_id`` / ``ds`` / ``y`` plus exogenous
columns. We rename the project's ``series_id`` / ``timestamp`` / ``target`` accordingly and
apply the shared imputation so every consumer sees the same feature set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.features import (
    ID,
    STATIC_COLS,
    TARGET,
    TIME,
    active_aggregate_columns,
    futr_exog_list,
)
from src.data.impute import apply_fill, fit_fill_stats

NF_ID, NF_TIME, NF_TARGET = "unique_id", "ds", "y"


def _read(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[TIME] = pd.to_datetime(df[TIME])
    return df


def load_long(
    csv_path: str | Path, fill_stats: dict | None = None, strategy: str = "median"
) -> tuple[pd.DataFrame, dict]:
    """Raw labelled CSV -> (long df ready for neuralforecast, fill_stats).

    Fits the imputation stats when ``fill_stats is None`` (training); otherwise reuses the
    passed stats (so a held-out slice imputes with the same medians as its train slice).

    ``strategy`` selects how the scattered NaNs are reconstructed (``src.data.gap_fill``);
    ``"median"`` is the incumbent and is bit-exact with the previous behaviour.
    """
    df = _read(csv_path)
    if fill_stats is None:
        fill_stats = fit_fill_stats(df)
    df = apply_fill(df, fill_stats, strategy=strategy)
    long = df.rename(columns={ID: NF_ID, TIME: NF_TIME, TARGET: NF_TARGET})
    return long, fill_stats


def static_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per series with the static covariates (neuralforecast ``static_df``)."""
    return long_df.groupby(NF_ID, as_index=False)[STATIC_COLS].first()


def add_volume_sample_weight(long_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a per-series ``sample_weight`` column = the series' robust scale (MAD).

    neuralforecast computes the point loss in the ``local_scaler_type='robust'`` (MAD) space,
    where each residual is implicitly divided by its series' MAD — so plain MAE equalizes
    series regardless of volume. WAPE, our target metric, instead weights residuals by their
    ORIGINAL magnitude (high-load units dominate). Weighting each series' loss by its MAD
    exactly undoes the scaler's ``1/MAD`` so the optimized objective becomes original-scale L1
    = the WAPE numerator (the denominator is constant w.r.t. the model). neuralforecast picks
    up a ``sample_weight`` column automatically and multiplies the loss mask by it.

    Weights are normalized to mean 1 across series for a stable loss scale (a constant factor
    does not change the argmin). Constant per series, so it survives the CV / gapped slicing.
    """
    med = long_df.groupby(NF_ID)[NF_TARGET].transform("median")
    mad = (long_df[NF_TARGET] - med).abs().groupby(long_df[NF_ID]).transform("median")
    w = mad.clip(lower=1e-6)  # guard a degenerate flat series (MAD 0) from zeroing its loss
    out = long_df.copy()
    out["sample_weight"] = w / w.mean()
    return out


def build_futr_df(
    cov_df: pd.DataFrame,
    target_index: pd.DataFrame,
    fill_stats: dict,
    *,
    gap_strategy: str = "median+exact",
    nan_strategy: str = "median",
) -> pd.DataFrame:
    """Future-covariate table (``unique_id, ds, <futr cols>``) for exactly ``target_index``'s rows.

    ``target_index`` is ANY ``(series_id, timestamp)`` frame, not necessarily the forecast index.
    That generality is the whole point on the submission path. The model forecasts ``h`` steps from
    the end of its stored history, and with a covariate-absent gap the graded rows sit at the END of
    that horizon — so the frame handed to ``nf.predict`` must span **gap + scored**, of which only
    the scored part is a row anybody supplies a file for. Passing the forecast index alone (what
    this function used to take) silently produced a short frame describing the wrong hours.

    Rows the covariate file does not carry are reconstructed rather than left NaN, because a NaN in
    ``futr_exog`` propagates straight to a NaN loss. The reconstruction **mirrors
    ``chronos2_eval._withhold_gap_covariates`` exactly** — same ``reconstruct_block``, same
    strategy, same ``*_missing = 1`` — which is what makes inference see the state the CV runs
    trained and were scored under, rather than a second, subtly different imputation. That
    function's own docstring calls its output *"exactly the state src.data.impute would produce
    when a covariate row is absent at inference"*; this is that claim made true.

    Three groups, three treatments, because they fail differently:

    - **The 5 periodic calendar encodings** are computed by ``src.data.calendar`` and **verified
      against the supplied rows** before being trusted on the absent ones. NOT via the ``exact``
      modifier: that needs ``EXACT_MIN_OBS=3`` observations per (series, hour-of-week) bin, and a
      336h block is two whole weeks — exactly **2**. It misses by one and silently median-fills a
      CONSTANT ``hour_sin`` across the gap.
    - **``trend``** is monotone in absolute time, so no hour-of-week rule reaches it at any
      observation count. Fitted and extrapolated — see ``src.data.calendar``.
    - **The 13 planning signals** are genuinely unknowable over the gap and get ``gap_strategy``.
      ``median+exact`` is S3's adopted position: ``median`` is the strategy S3 measured and kept
      (its alternatives were a null), and ``+exact`` is Finding 1, adopted as **correctness** —
      ``workload_intensity`` is deterministic, carries no ``*_missing`` flag, and was being fed to
      the model wrong for 336 hours without the model being told. Its measured cost is −0.00008.

    NOTE the CV cubes behind the recorded numbers ran plain ``median`` (``_trial_body``'s default),
    so ``median+exact`` differs from them by that −0.00008. It is a parameter rather than a constant
    for exactly that reason. It is also **irrelevant to the leaderboard run**, whose gap is 0: with
    no absent rows there is nothing for any gap strategy to do.
    """
    from src.data.calendar import complete_periodic, complete_trend
    from src.data.features import KNOWN_FUTURE_SIGNALS, missing_indicator_cols
    from src.data.gap_fill import reconstruct_block

    cov = cov_df.copy()
    cov[TIME] = pd.to_datetime(cov[TIME])

    tgt = target_index[[ID, TIME]].drop_duplicates().copy()
    tgt[TIME] = pd.to_datetime(tgt[TIME])

    # SPRINT 2. DERIVED cross-series aggregates are excluded here — a correctness fix at the
    # SOURCE rather than at each call site. This function reconstructs covariates that were
    # SUPPLIED (calendar, planning signals, cascade channels); an aggregate is computed FROM those
    # after imputation, by `src.data.xs_attach`, so it can neither be built nor validated here.
    # Leaving it in `futr_cols` made every caller raise "futr columns absent after build" the
    # moment a bundle declared the channel — three call sites in `predict.py` alone, and patching
    # them one at a time is how the fourth gets missed.
    _xs = set(active_aggregate_columns())
    futr_cols = [c for c in futr_exog_list() if c not in _xs]
    # Everything the model expects EXCEPT the `*_missing` flags, which are derived below rather
    # than supplied. Deriving this from `futr_exog_list()` instead of listing the two known groups
    # is what keeps a CASCADE CHANNEL in scope: with the channel active `futr_exog_list()` gains
    # `chronos2_forecast`, and a hardcoded `TIME_ENCODINGS + KNOWN_FUTURE_SIGNALS` silently drops
    # it at the merge — the covariate worth +8.3% to the ship member, absent with no symptom until
    # the completeness check below fires.
    base_cols = [c for c in futr_cols if c not in set(missing_indicator_cols())]
    supplied = [c for c in base_cols if c in cov.columns]
    merged = tgt.merge(cov[[ID, TIME, *supplied]], on=[ID, TIME], how="left")

    # A row the covariate file never carried: every supplied column is NaN at once. Distinguished
    # from an ordinary scattered NaN (one column, real neighbours) because the two want different
    # treatment — this one gets the GAP surface and a `*_missing` flag, that one gets `apply_fill`.
    absent = (
        merged[supplied].isna().all(axis=1) if supplied else pd.Series(True, index=merged.index)
    )

    if absent.any():
        # The DETERMINISTIC six first, and NOT through the `exact` modifier — it cannot fire here.
        # `exact` needs EXACT_MIN_OBS=3 observations per (series, hour-of-week) bin; the submission
        # supplies a 336h block, which is two whole weeks, which is exactly 2. It misses by one and
        # every encoding then falls through to a per-series median — a CONSTANT `hour_sin` across
        # the gap. `src.data.calendar` computes them instead and verifies the formula against the
        # supplied rows before trusting it on the absent ones.
        merged = complete_periodic(merged, reference=cov)
        merged = complete_trend(merged, reference=cov)

        # The 13 planning signals are genuinely unknowable and get the gap strategy. `exact` stays
        # in the spec because on a longer covariate block it does fire, and `workload_intensity`
        # is the column it exists for.
        recon = list(KNOWN_FUTURE_SIGNALS)
        for col in recon:
            if col not in merged.columns:
                merged[col] = float("nan")
        fills = reconstruct_block(
            target=merged,
            history=cov,
            cols=recon,
            fill_mask=absent,
            strategy=gap_strategy,
            id_col=ID,
            time_col=TIME,
        )
        for col in recon:
            merged.loc[absent, col] = fills.loc[absent, col]

    # `apply_fill` handles the SCATTERED surface — isolated NaNs inside rows the file did supply —
    # and derives `*_missing` from what is NaN at this moment.
    #
    # `nan_strategy` is per-MEMBER, which is S3's adopted split rather than a knob for its own sake:
    # the tree gained +0.00401 (3/3 windows) under `interp` while the cascade LOST 0.00937 under it,
    # so the two ship members genuinely want different imputation and the submission path builds a
    # frame for each. A single shared frame would have to mis-serve one of them.
    merged = apply_fill(merged, fill_stats, strategy=nan_strategy)

    # Only now flag the absent rows. `apply_fill` recomputes the indicators from NaN presence, and
    # the gap block was just reconstructed above, so running this earlier would have the fill
    # silently reset every flag to a confident 0 — the model told nothing was imputed precisely
    # where everything was. This is the same 1.0 `_withhold_gap_covariates` writes.
    if absent.any():
        for col in missing_indicator_cols():
            merged.loc[absent, col] = 1.0
    still_missing = [c for c in futr_cols if c not in merged.columns]
    if still_missing:
        raise ValueError(f"futr columns absent after build: {still_missing}")
    if merged[futr_cols].isna().any().any():
        bad = [c for c in futr_cols if merged[c].isna().any()]
        raise ValueError(f"futr columns still NaN after imputation: {bad}")

    merged = merged.rename(columns={ID: NF_ID, TIME: NF_TIME})
    return merged[[NF_ID, NF_TIME, *futr_cols]]
