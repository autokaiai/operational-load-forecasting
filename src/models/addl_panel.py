"""S8's additional-dataset panel: run THIS project's architecture on an external corpus.

The spec requires the architecture to be *run* externally, not cited, so every arm here is the
in-repo model transplanted onto a foreign frame — same gapped split shape, same Chronos->TFT
cascade construction, same WAPE. It lives in ``src/`` rather than inside the Modal launcher for
two reasons: ``src`` is what is importable in the container, and a panel that produces a graded
number should be reachable by ``pytest`` (S8.0's synthetic-fallback trap is what that rule is
for).

Why the split is ``train | gap(h) | score(h)``
----------------------------------------------
Our private test sits one full horizon past the freshest label, so the model must cross a block
it never observes. Reproducing that offset is the whole point of running externally: a
same-day-continuation split would measure a different problem and transfer nothing.

The cascade covariate is ROLLED, not faked
------------------------------------------
An earlier draft of this panel set the training covariate to ``y.shift(h)`` — the realised
target, lagged. That is not the architecture. It hands the TFT a *noiseless oracle* during
training and an actual forecast at inference, so the VSN calibrates its trust on a channel
quality that does not exist at test time; S4 measured that exact mismatch costing 10.8% on our
own data, and here it would have been far larger because the training channel was perfect. It
also breaks S8's contamination detector, whose whole content is comparing this dataset's
cascade-over-plain-TFT lift against our clean +8.3% — a lift measured against a different
covariate construction is not comparable to it.

So :func:`roll_chronos_covariate` mirrors ``chronos2_oof.generate_train``: block starting at
``s`` conditions on ``[0, s)`` and forecasts ``[s, s+block)``, grid aligned to the train end.
Context stops before every forecast step, so the covariate is gap-honest by construction rather
than by inspection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

NF_ID, NF_TIME, NF_TARGET = "unique_id", "ds", "y"
CHRONOS_COL = "chronos2_forecast"
CHRONOS_MISSING = "chronos2_forecast_missing"


# --------------------------------------------------------------------------------------------
# metrics — the same six, but two of them DEGENERATE on a corpus with zeros
# --------------------------------------------------------------------------------------------


def panel_metrics(y, yhat, seasonality: int | None = None) -> dict:
    """All six metrics, plus the annotations a zero-inflated target makes mandatory.

    ``src.metrics`` was written against OUR target, whose docstring assumption — *"the target is
    strictly positive, so this is well-defined"* — is literally true: 0 of 414,720 training rows
    are zero and the minimum is 0.164. On M5 it is false, and two of the six break in ways that
    are obvious once measured and invisible if the numbers are simply tabulated:

    ===========  ====================  ==========================================================
    metric       measured on M5        what happens
    ===========  ====================  ==========================================================
    **WAPE**     0.7961                fine. The denominator is POOLED ``sum|y|`` (91,616 over the
                                       scored block), so individual zero rows cannot divide by
                                       zero. **This is why WAPE is the primary metric and the one
                                       that transfers.**
    MAE/MSE/RMSE 1.30 / 6.88 / 2.62    fine, but SCALE-BOUND — M5 sells 1.6 units/day where our
                                       load index averages 9.9. Valid for ranking arms *within*
                                       M5, meaningless across datasets.
    **MAPE**     **33,937,117.68**     **BROKEN.** 49% of scored rows are zero and ``mape``
                                       divides by ``max(|y|, 1e-8)``, so each contributes ~1e8.
                                       Reported as ``null`` with the reason, never as a number.
    **sMAPE**    1.3040                **DEGENERATE.** A ``y=0, yhat>0`` row scores exactly 2.0,
                                       the ceiling. 0.98 of the 1.30 — **75%** — is just the zero
                                       rows. It measures how often the truth is zero, not skill.
    ===========  ====================  ==========================================================

    So this returns the six unchanged where they are meaningful, ``mape=None`` where it is not,
    and three extra fields that let a reader see *why*: ``zero_frac``, ``mape_nonzero`` (the
    honest MAPE, over ``y>0`` rows only) and ``smape_zero_share``.
    """
    from src.metrics import all_metrics

    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    out = dict(all_metrics(y, yhat))

    zero = y == 0
    zero_frac = float(zero.mean()) if y.size else 0.0
    out["zero_frac"] = round(zero_frac, 6)
    if zero.any():
        out["mape"] = None
        out["mape_note"] = (
            f"undefined: {int(zero.sum())}/{y.size} scored rows have y=0; "
            "MAPE's per-row division is not rescuable on a zero-inflated target"
        )
        nz = ~zero
        out["mape_nonzero"] = float(np.mean(np.abs(y[nz] - yhat[nz]) / y[nz])) if nz.any() else None
        denom = out["smape"]
        out["smape_zero_share"] = round(2.0 * zero_frac / denom, 4) if denom else None
        out["smape_note"] = (
            "degenerate: every y=0 row scores the 2.0 ceiling, so smape_zero_share of this "
            "figure is the zero rate rather than forecast skill"
        )
    if seasonality:
        out["seasonality"] = int(seasonality)
    return out


# --------------------------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------------------------


@dataclass
class Split:
    """``train | gap(h) | score(h)``, plus the two lengths every arm needs."""

    train: pd.DataFrame
    horizon_block: pd.DataFrame  # the full 2h block the model forecasts
    h: int  # scored length (== gap length)
    full_h: int  # 2h — what nf is asked to forecast

    @property
    def score_labels(self) -> pd.DataFrame:
        return self.horizon_block.groupby(NF_ID, sort=False).tail(self.h)[
            [NF_ID, NF_TIME, NF_TARGET]
        ]

    @property
    def block_labels(self) -> pd.DataFrame:
        """The FULL 2h block with an ``is_scored`` flag on its second half.

        Arms keep their whole forecast rather than only the scored tail, because the gap half is
        the one place a blend weight can be fitted honestly: it has labels (it is inside our own
        data — "unobserved" only from the model's point of view) and it is disjoint from the rows
        the weight is judged on. That is the same fit-here / score-there discipline our own S6
        weight fit uses (``blk < 224`` fits, ``blk >= 224`` scores), at this corpus' geometry.
        """
        b = add_step_index(self.horizon_block[[NF_ID, NF_TIME, NF_TARGET]])
        b["is_scored"] = b["_sidx"] >= (self.full_h - self.h)
        return b.drop(columns="_sidx")


def add_step_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([NF_ID, NF_TIME]).reset_index(drop=True)
    out["_sidx"] = out.groupby(NF_ID, sort=False).cumcount()
    return out


def gapped_split(long_df: pd.DataFrame, horizon: int) -> Split:
    """Carve ``train | gap(h) | score(h)`` off each series' tail.

    Per-series on a 0-based step index, so series of unequal length still align on their tails.
    ``horizon`` shrinks if a series is too short to give ``2h`` plus context — which only ever
    fires on the synthetic fixture, and is what lets the plumbing smoke run at all.
    """
    df = add_step_index(long_df)
    min_len = int(df.groupby(NF_ID, sort=False).size().min())
    if min_len < 2 * horizon + 16:
        horizon = max(2, (min_len - 16) // 2)
    cut = df.groupby(NF_ID, sort=False)["_sidx"].transform("max") - (2 * horizon) + 1
    train = df[df["_sidx"] < cut].drop(columns="_sidx").reset_index(drop=True)
    block = df[df["_sidx"] >= cut].drop(columns="_sidx").reset_index(drop=True)
    full_h = int(block.groupby(NF_ID, sort=False).size().min())
    return Split(train=train, horizon_block=block, h=horizon, full_h=full_h)


# --------------------------------------------------------------------------------------------
# the Chronos covariate — the cascade's whole content
# --------------------------------------------------------------------------------------------


def _seasonal_naive(context: pd.DataFrame, future: pd.DataFrame, season: int) -> pd.DataFrame:
    """Dry-run backend: tile each series' last ``season`` observations. NOT for real results."""
    out = []
    for sid, fg in future.groupby(NF_ID, sort=False):
        hist = context.loc[context[NF_ID] == sid, NF_TARGET].to_numpy()
        period = hist[-season:] if hist.size >= season else hist
        fill = np.resize(period, len(fg)) if period.size else np.zeros(len(fg))
        g = fg[[NF_ID, NF_TIME]].copy()
        g[CHRONOS_COL] = fill
        out.append(g)
    return pd.concat(out, ignore_index=True)


def forecast_block(
    pipe,
    context: pd.DataFrame,
    future: pd.DataFrame,
    exog: list[str],
    h: int,
    batch_series: int = 0,
    dry_run: bool = False,
    season: int = 7,
) -> pd.DataFrame:
    """One Chronos-2 block -> ``(unique_id, ds, chronos2_forecast)``, clipped at 0.

    Delegates to ``chronos2_eval._predict``, the SAME batched entry point the project's own
    cascade uses — it is dataset-agnostic (it takes the id/time/target column names), so the
    external run genuinely exercises our code rather than a re-implementation of it. Batching
    matters here far more than on our 96 series: a per-series loop over a 500-series panel is
    500 pipeline calls per block against one.
    """
    if dry_run:
        pred = _seasonal_naive(context, future, season)
    else:
        from src.models.chronos2_eval import _pick_pred_column, _predict

        ctx = context[[NF_ID, NF_TIME, NF_TARGET, *exog]]
        fut = future[[NF_ID, NF_TIME, *exog]]
        raw = _predict(pipe, ctx, fut, h, batch_series)
        pcol = _pick_pred_column(raw)
        pred = raw[[NF_ID, NF_TIME, pcol]].rename(columns={pcol: CHRONOS_COL})
    pred[CHRONOS_COL] = pred[CHRONOS_COL].clip(lower=0.0)
    return pred


def roll_chronos_covariate(
    pipe,
    train: pd.DataFrame,
    exog: list[str],
    block: int,
    batch_series: int = 0,
    dry_run: bool = False,
    season: int = 7,
    max_blocks: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Rolling-origin zero-shot forecast over the train region (``chronos2_oof.generate_train``).

    Block ``s`` sees ``[0, s)`` and forecasts ``[s, s+block)``, so the context ends strictly
    before every forecast step. The grid is aligned to the *train end*, which puts training rows
    on the same lead profile as the horizon block — S4 measured that the 1..block profile beats
    a matched 1..2*block one by 10.8%, so the alignment is a decision, not an accident.

    ``max_blocks`` caps the pass to the most recent K blocks (earlier rows keep the warm-up NaN
    and its missing flag). **Smoke use only** — it changes the covariate the model trains on.
    """
    df = add_step_index(train)
    n = int(df.groupby(NF_ID, sort=False)["_sidx"].max().min()) + 1
    starts = list(range(n - block, 0, -block))
    if max_blocks is not None:
        starts = starts[:max_blocks]
    out = []
    for i, s in enumerate(starts, 1):
        ctx = df[df["_sidx"] < s]
        fut = df[(df["_sidx"] >= s) & (df["_sidx"] < s + block)]
        assert int(ctx["_sidx"].max()) == s - 1, "context overruns the origin"
        assert int(fut["_sidx"].min()) >= s, "forecast step precedes the origin"
        t0 = time.perf_counter()
        out.append(forecast_block(pipe, ctx, fut, exog, block, batch_series, dry_run, season))
        print(
            f"  [cov {i}/{len(starts)}] steps [{s}..{s + block - 1}] "
            f"({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
    covered = len(starts) * block
    meta = {
        "blocks": [(int(s), int(block)) for s in starts],
        "n_steps": n,
        "warmup_steps": int(n - covered),
        "capped": max_blocks is not None,
    }
    frame = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[NF_ID, NF_TIME])
    return frame, meta


def attach_covariate(frame: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Left-join the covariate on (id, ds) and add its missing flag.

    Rows the rolling pass could not reach — the warm-up prefix, which has no prior origin to
    condition on — are filled with the series' own covariate median and flagged. That mirrors
    ``apply_fill``: the model is told the value was reconstructed rather than being handed a
    confident number, which is the whole reason the flag columns exist.
    """
    out = frame.merge(cov, on=[NF_ID, NF_TIME], how="left")
    out[CHRONOS_MISSING] = out[CHRONOS_COL].isna().astype(float)
    med = out.groupby(NF_ID, sort=False)[CHRONOS_COL].transform("median")
    out[CHRONOS_COL] = out[CHRONOS_COL].fillna(med).fillna(0.0)
    return out


# --------------------------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------------------------


# THE SHIPPED HYPERPARAMETERS (configs/tft_chronos.yaml). Mirrored explicitly rather than left
# to neuralforecast's defaults, because the defaults are a DIFFERENT MODEL: hidden_size 128 vs our
# 64, and windows_batch_size 1024 vs our 32. S8 asks whether OUR architecture transfers, so
# running the library's default TFT would answer a question nobody asked — and the 1024 default
# OOM'd an 8 GiB card on an 8-series fixture, which is the same memory ceiling that killed 11 of
# S5's 32 trials. Same discipline as pinning `grn_activation`: silent when wrong.
SHIPPED_TFT = {
    "hidden_size": 64,
    "n_head": 4,
    "dropout": 0.1,
    "windows_batch_size": 32,
    "inference_windows_batch_size": 32,
    "grn_activation": "ELU",  # S5b: pinned, never left to the upstream default
}


def build_nf(model_name, h, freq, futr_exog, stat_exog, seed=42, max_steps=1000, input_size=None):
    """Construct a neuralforecast TFT/LSTM at THE SHIPPED hyperparameters (see ``SHIPPED_TFT``)."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import LSTM, TFT

    cls = {"TFT": TFT, "LSTM": LSTM}[model_name]
    kwargs = {
        "h": h,
        "input_size": input_size or max(2 * h, 96),
        "loss": MAE(),  # S5 swept {MAE, Huber} and MAE won 20/21 survivors
        "max_steps": max_steps,
        "scaler_type": "robust",
        "enable_progress_bar": False,
        "logger": False,
        "random_seed": seed,  # S5.2: `seed` alone never reaches the model
        **SHIPPED_TFT,
    }
    if cls is not TFT:
        for k in ("grn_activation", "n_head", "hidden_size"):
            kwargs.pop(k, None)
    if cls.EXOGENOUS_FUTR and futr_exog:
        kwargs["futr_exog_list"] = list(futr_exog)
    if cls.EXOGENOUS_STAT and stat_exog:
        kwargs["stat_exog_list"] = list(stat_exog)
    return NeuralForecast(models=[cls(**kwargs)], freq=freq, local_scaler_type="robust")


def _fit_predict_nf(model_name, split, bundle, futr_exog, train, futr_block, max_steps, seed):
    stat_exog = list(bundle.stat_exog)
    nf = build_nf(model_name, split.full_h, bundle.freq, futr_exog, stat_exog, seed, max_steps)
    fit_kwargs = {"val_size": split.full_h}
    if stat_exog:
        fit_kwargs["static_df"] = bundle.static_df
    nf.fit(train, **fit_kwargs)
    preds = nf.predict(futr_df=futr_block) if futr_exog else nf.predict()
    if NF_ID not in preds.columns:
        preds = preds.reset_index()
    return preds


def _score(preds: pd.DataFrame, col: str, split: Split, out_name: str, seasonality=None):
    """Join the arm's FULL-block forecast onto the labels by TIMESTAMP; score the tail only.

    Positional matching is what let the #32 horizon defect produce a perfectly-shaped CSV that
    described the wrong hours, so the join is on (id, ds) and completeness is asserted rather
    than assumed.

    The returned frame spans the whole ``2h`` block with an ``is_scored`` flag; only the flagged
    rows enter the metrics. Keeping the gap half is what makes an honest blend-weight fit
    possible later at zero extra compute — see :func:`fit_blend_weight`.
    """
    got = preds[[NF_ID, NF_TIME, col]].rename(columns={col: out_name})
    merged = split.block_labels.merge(got, on=[NF_ID, NF_TIME], how="left")
    if merged[out_name].isna().any():
        n = int(merged[out_name].isna().sum())
        raise ValueError(f"{out_name}: {n}/{len(merged)} block rows got no prediction")
    scored = merged[merged["is_scored"]]
    return merged, panel_metrics(scored[NF_TARGET], scored[out_name], seasonality)


def score_tft(bundle, split, max_steps=1000, seed=42):
    """Plain TFT — the ablation the cascade is measured against, and the contamination baseline."""
    futr = list(bundle.futr_exog)
    futr_block = split.horizon_block[[NF_ID, NF_TIME, *futr]]
    preds = _fit_predict_nf("TFT", split, bundle, futr, split.train, futr_block, max_steps, seed)
    return _score(preds, "TFT", split, "tft", bundle.seasonality)


def score_lstm(bundle, split, max_steps=1000, seed=42):
    """LSTM — kept runnable but OUT of the default panel (S8.1: it is in no ship candidate)."""
    futr = list(bundle.futr_exog)
    futr_block = split.horizon_block[[NF_ID, NF_TIME, *futr]]
    preds = _fit_predict_nf("LSTM", split, bundle, futr, split.train, futr_block, max_steps, seed)
    return _score(preds, "LSTM", split, "lstm", bundle.seasonality)


def score_cascade(bundle, split, cov_train, cov_horizon, max_steps=1000, seed=42):
    """The architecture: a zero-shot Chronos-2 forecast as an extra known-future TFT covariate."""
    futr = [*bundle.futr_exog, CHRONOS_COL, CHRONOS_MISSING]
    train = attach_covariate(split.train, cov_train)
    futr_block = attach_covariate(
        split.horizon_block[[NF_ID, NF_TIME, *bundle.futr_exog]], cov_horizon
    )
    preds = _fit_predict_nf("TFT", split, bundle, futr, train, futr_block, max_steps, seed)
    return _score(preds, "TFT", split, "cascade", bundle.seasonality)


def score_chronos2(bundle, split, cov_horizon):
    """Zero-shot Chronos-2 alone — free, and S8's second, independent contamination detector."""
    cov = cov_horizon.rename(columns={CHRONOS_COL: "chronos2_zeroshot"})
    return _score(cov, "chronos2_zeroshot", split, "chronos2_zeroshot", bundle.seasonality)


def score_naive(bundle, split):
    """Hold the last observed value flat. Our ``naive_last_value``, transplanted (ours: 0.5471)."""
    last = split.train.sort_values([NF_ID, NF_TIME]).groupby(NF_ID, sort=False)[NF_TARGET].last()
    block = split.horizon_block[[NF_ID, NF_TIME]].copy()
    block["naive"] = block[NF_ID].map(last).astype(float)
    return _score(block, "naive", split, "naive", bundle.seasonality)


def score_seasonal_naive(bundle, split):
    """Tile the last full season forward. Our ``lag168_repeat``, at this corpus' own period.

    **This arm is why the panel's WAPE is comparable to our project's at all.** An absolute WAPE
    of 0.80 on M5 against 0.13 on our data says nothing about transfer — M5's target is
    zero-inflated count data at a mean of 1.6 units, ours is a smooth strictly-positive index at
    9.9, and the two are simply not the same difficulty. What IS comparable is the RATIO to a
    baseline that is defined identically on both: *"the cascade cuts seasonal-naive's error by
    X% here and by Y% on our data."* Costs nothing — no model, no GPU.
    """
    season = int(bundle.seasonality)
    hist = {
        u: g[NF_TARGET].to_numpy(dtype=float)
        for u, g in split.train.sort_values([NF_ID, NF_TIME]).groupby(NF_ID, sort=False)
    }
    out = []
    for uid, g in split.horizon_block.sort_values([NF_ID, NF_TIME]).groupby(NF_ID, sort=False):
        h_ = hist.get(uid, np.zeros(0))
        period = h_[-season:] if h_.size >= season else h_
        f = g[[NF_ID, NF_TIME]].copy()
        f["seasonal_naive"] = np.resize(period, len(g)) if period.size else np.zeros(len(g))
        out.append(f)
    block = pd.concat(out, ignore_index=True)
    return _score(block, "seasonal_naive", split, "seasonal_naive", bundle.seasonality)


def score_seasonal_mean(bundle, split):
    """Per (series, phase-in-season) MEAN over the train slice — the handout's named baseline.

    Stronger than seasonal-naive because it averages every observed period instead of trusting
    the last one (S3 measured the same effect on the covariate surface: tiling a single week is
    *worse* than a median over ~20, because one week carries its own noise). Reported for naming
    parity with the spec; NOT used as the skill denominator — see ``SKILL_BASELINE``.
    """
    season = int(bundle.seasonality)
    train = add_step_index(split.train)
    train["_phase"] = train["_sidx"] % season
    table = train.groupby([NF_ID, "_phase"], sort=False)[NF_TARGET].mean()
    block = add_step_index(split.horizon_block)
    n_train = train.groupby(NF_ID, sort=False)["_sidx"].max() + 1
    block["_phase"] = (block["_sidx"] + block[NF_ID].map(n_train)) % season
    keys = pd.MultiIndex.from_arrays([block[NF_ID], block["_phase"]])
    out = block[[NF_ID, NF_TIME]].copy()
    out["seasonal_mean"] = table.reindex(keys).to_numpy()
    out["seasonal_mean"] = out["seasonal_mean"].fillna(split.train[NF_TARGET].median())
    return _score(out, "seasonal_mean", split, "seasonal_mean", bundle.seasonality)


def score_lgbm(bundle, split, n_estimators=300, seed=42):
    """Recursive LightGBM over the horizon block, rolled ACROSS SERIES rather than row by row.

    The previous implementation walked ``iterrows`` and called ``predict`` once per row: on a
    500-series panel with a 56-step block that is 28,000 single-row DataFrame constructions and
    28,000 booster calls, minutes of pure Python inside a leased GPU container. A recursive roll
    cannot be vectorised over TIME — step t+1 genuinely needs step t's prediction — but every
    series advances through the same step together, so it vectorises over SERIES exactly. Same
    arithmetic, one predict call per step instead of one per row.
    """
    import lightgbm as lgb

    h, full_h = split.h, split.full_h
    lags = (1, 2, 7, 14, h)
    futr = [c for c in bundle.futr_exog if c in split.train.columns]

    def design(df):
        d = df.sort_values([NF_ID, NF_TIME]).copy()
        g = d.groupby(NF_ID, sort=False)[NF_TARGET]
        for lag in lags:
            d[f"lag_{lag}"] = g.shift(lag)
        d["roll_mean_7"] = g.shift(1).rolling(7).mean().reset_index(0, drop=True)
        ds = pd.to_datetime(d[NF_TIME])
        d["dow"], d["hour"] = ds.dt.dayofweek, ds.dt.hour
        return d

    tr = design(split.train).dropna(subset=[f"lag_{lag}" for lag in lags] + ["roll_mean_7"])
    feat_cols = [*(f"lag_{lag}" for lag in lags), "roll_mean_7", "dow", "hour", *futr]
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        verbose=-1,
        random_state=seed,
    )
    model.fit(tr[feat_cols], tr[NF_TARGET])

    # Series x time matrices. Every series has the same block length (gapped_split is
    # tail-aligned), so one lockstep walk covers the panel.
    ids = sorted(split.train[NF_ID].unique())
    pos = {u: i for i, u in enumerate(ids)}
    tr_sorted = split.train.sort_values([NF_ID, NF_TIME])
    lens = tr_sorted.groupby(NF_ID, sort=True).size().to_numpy()
    if len(set(lens.tolist())) != 1:
        raise ValueError("ragged train lengths; the lockstep roll needs equal-length series")
    hist = tr_sorted[NF_TARGET].to_numpy(dtype=float).reshape(len(ids), int(lens[0]))

    blk = split.horizon_block.sort_values([NF_ID, NF_TIME])
    blk_idx = blk[NF_ID].map(pos).to_numpy()
    order = np.lexsort((np.arange(len(blk)), blk_idx))
    blk = blk.iloc[order]
    ds_mat = pd.to_datetime(blk[NF_TIME]).to_numpy().reshape(len(ids), full_h)
    exog_mat = {c: blk[c].to_numpy(dtype=float).reshape(len(ids), full_h) for c in futr}

    grown = hist
    preds = np.empty((len(ids), full_h), dtype=float)
    for t in range(full_h):
        n = grown.shape[1]
        cols = {f"lag_{lag}": grown[:, n - lag] for lag in lags}
        cols["roll_mean_7"] = grown[:, -7:].mean(axis=1)
        step_ds = pd.to_datetime(ds_mat[:, t])
        cols["dow"] = step_ds.dayofweek.to_numpy()
        cols["hour"] = step_ds.hour.to_numpy()
        for c in futr:
            cols[c] = exog_mat[c][:, t]
        yhat = model.predict(pd.DataFrame(cols)[feat_cols])
        preds[:, t] = yhat
        grown = np.hstack([grown, yhat.reshape(-1, 1)])

    flat = pd.DataFrame(
        {
            NF_ID: np.repeat(ids, full_h),
            NF_TIME: ds_mat.reshape(-1),
            "lgbm": preds.reshape(-1),
        }
    )
    return _score(flat, "lgbm", split, "lgbm", bundle.seasonality)


def _blend_join(frames: dict, tree: str, neural: str):
    if tree not in frames or neural not in frames:
        return None
    a, b = frames[neural], frames[tree]
    m = a[[NF_ID, NF_TIME, NF_TARGET, "is_scored", neural]].merge(
        b[[NF_ID, NF_TIME, tree]], on=[NF_ID, NF_TIME]
    )
    # Both sides, not just one: an inner join that drops rows from the SHORTER frame still
    # matches that frame's length, so comparing against one arm alone cannot see the loss.
    if not len(m) == len(a) == len(b):
        raise ValueError(
            f"blend join changed the row count: {neural}={len(a)} {tree}={len(b)} -> {len(m)}"
        )
    return m


def blend(frames: dict, weight: float, tree="lgbm", neural="cascade"):
    """Convex blend at a GIVEN weight, scored on the scored half."""
    m = _blend_join(frames, tree, neural)
    if m is None:
        return None, None
    m = m.copy()
    m["blend"] = (weight * m[tree] + (1.0 - weight) * m[neural]).clip(lower=0.0)
    s = m[m["is_scored"]]
    return m, panel_metrics(s[NF_TARGET], s["blend"])


def fit_blend_weight(frames: dict, tree="lgbm", neural="cascade", grid_step: float = 0.01):
    """Fit the tree's weight ON THE GAP HALF, then report it scored on the scored half.

    **The weight never sees the rows it is judged on.** The horizon block is ``gap(h) | score(h)``
    and both halves carry labels, so the first half is a free, disjoint fitting set — the same
    fit-here / score-there discipline as S6's own weight fit (``blk < 224`` fits, ``blk >= 224``
    scores), transplanted to this corpus' geometry. Costs no extra compute: both arms already
    forecast the whole block.

    This answers a DIFFERENT question from :func:`blend` at the shipped weight, and the write-up
    needs both. Shipped weight -> *"does the model we actually submit transfer?"*, which is the
    deliverable's question. Refitted weight -> *"could this architecture be tuned for M5?"*, which
    is the handout's "what had to be changed". Reporting only the refit would quietly upgrade a
    transfer claim into a tuning claim.

    Read the fitted weight itself as the interesting output, not the WAPE it buys: S1's 4.7, S2
    Stage 2 and S3 all ended with the simplex rearranging around a new arm and handing back noise,
    so the number worth quoting is *where the optimum sits* and how far it moved from our 0.24.
    """
    m = _blend_join(frames, tree, neural)
    if m is None:
        return None
    fit = m[~m["is_scored"]]
    if fit.empty:
        return None
    grid = np.arange(0.0, 1.0 + grid_step / 2, grid_step)
    y, t, n = (fit[NF_TARGET].to_numpy(), fit[tree].to_numpy(), fit[neural].to_numpy())
    denom = np.abs(y).sum()
    errs = [np.abs(y - np.clip(w * t + (1 - w) * n, 0, None)).sum() / denom for w in grid]
    w_star = float(grid[int(np.argmin(errs))])
    preds, metrics = blend(frames, w_star, tree=tree, neural=neural)
    return {
        "weight": round(w_star, 4),
        "fitted_on": "the gap half of the horizon block (disjoint from the scored rows)",
        "fit_rows": int(len(fit)),
        "fit_wape": round(float(min(errs)), 6),
        "metrics": metrics,
        "preds": preds,
    }


# --------------------------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------------------------

# S6's shipped weight. NOT refitted on the external corpus — see `blend`.
SHIP_TREE_WEIGHT = 0.24
# S8's calibration constant: the cascade's lift over plain TFT on OUR data, where contamination
# is impossible (the 96-series corpus was never public). A lift far above this on a public
# benchmark is the tell that Chronos-2 has memorised it rather than generalised to it.
CLEAN_CASCADE_LIFT = 0.083
# Baselines first: they are free, and until they exist the model WAPEs have no interpretation.
# `naive` and `seasonal_mean` are the two the HANDOUT names by name ("compare against the naive
# last-value baseline, the seasonal mean baseline"), so both are in the default panel.
DEFAULT_ARMS = (
    "naive",
    "seasonal_naive",
    "seasonal_mean",
    "chronos2",
    "lgbm",
    "tft",
    "cascade",
)

# Our own corpus, same definitions, for the skill ratios (`results/baselines.json`, ship figures).
OUR_REFERENCE = {
    "naive": 0.5471,  # naive_last_value, pooled over the 3 CV windows
    "seasonal_naive": 0.4792,  # lag168_repeat — ONE FULL WEEKLY PERIOD tiled forward
    "seasonal_mean": 0.3269,  # per (series, hour-of-week) mean — the handout's named baseline
    "tft": 0.15138,  # plain TFT, late regime, median fill
    "cascade": 0.13429,  # tft_cascade single draw (seed-averaged: 0.13908 +- 0.00278)
    "blend": 0.13162,  # the SHIPPED 0.24*tree(interp) + 0.76*cascade_bag5(median)
}
# The skill denominator. `seasonal_naive` tiles ONE seasonal period on both corpora (168h here,
# 7d on M5), so it is the baseline whose DEFINITION is identical across them — which is the only
# property that makes the ratio comparable. `seasonal_mean` is stronger and is reported beside
# it because the handout names it, but it averages over a different number of periods on each
# corpus (~26 weeks for us, ~270 for M5), so it is NOT the right denominator for transfer.
SKILL_BASELINE = "seasonal_naive"


def _open_pipe(device: str, model_id: str = "amazon/chronos-2", revision: str | None = None):
    from chronos import Chronos2Pipeline  # type: ignore

    kwargs = {"device_map": device}
    if revision:
        kwargs["revision"] = revision
    return Chronos2Pipeline.from_pretrained(model_id, **kwargs)


def run_panel(
    bundle,
    arms=DEFAULT_ARMS,
    max_steps: int = 1000,
    seed: int = 42,
    device: str = "cuda",
    batch_series: int = 0,
    dry_run: bool = False,
    max_cov_blocks: int | None = None,
    tree_weight: float = SHIP_TREE_WEIGHT,
    chronos_revision: str | None = None,
    xs: bool = False,
):
    """Run the arms on one external bundle and return a JSON-ready result dict.

    The Chronos covariate is generated ONCE and shared by the ``cascade`` and ``chronos2`` arms —
    the zero-shot forecast over the horizon block *is* the chronos2 arm, so the second detector
    costs nothing beyond the arm that already needs it.
    """
    import traceback

    arms = list(arms)
    # SPRINT 2 -- the M5 analogue of A9, attached BEFORE the split so every arm sees it and the
    # capacity weights are fitted on the train region only. `xs=False` leaves the bundle untouched,
    # so the recorded panel in `results/addl_dataset_metrics.json` is reproducible byte-for-byte.
    xs_names: list[str] = []
    if xs:
        from src.data.m5_xs import build_m5_a9

        bundle.long_df, xs_names = build_m5_a9(bundle)
        bundle.futr_exog = [*bundle.futr_exog, *xs_names]
        # LAW 5: a lever that does not reach the model returns a confident null.
        missing = [c for c in xs_names if c not in bundle.long_df.columns]
        if missing or not set(xs_names).issubset(bundle.futr_exog):
            raise SystemExit(f"[m5-a9] block did not reach the model: {missing or xs_names}")
        print(f"[m5-a9] {len(xs_names)} aggregate column(s) active: {xs_names}", flush=True)
    split = gapped_split(bundle.long_df, bundle.horizon)
    result = {
        "dataset": getattr(bundle, "name", "unknown"),
        "synthetic": bool(bundle.synthetic),
        "source": bundle.meta.get("source"),
        "selection": bundle.meta.get("selection"),
        "freq": bundle.freq,
        "n_series": int(bundle.long_df[NF_ID].nunique()),
        "n_rows": int(len(bundle.long_df)),
        "split": {
            "train_steps": int(split.train.groupby(NF_ID, sort=False).size().min()),
            "gap": split.h,
            "score_len": split.h,
            "forecast_len": split.full_h,
        },
        "seed": seed,
        "meta": bundle.meta,
        "xs_block": xs_names or None,
        "models": {},
        "notes": {
            "split": "train | gap(h) | score(h) — mirrors our private-test +h offset",
            "metric": "WAPE primary; MAE/MSE/RMSE/MAPE/sMAPE reported",
            "seeds": "SINGLE SEED (plan 3.6: single for an A/B, frozen five for a finalist). "
            "These are single-draw absolutes; on our own data that spread is ~0.003.",
            "blend": f"post-hoc at the SHIPPED weight {tree_weight}, not refitted on this corpus",
        },
    }

    cov_train = cov_horizon = None
    needs_cov = bool({"cascade", "chronos2"} & set(arms))
    if needs_cov:
        t0 = time.perf_counter()
        pipe = None if dry_run else _open_pipe(device, revision=chronos_revision)
        exog = list(bundle.futr_exog)
        cov_train, cov_meta = roll_chronos_covariate(
            pipe,
            split.train,
            exog,
            block=split.h,
            batch_series=batch_series,
            dry_run=dry_run,
            season=bundle.seasonality,
            max_blocks=max_cov_blocks,
        )
        cov_horizon = forecast_block(
            pipe,
            split.train,
            split.horizon_block,
            exog,
            split.full_h,
            batch_series,
            dry_run,
            bundle.seasonality,
        )
        cov_meta["horizon_block"] = [int(split.train.groupby(NF_ID).size().min()), split.full_h]
        cov_meta["seconds"] = round(time.perf_counter() - t0, 1)
        result["covariate"] = cov_meta
        print(
            f"[cov] done in {cov_meta['seconds']}s ({len(cov_meta['blocks'])} blocks)", flush=True
        )

    runners = {
        "tft": lambda: score_tft(bundle, split, max_steps, seed),
        "lstm": lambda: score_lstm(bundle, split, max_steps, seed),
        "cascade": lambda: score_cascade(bundle, split, cov_train, cov_horizon, max_steps, seed),
        "chronos2": lambda: score_chronos2(bundle, split, cov_horizon),
        "lgbm": lambda: score_lgbm(bundle, split, seed=seed),
        "naive": lambda: score_naive(bundle, split),
        "seasonal_naive": lambda: score_seasonal_naive(bundle, split),
        "seasonal_mean": lambda: score_seasonal_mean(bundle, split),
    }
    frames: dict[str, pd.DataFrame] = {}
    for arm in arms:
        if arm not in runners:
            result["models"][arm] = {"status": "unknown-arm"}
            continue
        t0 = time.perf_counter()
        try:
            preds, metrics = runners[arm]()
            frames[arm] = preds
            result["models"][arm] = {
                "status": "ok",
                "metrics": metrics,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            print(
                f"[{arm}] WAPE={metrics['wape']:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True
            )
        except Exception as exc:  # noqa: BLE001
            result["models"][arm] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }
            print(f"[{arm}] ERROR {exc}", flush=True)

    # TWO blends, because they answer two different questions and the write-up needs both.
    _, blend_metrics = blend(frames, tree_weight)
    if blend_metrics is not None:
        result["models"]["blend"] = {
            "status": "ok",
            "metrics": blend_metrics,
            "weight": tree_weight,
            "question": "does the model we ACTUALLY SUBMIT transfer? (shipped weight, no refit)",
        }
        print(f"[blend] w={tree_weight} (shipped) WAPE={blend_metrics['wape']:.4f}", flush=True)

    refit = fit_blend_weight(frames)
    if refit is not None:
        result["models"]["blend_refit"] = {
            "status": "ok",
            "metrics": refit["metrics"],
            "weight": refit["weight"],
            "shipped_weight": tree_weight,
            "weight_shift": round(refit["weight"] - tree_weight, 4),
            "fitted_on": refit["fitted_on"],
            "fit_rows": refit["fit_rows"],
            "fit_wape": refit["fit_wape"],
            "question": "could this architecture be TUNED here? (weight refit on the gap half)",
        }
        print(
            f"[blend_refit] w={refit['weight']} (ours {tree_weight}) "
            f"WAPE={refit['metrics']['wape']:.4f}",
            flush=True,
        )

    result["skill"] = _skill_table(result["models"])
    result["contamination_check"] = _contamination_check(result["models"])
    return result


def _skill_table(models: dict) -> dict:
    """Make the WAPEs comparable ACROSS datasets, which the raw numbers are not.

    An absolute WAPE cannot travel between corpora: ours is a smooth strictly-positive load index
    (min 0.164, mean 9.9, zero rows: none) and M5 is zero-inflated count data (49% of scored rows
    are 0, mean 1.6). A model scoring 0.13 on one and 0.80 on the other has not necessarily got
    worse — the second problem is harder in a way no metric normalises away.

    What DOES travel is **skill against a baseline defined identically on both**: the fraction of
    seasonal-naive's error a model removes. That is a ratio of two WAPEs computed on the same
    rows, so the dataset's scale and its zero rate cancel. The write-up's transfer claim should
    be made in this column, and the absolutes reported beside it as context rather than as the
    comparison.
    """

    def wape(name):
        m = models.get(name, {})
        v = m.get("metrics", {}).get("wape") if m.get("status") == "ok" else None
        return v if v is not None and np.isfinite(v) else None

    base = wape(SKILL_BASELINE)
    out = {
        "definition": "skill = 1 - wape(arm) / wape(SKILL_BASELINE)",
        "baseline": SKILL_BASELINE,
        "why": (
            "absolute WAPE is not comparable across corpora (ours has no zero rows and a mean of "
            "9.9; M5 is 49% zeros at a mean of 1.6). The ratio to a baseline defined identically "
            "on both datasets is the comparable quantity."
        ),
        "baseline_wape": base,
        "our_reference": OUR_REFERENCE,
        "our_reference_skill": (
            {
                k: round(1.0 - v / OUR_REFERENCE[SKILL_BASELINE], 4)
                for k, v in OUR_REFERENCE.items()
                if k != SKILL_BASELINE
            }
        ),
        "arms": {},
    }
    for name in models:
        w = wape(name)
        if w is None or not base:
            continue
        out["arms"][name] = round(1.0 - w / base, 4)
    return out


def _contamination_check(models: dict) -> dict:
    """S8's primary detector: is the cascade's lift over plain TFT implausibly large here?

    Calibrated on our own corpus, where contamination is impossible by construction. The branch
    is written down BEFORE the number arrives so the result cannot be read post-hoc either way.
    """

    def wape(name):
        m = models.get(name, {})
        return m.get("metrics", {}).get("wape") if m.get("status") == "ok" else None

    tft, casc = wape("tft"), wape("cascade")
    out = {
        "reference_lift_on_our_clean_data": CLEAN_CASCADE_LIFT,
        "rule": (
            "lift ~<= +8.3% -> the zero-shot transfer claim holds; lift >> +8.3% -> the corpus "
            "check is wrong and the dataset must be re-examined for pretraining overlap"
        ),
        "tft_wape": tft,
        "cascade_wape": casc,
    }
    if tft and casc and tft > 0:
        lift = (tft - casc) / tft
        out["cascade_lift_vs_tft"] = round(lift, 4)
        out["verdict"] = (
            "SUSPICIOUS — lift far above the clean reference"
            if lift > 2 * CLEAN_CASCADE_LIFT
            else "consistent with the clean reference"
        )
    else:
        out["verdict"] = "not evaluable (an arm failed)"
    return out
