"""Zero-shot Chronos-2 benchmark — NOT a neuralforecast model, NOT a submission candidate.

Chronos-2 (amazon/chronos-2, Oct 2025) is an encoder-only, direct multi-step foundation model
with native known-future-covariate support and a 1024-step max horizon. So we never roll it
step-by-step: we forecast the full horizon in one ``predict_df`` call and slice the scored block.

This scores it on the SAME splits and metrics as the trained nf models (see src.eval.cv /
src.eval.splits / src.metrics) and writes results/chronos2.json in the schema
scripts/summarize_results.py reads, so it tables head-to-head with DLinear/TiDE/TSMixerx/TFT.

Three modes isolate covariate quality from the +337h gap penalty:
  * gapped-optimistic : mirrors src.eval.cv.run_gapped_eval (gap covariates SUPPLIED locally) —
                        apples-to-apples vs the nf models' reported gapped_eval.
  * gapped-realistic  : gap planning-signal covariates WITHHELD (median-imputed + missing=1),
                        calendar kept — the true private-test condition (the Sprint-2 concern).
  * nogap             : contiguous +1h-offset 336h block (validation-like), full covariates.

    python -m src.models.chronos2_eval --mode all
    python -m src.models.chronos2_eval --mode gapped-optimistic --limit-series 3  # smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from src.data.features import KNOWN_FUTURE_SIGNALS, futr_exog_list, missing_indicator_cols
from src.data.loader import NF_ID, NF_TARGET, NF_TIME, load_long
from src.eval.splits import (
    BLOCK_REGIMES,
    GAP_TRAIN_END_IDX,
    SCORE_LEN,
    add_hour_index,
    gapped_horizon,
    take_block,
)
from src.metrics import all_metrics

MODEL_ID = "amazon/chronos-2"
MODES = ("gapped-optimistic", "gapped-realistic", "nogap")


def _load_pipeline(device: str, adapter: str | None = None):
    """Load Chronos-2; ``adapter`` (a LoRA fine-tune dir) loads the fine-tuned pipeline instead.

    Import lazily so the rest of the module (and --help) works without chronos installed.
    """
    from chronos import Chronos2Pipeline

    source = adapter if adapter else MODEL_ID
    return Chronos2Pipeline.from_pretrained(str(source), device_map=device)


def _pick_pred_column(pred: pd.DataFrame) -> str:
    """Find the point-forecast column in predict_df output (median 0.5 preferred)."""
    for cand in ("0.5", 0.5, "predictions", "median", "mean"):
        if cand in pred.columns:
            return cand  # type: ignore[return-value]
    numeric = [
        c
        for c in pred.columns
        if c not in (NF_ID, NF_TIME) and pd.api.types.is_numeric_dtype(pred[c])
    ]
    if len(numeric) == 1:
        return numeric[0]
    raise RuntimeError(f"Cannot identify prediction column among {list(pred.columns)}")


def _withhold_gap_covariates(
    future: pd.DataFrame,
    train: pd.DataFrame,
    gap_len: int,
    strategy: str = "median",
    cut_idx: int | None = None,
) -> pd.DataFrame:
    """Realistic test condition: blank the gap's planning signals (calendar stays known).

    Replaces the first ``gap_len`` rows/series of every KNOWN_FUTURE_SIGNAL with a reconstruction
    fitted on the TRAIN slice, and sets the matching ``*_missing`` flags to 1 — exactly the state
    src.data.impute would produce when a covariate row is absent at inference.

    ``strategy`` selects that reconstruction from ``src.data.gap_fill``. ``"median"`` is the
    incumbent (per-series train median, global fallback) and is bit-exact with what this function
    always did. Measured against the true withheld covariates over three cutoffs, the flat median
    reconstructs at 0.5333 while the per-series hour-of-week profile reaches 0.3667 — the
    incumbent throws away ~31% of the recoverable signal. ``"+exact"`` additionally restores any
    covariate that is a *deterministic* function of (series, hour-of-week); ``workload_intensity``
    is one, carries no ``*_missing`` flag because it has no NaNs to flag, and is therefore
    currently corrupted over the gap without the model being told.

    Only rows inside the gap are touched. The scored block that follows keeps its real covariates
    and is invisible to the statistic — the reconstruction is fitted on ``train`` alone.
    """
    from src.data.gap_fill import reconstruct_block

    future = future.sort_values([NF_ID, NF_TIME]).copy()
    gap_mask = future.groupby(NF_ID).cumcount() < gap_len
    filled = reconstruct_block(
        target=future,
        history=train,
        cols=KNOWN_FUTURE_SIGNALS,
        fill_mask=gap_mask,
        strategy=strategy,
        id_col=NF_ID,
        time_col=NF_TIME,
        cut_idx=cut_idx,
    )
    for col in KNOWN_FUTURE_SIGNALS:
        future.loc[gap_mask, col] = filled.loc[gap_mask, col]
    for col in missing_indicator_cols():
        if col in future.columns:
            future.loc[gap_mask, col] = 1.0
    return future


def _predict(
    pipe, context: pd.DataFrame, future: pd.DataFrame, h: int, batch_series: int
) -> pd.DataFrame:
    """Run predict_df, optionally chunking series to bound GPU memory; returns long preds."""
    ids = context[NF_ID].unique().tolist()
    chunks = (
        [ids]
        if batch_series <= 0
        else [ids[i : i + batch_series] for i in range(0, len(ids), batch_series)]
    )
    out = []
    for chunk in chunks:
        cset = set(chunk)
        c = context[context[NF_ID].isin(cset)]
        f = future[future[NF_ID].isin(cset)]
        pred = pipe.predict_df(
            c,
            future_df=f,
            prediction_length=h,
            quantile_levels=[0.5],
            id_column=NF_ID,
            timestamp_column=NF_TIME,
            target=NF_TARGET,
        )
        out.append(pred)
    return pd.concat(out, ignore_index=True)


def run_mode(
    pipe,
    long_df: pd.DataFrame,
    mode: str,
    pred_len: int | None,
    batch_series: int,
    return_preds: bool = False,
    cut_idx: int = GAP_TRAIN_END_IDX,
    regime: str = "far",
):
    """Score one mode -> {metrics, h, score_len, n_series}; with return_preds also the merged df.

    ``cut_idx`` parametrizes the train-end (default ``GAP_TRAIN_END_IDX``) so the multi-window
    architecture test can roll the origin back and produce FT-Chronos standalone preds per window.

    ``regime`` picks which half of the gapped horizon is scored — ``far`` (steps 337-672, the
    default and the graded block) or ``near`` (steps 1-336). One forward pass produces both, so the
    near cube is inference-free given a fitted checkpoint (final-push lane 1E). It is meaningless
    for ``nogap``, whose horizon IS the scored block, and raises there rather than silently
    returning the same rows under a second name.
    """
    if regime not in BLOCK_REGIMES:
        raise ValueError(f"regime must be one of {BLOCK_REGIMES}, got {regime!r}")
    if regime != "far" and mode == "nogap":
        raise ValueError("mode='nogap' has no gap, so its near and far blocks are the same rows")
    if regime == "near" and mode == "gapped-realistic":
        # Argument-level, before any data is touched: `_withhold_gap_covariates` blanks the first
        # 336 rows per series, which under regime="near" ARE the rows being scored. The near regime
        # is the validation scenario — no gap, real known-future covariates.
        raise ValueError(
            "regime='near' with mode='gapped-realistic' would withhold the covariates of the rows "
            "being scored. Use mode='gapped-optimistic' (gap_cov='real') for near."
        )
    df = add_hour_index(long_df)
    futr_cols = futr_exog_list()

    if mode == "nogap":
        cut = cut_idx + SCORE_LEN  # train .. then a contiguous 336h scored block
        score_len = SCORE_LEN
        target_h = SCORE_LEN  # contiguous 336h, no gap
    else:  # gapped-* : train to cut_idx, horizon = gap(336) + scored(336)
        cut = cut_idx
        score_len = SCORE_LEN
        target_h = 2 * SCORE_LEN  # 672

    # Fixed target window [cut, cut+target_h) via the shared helper — NOT every row >= cut, which
    # over-runs at earlier cutoffs and would mis-score the wrong block (see splits.gapped_horizon).
    horizon_df = gapped_horizon(df, cut, target_h)
    full_h = int(horizon_df.groupby(NF_ID).size().min())
    h = min(full_h, pred_len) if pred_len else full_h
    score_len = min(score_len, h)
    gap_len = h - score_len  # 0 for nogap, 336 for full gapped

    train = df[df["_hidx"] < cut]
    context = train[[NF_ID, NF_TIME, NF_TARGET, *futr_cols]].copy()
    future = (
        horizon_df.sort_values([NF_ID, NF_TIME])
        .groupby(NF_ID)
        .head(h)[[NF_ID, NF_TIME, *futr_cols]]
        .copy()
    )
    if mode == "gapped-realistic" and gap_len > 0:
        future = _withhold_gap_covariates(future, train, gap_len)

    pred = _predict(pipe, context, future, h, batch_series)
    pcol = _pick_pred_column(pred)
    pred = pred.sort_values([NF_ID, NF_TIME])
    score_pred = take_block(pred, regime, score_len)[[NF_ID, NF_TIME, pcol]].copy()
    score_pred[pcol] = score_pred[pcol].clip(lower=0.0)  # mirror predict.py

    labels = take_block(
        horizon_df.sort_values([NF_ID, NF_TIME]).groupby(NF_ID).head(h), regime, score_len
    )
    labels = labels[[NF_ID, NF_TIME, NF_TARGET]]
    merged = score_pred.merge(labels, on=[NF_ID, NF_TIME])
    if len(merged) != len(labels):
        raise RuntimeError(f"{mode}: pred/label misalignment ({len(merged)} vs {len(labels)})")

    result = {
        "metrics": all_metrics(merged[NF_TARGET], merged[pcol]),
        "h": h,
        "score_len": score_len,
        "cut_idx": cut_idx,
        "n_series": int(merged[NF_ID].nunique()),
    }
    if return_preds:
        preds_out = merged.rename(columns={pcol: "prediction"})[
            [NF_ID, NF_TIME, NF_TARGET, "prediction"]
        ]
        return result, preds_out
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot Chronos-2 benchmark on the project splits.")
    ap.add_argument("--config", type=Path, default=Path("configs/chronos2.yaml"))
    ap.add_argument("--train_csv", type=Path, default=Path("data/raw/train.csv"))
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
    ap.add_argument("--device", default="cuda", help="cuda | cpu | mps")
    ap.add_argument(
        "--batch-series", type=int, default=0, help="series per predict_df call (0 = all at once)"
    )
    ap.add_argument(
        "--limit-series", type=int, default=0, help="use only the first N series (smoke test)"
    )
    ap.add_argument(
        "--pred-len", type=int, default=0, help="cap prediction_length (smoke test); 0 = full"
    )
    ap.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA fine-tune dir (FT-Chronos); zero-shot if unset",
    )
    ap.add_argument(
        "--cut-idx", type=int, default=GAP_TRAIN_END_IDX, help="train-end _hidx (per-window)"
    )
    ap.add_argument("--out", type=Path, default=Path("results/chronos2.json"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    device = cfg.get("device", args.device)
    batch_series = int(cfg.get("batch_series", args.batch_series))

    long_df, _ = load_long(args.train_csv)
    if args.limit_series > 0:
        keep = sorted(long_df[NF_ID].unique())[: args.limit_series]
        long_df = long_df[long_df[NF_ID].isin(keep)]

    src = args.adapter if args.adapter else MODEL_ID
    print(f"Loading {src} on {device} ...")
    pipe = _load_pipeline(device, adapter=str(args.adapter) if args.adapter else None)

    modes = list(MODES) if args.mode == "all" else [args.mode]
    pred_len = args.pred_len or None
    results = {}
    for mode in modes:
        t0 = time.perf_counter()
        print(f"== mode: {mode} ==")
        results[mode] = {
            **run_mode(pipe, long_df, mode, pred_len, batch_series, cut_idx=args.cut_idx),
            "seconds": None,
        }
        results[mode]["seconds"] = round(time.perf_counter() - t0, 1)
        print(
            f"   {mode}: WAPE={results[mode]['metrics']['wape']:.4f} "
            f"(h={results[mode]['h']}, n_series={results[mode]['n_series']}, "
            f"{results[mode]['seconds']}s)"
        )

    # Map into the results-JSON schema summarize_results.py understands:
    #   gapped-optimistic -> gapped_eval.gapped_metrics (the nf models' "gapped" column)
    #   nogap             -> cross_validation.cv_wape_mean (validation-like, single window)
    import chronos as _chronos  # noqa: F401 — for version string

    out = {
        "name": "chronos2",
        "model": "Chronos-2 (zero-shot)",
        "seed": None,
        "zero_shot": True,
        "model_id": MODEL_ID,
        "versions": {"chronos-forecasting": getattr(_chronos, "__version__", "unknown")},
        "train_seconds": 0.0,
        "n_params": None,
        "train_wape": None,
        "chronos2_modes": results,  # full metrics for all three modes (source of truth)
    }
    if "gapped-optimistic" in results:
        out["gapped_eval"] = {
            "gapped_h": results["gapped-optimistic"]["h"],
            "score_len": results["gapped-optimistic"]["score_len"],
            "gapped_metrics": results["gapped-optimistic"]["metrics"],
        }
    if "nogap" in results:
        out["cross_validation"] = {
            "n_windows": 1,
            "note": "single contiguous +1h-offset 336h block (validation-like), not 3-fold",
            "cv_wape_mean": results["nogap"]["metrics"]["wape"],
            "cv_wape_std": 0.0,
            "cv_metrics_pooled": results["nogap"]["metrics"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
