"""Overfitting / generalization diagnostics via neuralforecast.

Cross-validation here is PURELY a generalization estimate, never model selection; the gapped
eval sizes the private-test offset penalty. Neither produces the submitted model — that is
trained on the full train.csv by ``src.train``.
"""

from __future__ import annotations

import gc

import numpy as np
import torch

from src.data.features import futr_exog_list
from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import (
    CV_N_WINDOWS,
    CV_STEP_SIZE,
    GAP_TRAIN_END_IDX,
    HOUR_IDX,
    SCORE_LEN,
    add_hour_index,
    gapped_horizon,
)
from src.metrics import all_metrics, wape
from src.models.registry import build_nf, supports_futr


def _free_gpu() -> None:
    """Release cached GPU memory between phases so the diagnostics don't stack and OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_cross_validation(
    cfg: dict,
    long_df,
    static_df=None,
    n_windows: int = CV_N_WINDOWS,
    step_size: int = CV_STEP_SIZE,
) -> dict:
    """Rolling-origin CV (contiguous, validation-like) -> per-fold + pooled WAPE."""
    nf = build_nf(cfg)
    model_col = cfg["model"]
    # val_size reserves a validation tail inside CV training so the early-stopping monitor
    # (ptl/val_loss) exists; without it neuralforecast raises when early_stop_patience_steps > 0.
    kwargs = {
        "n_windows": n_windows,
        "step_size": step_size,
        "val_size": int(cfg.get("val_size", 504)),
    }
    if static_df is not None:
        kwargs["static_df"] = static_df
    cv = nf.cross_validation(long_df, **kwargs)

    per_fold = [wape(g[NF_TARGET], g[model_col]) for _, g in cv.groupby("cutoff")]
    result = {
        "n_windows": n_windows,
        "step_size": step_size,
        "per_fold_wape": [float(x) for x in per_fold],
        "cv_wape_mean": float(np.mean(per_fold)),
        "cv_wape_std": float(np.std(per_fold)),
        "cv_metrics_pooled": all_metrics(cv[NF_TARGET], cv[model_col]),
    }
    del nf
    _free_gpu()
    return result


def insample_wape(nf, model_col: str, step_size: int = 1) -> float | None:
    """In-sample (train) WAPE via ``predict_insample`` — for the train-vs-CV overfit gap.

    ``step_size`` subsamples the forecast origins. neuralforecast's default of 1 forecasts at
    *every* origin (~hundreds of thousands of windows here) — a pathologically slow pass for
    heavy per-forward models like TimesNet, and a no-op that returns None for some multivariate
    models. Passing ``step_size=h`` uses non-overlapping windows (~h x fewer), which is fast for
    every model and still a representative train-WAPE estimate.
    """
    try:
        ins = nf.predict_insample(step_size=step_size)
    except Exception:
        return None
    ins = ins.dropna(subset=[model_col, NF_TARGET])
    if ins.empty:
        return None
    return float(wape(ins[NF_TARGET], ins[model_col]))


def run_gapped_eval(cfg: dict, long_df, static_df=None) -> dict:
    """Gapped eval at the default cutoff (``GAP_TRAIN_END_IDX``). Thin wrapper — see below."""
    return run_gapped_eval_at(cfg, long_df, GAP_TRAIN_END_IDX, static_df=static_df)


def run_gapped_eval_at(cfg: dict, long_df, cut_idx: int, static_df=None) -> dict:
    """Train to ``cut_idx``, forecast the +337h-offset 336h block, score it (test-like).

    Identical to the headline gapped eval but with the train-end cutoff parametrized, so the
    multi-window architecture test (``splits.cv_window_cutoffs``) can roll the origin back and
    score disjoint blocks under the same h=672 / last-336 semantics. ``cut_idx`` is a per-series
    hour index (``_hidx``); train = rows ``< cut_idx``, horizon = the rest (gap 336 + scored 336).

    Note: locally we still supply the gap's covariates (they exist in train.csv), so this is
    an *optimistic* gap estimate — at real test time the gap covariates are absent. It isolates
    the horizon-distance penalty; the covariate-absent case is a Sprint-2 concern.
    """
    model_col = cfg["model"]
    df = add_hour_index(long_df)
    train = df[df[HOUR_IDX] < cut_idx].drop(columns=HOUR_IDX)
    # Fixed 672h gapped window [cut, cut+672) via the shared helper — NOT every row >= cut_idx,
    # which over-runs at earlier cutoffs and mis-scores the wrong block (see gapped_horizon).
    horizon = gapped_horizon(df, cut_idx).drop(columns=HOUR_IDX)
    h = int(horizon.groupby(NF_ID).size().min())  # == 672 at every cutoff

    # The gapped horizon (h=672) ~doubles per-window memory; halve the training window batch so
    # the heaviest model (TFT) keeps headroom on the 24GB GPU. Diagnostic-only — does not affect
    # the headline CV / full-fit numbers. On an OutOfMemoryError we keep halving and refit rather
    # than crashing the whole run: a deterministic memory-pressure adaptation in one already
    # leased container — not a blind whole-job retry — and the lowest batch that fits is recorded.
    #
    # Crucially we halve `batch_size` (series per batch), NOT just `windows_batch_size`. The big
    # allocation is the `temporal.unfold(...).flatten()` in _create_windows, which materializes
    # every window for the whole series-batch BEFORE subsampling to windows_batch_size — so its
    # peak scales with batch_size and the h=672 window length, and windows_batch_size can't shrink
    # it. (This is why the LSTM volume-weighted runs OOM'd even at windows_batch_size=8.)
    #
    # We start PROACTIVELY at half batch_size, not the configured value. A CUDA OOM during the
    # backward pass poisons the process's CUDA context — a later in-process retry at a smaller
    # batch then fails anyway and, worse, the subsequent full-data fit dies too. So the goal is to
    # never hit the first OOM: begin lean for this one heavy diagnostic. The reactive halving below
    # remains a backstop, but in practice the proactive start means it is rarely needed.
    bs = max(2, int(cfg.get("batch_size", 32)) // 2)
    wbs = max(8, int(cfg.get("windows_batch_size", 128)) // 2)
    futr = horizon[[NF_ID, NF_TIME, *futr_exog_list()]] if supports_futr(cfg["model"]) else None
    nf = preds = None
    while True:
        gcfg = {
            **cfg,
            "h": h,
            "batch_size": bs,
            "windows_batch_size": min(wbs, bs * 4),
            "inference_windows_batch_size": min(wbs, bs * 4),
        }
        try:
            nf = build_nf(gcfg)
            fit_kwargs = {"val_size": h}
            if static_df is not None:
                fit_kwargs["static_df"] = static_df
            nf.fit(train, **fit_kwargs)
            preds = nf.predict() if futr is None else nf.predict(futr_df=futr)
            break
        except torch.cuda.OutOfMemoryError:
            del nf
            nf = None
            _free_gpu()
            if bs <= 2:
                raise
            bs = max(2, bs // 2)
            print(f"[gapped] OOM — retrying at batch_size={bs}", flush=True)

    score_preds = preds.groupby(NF_ID).tail(SCORE_LEN)
    score_labels = horizon.groupby(NF_ID).tail(SCORE_LEN)[[NF_ID, NF_TIME, NF_TARGET]]
    merged = score_preds.merge(score_labels, on=[NF_ID, NF_TIME])
    result = {
        "cut_idx": cut_idx,
        "gapped_h": h,
        "score_len": SCORE_LEN,
        "gapped_metrics": all_metrics(merged[NF_TARGET], merged[model_col]),
    }
    del nf
    _free_gpu()
    return result
