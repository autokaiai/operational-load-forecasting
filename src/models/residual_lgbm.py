"""Arm E — single-model residual correction:  final = TFT_pred + LGBM(y − TFT_oof).

A LightGBM regressor (MAE) is trained to predict *only what TFT missed*, then its correction is
added to TFT's scored-block forecast. The error-correction target is built leakage-free:

  1. **TFT OOF over the train region** via ``nf.cross_validation`` on the window's train slice
     (``_hidx < cut``) with a **gapped horizon** (h=672, step=336): for each rolling origin we keep
     only the final 336 steps (k=337..672), so the OOF prediction sits at the SAME +337h horizon
     distance as the real scored block — the residual model never sees a "fresh" +1h forecast it
     would then have to extrapolate to +337h.
  2. **Residual = y − TFT_oof** on those OOF rows.
  3. The shared ``src.models.lgbm`` origin/step feature core builds the design (lags anchored at
     each origin, ``horizon_step`` = k, known-future covariates at the forecast hour); LightGBM
     fits the residual with ``objective='regression_l1'``.
  4. **Inference:** features at origin ``cut-1`` for the scored steps (k=337..672) → predicted
     residual, added to the TFT member's scored-block prediction (passed in via ``--tft-preds`` so
     we do not refit TFT a second time).

Output: ``[unique_id, ds, y, residual_corrected]`` for the scored block.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import NF_ID, NF_TARGET, NF_TIME, static_frame
from src.eval.splits import HOUR_IDX, SCORE_LEN, add_hour_index
from src.models import lgbm as lgbm_mod
from src.train import load_config

GAPPED_H = lgbm_mod.GAPPED_H  # 672
CUTOFF = "cutoff"


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tft_oof(long_df: pd.DataFrame, cut_idx: int, n_windows: int, cfg: dict) -> pd.DataFrame:
    """Rolling-origin gapped TFT OOF over the train slice -> [unique_id, ds, cutoff, y, TFT].

    Keeps only the final ``SCORE_LEN`` steps of each 672h window so every OOF row sits at the
    +337h..+672h horizon distance, matching the scored block.
    """
    from src.models.registry import build_nf, supports_stat

    df = add_hour_index(long_df)
    train = df[df[HOUR_IDX] < cut_idx].drop(columns=HOUR_IDX)
    model_col = cfg["model"]
    wbs = int(cfg.get("windows_batch_size", 128))
    gcfg = {
        **cfg,
        "h": GAPPED_H,
        "windows_batch_size": max(8, wbs // 2),
        "inference_windows_batch_size": max(8, wbs // 2),
    }
    nf = build_nf(gcfg)
    cvkwargs = {"n_windows": n_windows, "step_size": SCORE_LEN, "val_size": GAPPED_H}
    if supports_stat(model_col):
        cvkwargs["static_df"] = static_frame(train)
    cv = nf.cross_validation(train, **cvkwargs)
    del nf
    _free_gpu()

    cv = cv.sort_values([NF_ID, CUTOFF, NF_TIME])
    gapped = cv.groupby([NF_ID, CUTOFF]).tail(SCORE_LEN).reset_index(drop=True)
    return gapped.rename(columns={model_col: "TFT"})[[NF_ID, NF_TIME, CUTOFF, NF_TARGET, "TFT"]]


def _hidx_map(long_df: pd.DataFrame) -> pd.DataFrame:
    """[unique_id, ds, _hidx] lookup for mapping timestamps to per-series hour indices."""
    df = add_hour_index(long_df)
    return df[[NF_ID, NF_TIME, HOUR_IDX]]


def build_residual_samples(oof: pd.DataFrame, hmap: pd.DataFrame) -> pd.DataFrame:
    """OOF rows -> [unique_id, o, k, residual], sorted by series (build_design-aligned)."""
    fc = oof.merge(hmap, on=[NF_ID, NF_TIME], how="inner").rename(columns={HOUR_IDX: "fc_hidx"})
    origin = hmap.rename(columns={NF_TIME: CUTOFF, HOUR_IDX: "o"})
    fc = fc.merge(origin, on=[NF_ID, CUTOFF], how="inner")
    fc["k"] = fc["fc_hidx"] - fc["o"]
    fc["residual"] = fc[NF_TARGET] - fc["TFT"]
    fc = fc[fc["k"] >= 1]
    return fc.sort_values([NF_ID]).reset_index(drop=True)[[NF_ID, "o", "k", "residual"]]


def predict_residual_corrected(
    long_df: pd.DataFrame,
    cut_idx: int,
    tft_preds: pd.DataFrame,
    n_windows: int = 4,
    params: dict | None = None,
    score_len: int = SCORE_LEN,
    horizon: int = GAPPED_H,
    max_steps: int | None = None,
) -> pd.DataFrame:
    """Train residual LGBM on gapped TFT OOF, correct the TFT member's scored-block preds."""
    cfg = load_config("configs/tft.yaml")
    if max_steps:  # smoke override for the OOF TFT fit
        cfg = {**cfg, "max_steps": int(max_steps)}
    long_df = add_hour_index(long_df)
    futr_cols, stat_cols = lgbm_mod.futr_exog_list(), lgbm_mod.stat_exog_list()

    oof = tft_oof(long_df, cut_idx, n_windows, cfg)
    hmap = _hidx_map(long_df)
    samples = build_residual_samples(oof, hmap)

    Xtr, _ = lgbm_mod.build_design(
        long_df, samples[[NF_ID, "o", "k"]], futr_cols, stat_cols, with_target=False
    )
    ytr = samples["residual"].to_numpy()
    ok = np.isfinite(ytr)
    booster, feat = lgbm_mod.fit_lgbm(Xtr.loc[ok].reset_index(drop=True), ytr[ok], params)

    # inference: scored steps (k=337..672) anchored at origin cut-1
    inf = lgbm_mod.inference_pairs(long_df, cut_idx, horizon)
    inf = inf[inf["k"] > horizon - score_len].reset_index(drop=True)
    Xinf, _ = lgbm_mod.build_design(long_df, inf, futr_cols, stat_cols, with_target=False)
    resid = booster.predict(Xinf[feat])
    corr = Xinf[[NF_ID, NF_TIME]].assign(resid=resid)

    tcol = "tft" if "tft" in tft_preds.columns else tft_preds.columns[-1]
    merged = tft_preds.merge(corr, on=[NF_ID, NF_TIME], how="inner")
    merged["residual_corrected"] = np.clip(merged[tcol] + merged["resid"], 0.0, None)
    return merged[[NF_ID, NF_TIME, NF_TARGET, "residual_corrected"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Arm E residual correction (TFT + LGBM(y-TFT_oof)).")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--cut-idx", type=int, required=True)
    ap.add_argument("--tft-preds", required=True, help="path to the window's tft_preds.csv")
    ap.add_argument("--train_csv", default="data/raw/train.csv")
    ap.add_argument("--n-windows", type=int, default=4, help="rolling OOF windows for the TFT")
    ap.add_argument("--limit-series", type=int, default=0, help="first N series only (smoke)")
    ap.add_argument("--max-steps", type=int, default=0, help="override TFT max_steps (smoke)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from src.models.members import load_window_long

    long_df = load_window_long(args.train_csv, args.cut_idx)
    tft_preds = pd.read_csv(args.tft_preds, parse_dates=[NF_TIME])
    if args.limit_series > 0:
        keep = sorted(long_df[NF_ID].unique())[: args.limit_series]
        long_df = long_df[long_df[NF_ID].isin(keep)]
        tft_preds = tft_preds[tft_preds[NF_ID].isin(keep)]
    out = predict_residual_corrected(
        long_df,
        args.cut_idx,
        tft_preds,
        n_windows=args.n_windows,
        max_steps=args.max_steps or None,
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / f"window{args.window}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "residual_preds.csv"
    out.to_csv(out_csv, index=False)
    from src.metrics import wape

    w = wape(out[NF_TARGET], out["residual_corrected"])
    print(
        f"[residual_lgbm] W{args.window} cut={args.cut_idx}: "
        f"{len(out)} rows, WAPE={w:.4f} -> {out_csv}"
    )


if __name__ == "__main__":
    main()
