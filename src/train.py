"""Train one POC model and record its results.

    python -m src.train --config configs/tide.yaml

Pipeline (per the Sprint-1 plan):
1. Diagnostics (overfit estimate ONLY): rolling-origin CV + the gapped test-offset eval.
2. Final fit on the full train.csv (a small validation tail drives early stopping).
3. Save a self-contained checkpoint and write a results JSON (CV mean+/-std, train-vs-CV gap,
   gapped penalty, params, train time, versions, seed).

Cross-validation never selects the model; it only estimates generalization. The submitted
model is the full-data fit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import neuralforecast
import torch
import yaml

from src import bundle
from src.data.loader import add_volume_sample_weight, load_long, static_frame
from src.eval.cv import insample_wape, run_cross_validation, run_gapped_eval
from src.models.registry import build_nf, supports_stat

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
BASE_CONFIG = Path("configs/base.yaml")


def load_config(path: str | Path) -> dict:
    """Merge a per-model config over configs/base.yaml (model values win)."""
    base = yaml.safe_load(BASE_CONFIG.read_text()) or {}
    override = yaml.safe_load(Path(path).read_text()) or {}
    base.update(override)
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Sprint-1 POC forecasting model.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--train_csv", default=str(DATA_DIR / "train.csv"))
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument(
        "--skip-diagnostics", action="store_true", help="Skip CV + gapped eval (final fit only)."
    )
    ap.add_argument(
        "--skip-insample",
        action="store_true",
        help="Skip the in-sample train_wape pass entirely (predict_insample); use when even "
        "the subsampled pass is not worth it.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg.get("name", cfg["model"].lower())
    model_col = cfg["model"]
    has_static = supports_stat(model_col)

    long_df, fill_stats = load_long(args.train_csv)
    # Volume-weighted L1: attach a per-series sample_weight so the loss matches WAPE instead of
    # the scaler-equalized MAE (opt-in via `sample_weight: volume`). Added before CV/gapped so
    # every fit path — which all receive this long_df — trains on the same weighting.
    if cfg.get("sample_weight") == "volume":
        long_df = add_volume_sample_weight(long_df)
        print(f"[{name}] volume-weighted L1: attached per-series sample_weight (MAD)")
    static_df = static_frame(long_df) if has_static else None

    results: dict = {
        "name": name,
        "model": model_col,
        "seed": int(cfg.get("seed", 42)),
        "config": {k: v for k, v in cfg.items() if k != "loss"},
        "versions": {
            "neuralforecast": neuralforecast.__version__,
            "torch": torch.__version__,
        },
    }

    # Flush results to disk after each phase so a later-phase crash (e.g. a gapped-eval OOM that
    # poisons the CUDA context and takes the final fit down with it) never discards the CV result
    # already computed. A remote runner can then collect results/<name>.json even on a
    # non-zero exit.
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"

    def flush() -> None:
        out.write_text(json.dumps(results, indent=2))

    if not args.skip_diagnostics:
        print(f"[{name}] cross-validation (generalization estimate) ...")
        t0 = time.time()
        results["cross_validation"] = run_cross_validation(cfg, long_df, static_df)
        flush()  # persist CV before the memory-hungry gapped phase
        print(f"[{name}] gapped eval (test-offset penalty) ...")
        # The gapped eval (h=672) is the most memory-hungry phase; if it still fails after the
        # in-phase batch-halving retries, record the failure but DON'T discard the completed CV
        # + the final fit/checkpoint below — a diagnostic must never sink a successful run.
        try:
            results["gapped_eval"] = run_gapped_eval(cfg, long_df, static_df)
        except Exception as e:  # noqa: BLE001 — diagnostic is best-effort; preserve CV + checkpoint
            torch.cuda.empty_cache()
            results["gapped_eval"] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"[{name}] gapped eval FAILED ({type(e).__name__}) — keeping CV + final fit")
        results["diagnostics_seconds"] = round(time.time() - t0, 1)
        flush()  # persist CV + gapped before the final fit

    print(f"[{name}] final fit on full train.csv ...")
    t0 = time.time()
    nf = build_nf(cfg)
    fit_kwargs = {"val_size": int(cfg.get("val_size", 504))}
    if static_df is not None:
        fit_kwargs["static_df"] = static_df
    nf.fit(long_df, **fit_kwargs)
    results["train_seconds"] = round(time.time() - t0, 1)
    results["n_params"] = int(sum(p.numel() for p in nf.models[0].parameters()))

    # In-sample WAPE feeds only the train-vs-CV overfit gap; subsample the origins
    # (step_size=h, non-overlapping) so this pass stays cheap even for heavy models, or skip it.
    if args.skip_insample:
        results["train_wape"] = None
    else:
        results["train_wape"] = insample_wape(nf, model_col, step_size=int(cfg.get("h", 336)))
    if results.get("cross_validation") and results["train_wape"] is not None:
        gap = results["cross_validation"]["cv_wape_mean"] - results["train_wape"]
        results["train_vs_cv_gap"] = float(gap)

    bundle.save(nf, fill_stats, cfg, args.checkpoint)
    results["checkpoint"] = str(args.checkpoint)

    flush()
    print(f"[{name}] wrote {out}")
    if "cross_validation" in results:
        cv = results["cross_validation"]
        gapped = results.get("gapped_eval", {}).get("gapped_metrics", {})
        gapped_str = f"{gapped['wape']:.4f}" if "wape" in gapped else "FAILED"
        print(
            f"[{name}] CV WAPE {cv['cv_wape_mean']:.4f} +/- {cv['cv_wape_std']:.4f} | "
            f"train WAPE {results['train_wape']} | "
            f"gapped WAPE {gapped_str}"
        )


if __name__ == "__main__":
    main()
