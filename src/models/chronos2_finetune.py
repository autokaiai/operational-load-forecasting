"""LoRA fine-tune Chronos-2 on the project data, then score it on the SAME splits as the nf models.

Benchmark / ensemble candidate — NOT a submission model (no predict.py/bundle changes).

Training contract (verified in chronos/chronos2/dataset.py): pass ONE dict per series with
full-length arrays — `target` + `past_covariates` for all 29 futr columns — and a
`future_covariates` dict whose values are None (TRAIN ignores them; the key just *declares*
which covariates are known-future). The trainer samples random (context, horizon) windows
internally, so leakage control = cap the input series at idx < GAP_TRAIN_END_IDX (3648). Every
sampled window then stays inside the train region and the scored block (3984-4319) is never seen.

Evaluation reuses src.models.chronos2_eval.run_mode unchanged, so the fine-tuned WAPE is directly
comparable to TFT (gapped 0.136) and zero-shot Chronos-2 (gapped 0.190).

    python -m src.models.chronos2_finetune --config configs/chronos2_ft.yaml   # full run
    python -m src.models.chronos2_finetune --num-steps 20 --limit-series 3 \
        --pred-len 168 --batch-size 8 --no-eval                                 # smoke + VRAM probe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.features import futr_exog_list
from src.data.loader import NF_ID, NF_TARGET, NF_TIME, load_long
from src.eval.splits import GAP_TRAIN_END_IDX, add_hour_index
from src.models.chronos2_eval import MODEL_ID, MODES, run_mode

DEFAULTS = {
    "finetune_mode": "lora",
    "learning_rate": 1e-5,
    "num_steps": 1000,
    "batch_size": 64,
    "context_length": 2048,
    "prediction_length": 672,
    "min_past": 512,
    "seed": 42,
    "device": "cuda",
}


# `transformers.TrainingArguments`' OWN default. Ours matches it, which is why forwarding the
# seed (below) leaves every historical checkpoint bit-identical rather than silently re-fitting it.
TRAINING_ARGS_DEFAULT_SEED = 42


def trainer_extra(seed: int, device: str, no_bf16: bool) -> dict:
    """The kwargs `Chronos2Pipeline.fit` forwards VERBATIM into `TrainingArguments`.

    The seed has to travel this way because `fit` has no `seed` parameter of its own — its source
    says "Extra kwargs are directly forwarded to `TrainingArguments`", and `TrainingArguments`
    carries `seed: int = 42` which the HF `Trainer` then applies internally.

    S10 G1 found this the expensive way: three `--seed` values produced BIT-IDENTICAL checkpoints,
    because `set_seed` below seeds the global RNGs and the `Trainer` promptly re-seeds to 42. That
    is S5.2's defect in a second library — there, `pl.seed_everything(random_seed)` inside
    neuralforecast's `BaseModel` undid `registry.set_seed` and every neural run in this project
    trained at an effective seed of 1.

    Nothing recorded moves: our default seed IS 42, so the forwarded value equals the default the
    trainer was already using. Pinned by `tests/test_chronos_seed_reaches_trainer.py`.
    """
    extra: dict = {"seed": int(seed)}
    if device == "cuda" and not no_bf16:
        extra["bf16"] = True
    return extra


def set_seed(seed: int) -> None:
    """Seed numpy + torch. NOT sufficient on its own — see `trainer_extra`: the HF Trainer re-seeds
    from `TrainingArguments.seed`, so this alone does not reach the fit."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_inputs(long_df, cut_idx: int) -> list[dict]:
    """One dict per series from rows with _hidx < cut_idx (leakage-safe train region)."""
    df = add_hour_index(long_df)
    train = df[df["_hidx"] < cut_idx]
    futr_cols = futr_exog_list()
    inputs = []
    for _, g in train.groupby(NF_ID, sort=True):
        g = g.sort_values(NF_TIME)
        inputs.append(
            {
                "target": g[NF_TARGET].to_numpy(dtype=np.float32),
                "past_covariates": {c: g[c].to_numpy(dtype=np.float32) for c in futr_cols},
                # values ignored in TRAIN — the key only declares these as known-future
                "future_covariates": dict.fromkeys(futr_cols),
            }
        )
    return inputs


def _peak_vram_mb() -> dict | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LoRA fine-tune Chronos-2 and score on the project splits."
    )
    ap.add_argument("--config", type=Path, default=Path("configs/chronos2_ft.yaml"))
    ap.add_argument("--train_csv", type=Path, default=Path("data/raw/train.csv"))
    ap.add_argument("--finetune-mode", choices=["lora", "full"])
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--num-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--context-length", type=int)
    ap.add_argument(
        "--pred-len", type=int, help="fine-tune prediction_length (also caps eval horizon)"
    )
    ap.add_argument("--min-past", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--cut-idx",
        type=int,
        default=GAP_TRAIN_END_IDX,
        help="train-end _hidx (per-window adapter refit; default 3648 = W0)",
    )
    ap.add_argument("--limit-series", type=int, default=0, help="first N series only (smoke)")
    ap.add_argument(
        "--no-bf16", action="store_true", help="disable bf16 autocast (default on for cuda)"
    )
    ap.add_argument("--no-eval", action="store_true", help="fit + VRAM probe only, skip scoring")
    # SPRINT 2. Chronos-2 is fed ONE SERIES AT A TIME, so a cross-series aggregate is the only
    # channel through which it can see the rest of the panel at all. `build_train_inputs` reads
    # `futr_exog_list()` at call time, so activating the channel here reaches `past_covariates`
    # and the `future_covariates` declaration together — but the COLUMNS must be materialised on
    # the frame first or the dict comprehension raises KeyError.
    ap.add_argument("--xs-block", default=None, help="cross-series block to attach, e.g. A9")
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints/chronos2_ft"))
    ap.add_argument("--out", type=Path, default=Path("results/chronos2_ft.json"))
    args = ap.parse_args()

    cfg = {**DEFAULTS, **(yaml.safe_load(args.config.read_text()) if args.config.exists() else {})}
    # CLI overrides config
    over = {
        "finetune_mode": args.finetune_mode,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "prediction_length": args.pred_len,
        "min_past": args.min_past,
        "seed": args.seed,
        "device": args.device,
    }
    cfg.update({k: v for k, v in over.items() if v is not None})

    set_seed(int(cfg["seed"]))
    device = cfg["device"]
    pred_len = int(cfg["prediction_length"])

    # Full labelled frame for EVAL (run_mode carves context per mode); train uses only _hidx<3648.
    long_df, _ = load_long(args.train_csv)
    if args.limit_series > 0:
        keep = sorted(long_df[NF_ID].unique())[: args.limit_series]
        long_df = long_df[long_df[NF_ID].isin(keep)]

    cut_idx = int(args.cut_idx)
    xs_names: list[str] = []
    if args.xs_block:
        import contextlib as _ctx

        from src.data.features import aggregate_columns
        from src.data.xs_blocks import BLOCKS, NEEDS_CUT, _zone_labels
        from src.eval.splits import add_hour_index

        _, fn = BLOCKS[args.xs_block]
        long_df = add_hour_index(long_df)
        zones = _zone_labels(long_df)
        long_df, xs_names = (
            fn(long_df, zones, cut_idx) if args.xs_block in NEEDS_CUT else fn(long_df, zones)
        )
        if long_df[xs_names].isna().any().any():
            raise SystemExit(f"[xs] {args.xs_block} produced NaN; aggregate AFTER imputation")
        scope = aggregate_columns(xs_names)
    else:
        import contextlib as _ctx

        scope = _ctx.nullcontext()

    with scope:
        if xs_names:
            # LAW 5, checked on the real conditioning set: a channel that does not reach
            # `futr_exog_list()` would train Chronos without the aggregates and return a
            # confident null after a full fine-tune.
            from src.data.features import futr_exog_list as _fx

            gone = [c for c in xs_names if c not in _fx()]
            if gone:
                raise SystemExit(f"[xs] {gone} never reached futr_exog_list()")
            print(f"[xs] {args.xs_block}: {len(xs_names)} aggregate columns -> chronos covariates")
        train_inputs = build_train_inputs(long_df, cut_idx)
    print(
        f"Built {len(train_inputs)} training series (rows _hidx<{cut_idx}); "
        f"mode={cfg['finetune_mode']} h={pred_len} steps={cfg['num_steps']} bs={cfg['batch_size']}"
    )

    from chronos import Chronos2Pipeline

    print(f"Loading {MODEL_ID} on {device} ...")
    pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)

    extra = trainer_extra(cfg["seed"], device, args.no_bf16)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    ft = pipe.fit(
        train_inputs,
        prediction_length=pred_len,
        finetune_mode=cfg["finetune_mode"],
        learning_rate=float(cfg["learning_rate"]),
        num_steps=int(cfg["num_steps"]),
        batch_size=int(cfg["batch_size"]),
        context_length=int(cfg["context_length"]),
        min_past=int(cfg["min_past"]),
        output_dir=str(args.output_dir),
        remove_printer_callback=True,
        **extra,
    )
    train_seconds = round(time.perf_counter() - t0, 1)
    vram = _peak_vram_mb()
    print(f"Fine-tune done in {train_seconds}s | peak VRAM: {vram}")

    # Self-eval is skipped when (a) --no-eval, or (b) there is no horizon to score: at the
    # submission cutoff (cut_idx == data end, e.g. 4320) no rows sit at _hidx >= cut_idx, so the
    # gapped horizon is empty and run_mode would `int(NaN)`-crash. The adapter is what we keep;
    # WAPE comes from the CV harness + the leaderboard, never from this in-process score.
    n_horizon = int((add_hour_index(long_df)["_hidx"] >= cut_idx).sum())
    if args.no_eval or n_horizon == 0:
        why = "--no-eval set" if args.no_eval else f"no horizon rows at _hidx>={cut_idx}"
        print(f"Skipping self-eval ({why}); adapter saved to {args.output_dir}.")
        return

    results = {}
    for mode in MODES:
        tm = time.perf_counter()
        results[mode] = {
            **run_mode(ft, long_df, mode, pred_len, batch_series=0, cut_idx=cut_idx),
            "seconds": None,
        }
        results[mode]["seconds"] = round(time.perf_counter() - tm, 1)
        print(
            f"   {mode}: WAPE={results[mode]['metrics']['wape']:.4f} "
            f"(h={results[mode]['h']}, n_series={results[mode]['n_series']})"
        )

    import chronos as _chronos  # noqa: F401

    out = {
        "name": "chronos2_ft",
        "model": f"Chronos-2 ({cfg['finetune_mode']} FT)",
        "seed": int(cfg["seed"]),
        "zero_shot": False,
        "model_id": MODEL_ID,
        "finetune": {
            k: cfg[k]
            for k in (
                "finetune_mode",
                "learning_rate",
                "num_steps",
                "batch_size",
                "context_length",
                "prediction_length",
                "min_past",
            )
        },
        "versions": {
            "chronos-forecasting": getattr(_chronos, "__version__", "unknown"),
            "peft": __import__("peft").__version__,
        },
        "train_seconds": train_seconds,
        "peak_vram": vram,
        "n_params": None,
        "train_wape": None,
        "chronos2_modes": results,
    }
    out["gapped_eval"] = {
        "gapped_h": results["gapped-optimistic"]["h"],
        "score_len": results["gapped-optimistic"]["score_len"],
        "gapped_metrics": results["gapped-optimistic"]["metrics"],
    }
    out["cross_validation"] = {
        "n_windows": 1,
        "note": "single contiguous +1h-offset 336h block (nogap), not CV",
        "cv_wape_mean": results["nogap"]["metrics"]["wape"],
        "cv_wape_std": 0.0,
        "cv_metrics_pooled": results["nogap"]["metrics"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
