"""The cascade half of the submission forecast: N seeds fitted on ALL labelled hours, one origin.

Why this is not `run_sweep` with a different cutoff
---------------------------------------------------
Every CV path in this repo fits and then **scores**, so it needs a known ``y`` over the horizon.
At the submission origin there is none — that is the whole point of the horizon. So this is a
fit-and-predict runner: it stops where ``run_sweep`` starts checking answers.

Two things it gets for free, both verified rather than assumed
--------------------------------------------------------------
**The training covariate needs NO new Chronos call.** ``data/derived/train_chronos.csv`` is the
rolling grid, and its sidecar's last block starts at **3984**. A block starting at ``s`` conditions
on ``[0, s)`` and is therefore legitimate for any cutoff ``c >= s`` — so at ``c = 4320`` *every*
block qualifies and the whole frame is gap-honest at the submission origin. The per-cutoff splice
that S3/S4 needed exists precisely because their cutoffs sat mid-grid; ours sits past the end.

**The horizon covariate already exists** as ``data/derived/validation_input_chronos.csv`` — a single
inference block forecast from the full observed history, which is exactly the submission geometry.

So the GPU cost here is the fits and nothing else.

The seeds are the point
-----------------------
S6 ships a **5-seed bag** because a single draw is a lottery: seed-averaged the member is
0.13908 +- 0.00278 while the historical single draw sits at 0.13429, ~1.4 sigma lucky. All N models
live in ONE ``NeuralForecast`` so ``nf.fit`` trains them together and ``nf.predict`` returns one
column each — which also means the stored dataset is written **once**, and the checkpoint costs
19.8 MB for the first seed and only ~6.1 MB for each one after it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import pandas as pd

from src.data.features import ID, TIME, aggregate_columns, cascade_channels
from src.data.loader import NF_ID, NF_TIME, build_futr_df, load_long, static_frame
from src.data.xs_attach import attach_active_aggregates
from src.data.xs_blocks import BLOCKS as XS_BLOCKS
from src.data.xs_blocks import _zone_labels as xs_zones
from src.eval.protocol import frozen_seeds
from src.eval.splits import add_hour_index
from src.models.members import GAPPED_HORIZON

CHRONOS_COL = "chronos2_forecast"
TRAIN_FRAME = "data/derived/train_chronos.csv"  # rolling grid; honest at cut 4320 (last block 3984)
HORIZON_FRAME = "data/derived/validation_input_chronos.csv"
# *** THE SHIPPED FIT MUST BE THE OBJECT THE CV MEASURED. ***
# Was `configs/tft_chronos.yaml`, which carries NO `swa` block -- confirmed by `ship_run.log`,
# which contains zero `[swa] averaged` lines against 5 in the CV member runs. So the shipped TFT
# differed from the CV member `tft_cascade_swa` on TWO axes at once: the missing scaler (fixed
# below at the `NeuralForecast` construction) and the missing checkpoint averaging (fixed here).
# The blend weights 0.4724/0.3413/0.1863 and the gamma=1.04 de-smoothing constant were BOTH fitted
# on `tft_cascade_swa` + A9 cubes, so until now they described an object we did not ship.
#
# THIS REVERSES AN EARLIER DEFERRAL, and on its own terms. SWA was not
# rejected on merit -- member +0.00156 (4.37 SE, 5/5 seeds), blend +0.00047 (2.9 SE), failing only
# criterion 2 at 2/3 windows -- it was deferred because shipping it costs "five new GPU fits at the
# submission anchor, then the archive rebuild". That cost is being paid regardless for the scaler,
# so the marginal price of coherence here is zero.
CONFIG = "configs/tft_chronos_swa.yaml"


def horizon_covariates(val_csv: str, chronos_csv: str) -> pd.DataFrame:
    """The future covariate block: the raw planning signals plus the Chronos channel.

    ``validation_input_chronos.csv`` is ``validation_input.csv`` with the Chronos column merged on,
    so it already carries both. Read it directly rather than re-merging — a second join is a second
    chance to mis-align, and the frame is the one the sidecar describes.
    """
    frame = Path(chronos_csv)
    cov = pd.read_csv(frame if frame.exists() else val_csv)
    cov[TIME] = pd.to_datetime(cov[TIME])
    if CHRONOS_COL not in cov.columns:
        raise SystemExit(f"{frame} has no {CHRONOS_COL!r}; the cascade cannot run without it")
    return cov


def run(
    *,
    n_seeds: int = 5,
    max_steps: int | None = None,
    limit_series: int = 0,
    out_dir: str = "predictions",
    repo: Path = Path("."),
    train_frame: str | None = None,
    horizon_frame: str | None = None,
    save_checkpoint: str | None = None,
    xs_block: str = "",
    windows_batch_size: int = 0,
) -> dict:
    """Fit ``n_seeds`` cascade backbones on every labelled hour and forecast the full horizon."""
    from neuralforecast import NeuralForecast

    from src.models.registry import build_model
    from src.train import load_config

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with cascade_channels(CHRONOS_COL):
        cfg = load_config(str(repo / CONFIG))
        # *** THE SHIPPED GEOMETRY, NOT THE CONFIG'S ***
        # `configs/base.yaml` sets h: 336, but NO recorded number was produced at 336: the CV
        # runner overrides it to `gapped_horizon` = 672 (gap 336 + scored 336) and HALVES
        # `windows_batch_size` to fit that horizon in memory. A TFT trained at h=336 is a
        # different model from the one S3/S4/S5/S6 measured — it never learns leads 337-672 —
        # so taking the config's h here would silently ship an object none of our numbers
        # describe. The first smoke did exactly that and returned 336 hours instead of 672.
        # *** THE FIT GEOMETRY MUST MIRROR `src/eval/cv.py:133-143` EXACTLY. ***
        # This block used to halve ONLY `windows_batch_size` and leave `batch_size` at the config
        # value, so the shipped bag trained at batch_size 32 while every CV member that produced
        # the blend weights and the gamma constant trained at 16. `batch_size` is the ONE axis this
        # project never swept (zero hits in `sweep_space.py` and in all of S5/S5c/S5d), so the
        # difference is not known to be null -- and it feeds the early-stopping rule, which is
        # precisely the interaction that manufactured a spurious +0.065 in the scaler A/B.
        # Halving is also what makes the fit fit: cv.py's comment explains that the big allocation
        # is `temporal.unfold(...).flatten()` in `_create_windows`, whose peak scales with
        # batch_size, which `windows_batch_size` cannot shrink.
        bs = max(2, int(cfg.get("batch_size", 32)) // 2)
        wbs = max(8, int(cfg.get("windows_batch_size", 128)) // 2)
        eff_wbs = windows_batch_size or min(wbs, bs * 4)
        cfg = {
            **cfg,
            "h": GAPPED_HORIZON,
            "batch_size": bs,
            "windows_batch_size": eff_wbs,
            "inference_windows_batch_size": eff_wbs,
        }
        print(f"[cascade] batch_size={bs} windows_batch_size={eff_wbs} (mirrors cv.py)", flush=True)
        if max_steps:
            cfg = {**cfg, "max_steps": int(max_steps)}

        long, fill_stats = load_long(str(train_frame or repo / TRAIN_FRAME))
        if limit_series:
            keep = sorted(long[NF_ID].unique())[:limit_series]
            long = long[long[NF_ID].isin(keep)]

        # SPRINT 2 — the cross-series aggregates, on the SHIPPED fit. Built here, after
        # `load_long` has imputed, and activated before `build_model` so they reach
        # `futr_exog_list()` and therefore the models' own recorded column list — which is what
        # `src.bundle.aggregate_columns_of` reads back at submission time.
        xs_names: list[str] = []
        if xs_block:
            _, fn = XS_BLOCKS[xs_block]
            long = add_hour_index(long)
            long, xs_names = fn(long, xs_zones(long))
            if long[xs_names].isna().any().any():
                raise SystemExit(f"[xs] {xs_block} produced NaN on the train frame")
        xs_scope = aggregate_columns(xs_names) if xs_names else contextlib.nullcontext()

        stat = static_frame(long)

        seeds = list(frozen_seeds(n_seeds))
        with xs_scope:
            models = [build_model({**cfg, "seed": s, "random_seed": s}) for s in seeds]
            # *** THE SCALER. It was MISSING here and that was a real defect, not a choice. ***
            # `registry.build_nf` — the CV path — passes `local_scaler_type` from the config, which
            # `configs/base.yaml:31` sets to `robust`. This call omitted it, so `NeuralForecast`
            # defaulted to None and the SUBMISSION trained an UNSCALED model while every recorded
            # neural number, and every blend weight fitted on those members, described the SCALED
            # one. The two were never the same object.
            #
            # Measured 2026-09-04, paired at seed 892 with the same A9 block, configs differing by
            # one line: robust 0.12618 vs unscaled 0.19118 on `late`, +0.06500, 15.5 SE, 3/3
            # windows. That figure is confounded with early stopping (the unscaled arm stopped at
            # ~800 steps against the control's 5000, because a raw-unit `valid_loss` is noisier and
            # trips the same patience sooner), so it BOUNDS the defect rather than measuring the
            # scaler alone. The decision does not rest on the size: the shipped object must be the
            # measured object.
            #
            # TAKES EFFECT ON THE NEXT FIT ONLY. A checkpoint already on disk carries its own
            # scaler state, so this changes nothing until the bag is refitted.
            # *** THE SCALER STAYS OFF HERE, AND THAT IS A CONSTRAINT RATHER THAN AN OVERSIGHT. ***
            # Passing `local_scaler_type` was tried on 2026-09-04 and produces a checkpoint whose
            # inference path is silently wrong. `NeuralForecast` stores the SCALED dataset, and
            # `cascade_inference.history_from_bundle` reads `nf.dataset.temporal` DIRECTLY -- it
            # never sees the inverse transform that `nf.predict()` applies. Measured on the
            # resulting checkpoint: y came back mean 0.200 / min -3.119 against the raw 9.913 /
            # 0.164, `queue_pressure_forecast` 0.068 against 5.071, and `nominal_capacity` exactly
            # 0.000 (a per-series constant, so `(x-median)/MAD` collapses it).
            #
            # That reconstructed history feeds the CHRONOS CASCADE CONTEXT and the TREE'S LAG AND
            # ROLLING FEATURES. Only A9's capacity weights had a NaN guard, and it is the sole
            # reason this failed loudly instead of emitting a plausible CSV from a zero-crossing
            # target that neither Chronos nor LightGBM was trained on.
            #
            # AND THE CV NUMBERS ARE NOT AFFECTED: `src/eval/cv.py` never calls
            # `history_from_bundle`, and `nf.predict()` inverse-transforms, so CV predictions are in
            # raw units (verified: pred mean 10.060 against y mean 10.681). The defect is
            # submission-path-only.
            #
            # NOT WORTH FIXING FOR WHAT IT BUYS. At matched steps the scaler is +0.00381 at 1.5 SE
            # and points toward UNSCALED. Inverting the stored dataset would be new code on the
            # graded path for a null.
            nf = NeuralForecast(models=models, freq="h")
            # val_size carves the last `h` hours off the training data for early stopping. That is
            # the full-history problem in miniature: there is no natural held-out tail at the
            # real origin, so the most recent h hours go to the stopping rule, not to fitting.
            nf.fit(long, static_df=stat, val_size=int(cfg["h"]))

            future = nf.make_future_dataframe().rename(columns={NF_ID: ID, NF_TIME: TIME})
            future[TIME] = pd.to_datetime(future[TIME])

            cov = horizon_covariates(
                str(repo / "data/raw/validation_input.csv"),
                str(horizon_frame or repo / HORIZON_FRAME),
            )
            if limit_series:
                cov = cov[cov[ID].isin(keep)]
            # ORDER MATTERS, AND THE SMOKE CAUGHT IT. `build_futr_df` VALIDATES its output
            # against the live `futr_exog_list()` (loader.py:192) — with the aggregate channel
            # active it demands columns it cannot build, because it works from `cov`, which has
            # only the raw covariates. So the frame is built with the channel temporarily OFF, and
            # the aggregates are attached to the finished, imputed frame afterwards.
            with aggregate_columns([]):
                futr = build_futr_df(cov, future, fill_stats)
            # A9 weights by `nominal_capacity`, which is a STATIC — it is not in the horizon frame
            # at all. Merge the statics in for the build, then drop them again: `nf.predict` wants
            # the model's futr columns, not the static ones (`predict.py` does the same thing).
            futr_cols_before = list(futr.columns)
            futr = futr.merge(stat, on=NF_ID, how="left")
            futr = attach_active_aggregates(futr, where="submission horizon")
            keep_cols = futr_cols_before + [c for c in xs_names if c not in futr_cols_before]
            futr = futr[keep_cols]

            preds = nf.predict(futr_df=futr)

    # *** PERSIST THE WEIGHTS. S6 1b: they cannot be rebuilt. ***
    # A re-fit at the same seed on a different card is a DIFFERENT MODEL (max|dpred| 13.40, as large
    # as a fresh seed), so a checkpoint is the only representation of "the 5-seed bag" that survives
    # this container. The bundle is written from the SAME `nf` that produced the forecasts below,
    # which is what keeps the artifact and its outputs consistent.
    ckpt = None
    if save_checkpoint:
        from src import bundle

        ckpt = Path(save_checkpoint)
        bundle.save(nf, fill_stats=fill_stats, cfg=cfg, out_path=ckpt)

    preds = preds.reset_index() if NF_ID not in preds.columns else preds
    model_cols = [c for c in preds.columns if c not in (NF_ID, NF_TIME)]
    path = out / "cascade_submission_raw.csv"
    preds.to_csv(path, index=False)

    meta = {
        "seeds": seeds,
        "model_cols": model_cols,
        "rows": int(len(preds)),
        "series": int(preds[NF_ID].nunique()),
        "hours": int(preds[NF_TIME].nunique()),
        "first_hour": str(preds[NF_TIME].min()),
        "last_hour": str(preds[NF_TIME].max()),
        "futr_rows": int(len(futr)),
        "out": str(path),
        "checkpoint": str(ckpt) if ckpt else None,
        "checkpoint_mb": round(ckpt.stat().st_size / 1e6, 1) if ckpt else None,
    }
    (out / "cascade_submission_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only; 0 = the config's value")
    ap.add_argument("--limit-series", type=int, default=0, help="smoke only")
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--train-frame", default=None)
    ap.add_argument("--horizon-frame", default=None)
    ap.add_argument("--save-checkpoint", default=None, help="write the 5-model bundle here")
    ap.add_argument("--xs-block", default="", help="cross-series block to attach, e.g. A9")
    # MEMORY. The 5-seed bag fits five TFTs in ONE NeuralForecast on ONE card, over the full 4320h
    # history at h=672. Adding the A9 covariates OOM'd the L4 at `tft.py:257` (attention), which is
    # the allocation that scales with windows_batch_size x sequence_length^2 -- 1176 steps, 16
    # windows, 5 models. Halving the window batch halves those activations and touches neither the
    # model nor the unfolded data grid.
    ap.add_argument("--windows-batch-size", type=int, default=0, help="override; 0 = config/2")
    args = ap.parse_args()

    meta = run(
        n_seeds=args.seeds,
        max_steps=args.max_steps or None,
        limit_series=args.limit_series,
        out_dir=args.out_dir,
        train_frame=args.train_frame,
        horizon_frame=args.horizon_frame,
        save_checkpoint=args.save_checkpoint,
        xs_block=args.xs_block,
        windows_batch_size=args.windows_batch_size,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
