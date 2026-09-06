"""Generate the cascade covariate AT INFERENCE, and degrade honestly when it cannot be.

The ship member is a TFT conditioned on ``chronos2_forecast`` — a zero-shot Chronos-2 forecast of
the target over the horizon, worth **+8.3%** over plain TFT (0.15138 -> 0.13429). At CV time that
column arrived as a derived frame on disk. At submission time there is no such frame, and there
are only two other places it could come from, both closed:

- **the input dir** — the harness supplies ``test_input.csv``, not our ``*_chronos.csv``. Feeding
  our own derived frame would be scoring ourselves on inputs nobody gave us;
- **the checkpoint** — the specification: *"You should not bake any data into your checkpoint, as
  the timeframe might differ."*

So it has to be **generated here**, from the target history the bundle already carries. That is
also the only construction that stays honest by shape rather than by inspection: the context ends
at the last observed hour, so nothing after the origin can reach the forecast.

**Chronos-2 is 456 MB and the archive cap is 200 MB**, so the weights are not in the box. A
published clarification (2026-06-07) explicitly sanctions fetching them at test time, and this
module does — but the same specification says *"do not depend on internet access during final
inference"*, and a sandbox with no egress would otherwise cost the entire run. Hence
``resolve_source`` tries three places and ``attach_channel`` has a **defined degraded mode** rather
than an exception:
fall back to the per-series median the bundle already stores, with ``*_missing = 1``.

That fallback is not invented for the occasion — it is the exact state the model met during
training, where the rolled covariate leaves a 288-hour NaN warm-up prefix that ``apply_fill``
resolves the same way. The VSN has seen this input and knows to distrust the channel there.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import MISSING_SUFFIX
from src.data.loader import NF_ID, NF_TARGET, NF_TIME

CHRONOS_COL = "chronos2_forecast"
MODEL_ID = "amazon/chronos-2"

# PIN THE REVISION. `configs/chronos2.yaml` names `model_id` with no revision, so every run
# resolves whatever `main` points at on the day — and this repo moved once already during the
# project. Measured 2026-08-09: the two cached snapshots (0f8a440, 29ec376) symlink the SAME blob
# for both files, so no recorded number is in question; the pin guards a FUTURE revision silently
# changing the covariate between our CV and the graded run, which would have no symptom at all.
PINNED_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"

#: Where a bundled copy of the weights would live inside an unpacked checkpoint, if we ever ship
#: one. Absent by default — at 456 MB it does not fit the archive — but tried first so that a
#: fully-offline variant needs no code change, only a bigger box.
BUNDLED_WEIGHTS_SUBDIR = "chronos2"


def _log(msg: str) -> None:
    print(f"[cascade] {msg}", flush=True)


def resolve_source(bundle_dir: Path | None = None, *, allow_download: bool = True) -> str | None:
    """Locate Chronos-2 weights: bundled -> local HF cache -> hub. ``None`` if unreachable.

    Returns something ``Chronos2Pipeline.from_pretrained`` accepts. Every step is announced,
    because the difference between the cascade arm and the fallback arm is a ~6% relative swing in
    the graded metric and must never be a silent property of the machine it ran on.
    """
    if bundle_dir is not None:
        local = Path(bundle_dir) / BUNDLED_WEIGHTS_SUBDIR
        if (local / "config.json").exists():
            _log(f"weights: bundled copy at {local}")
            return str(local)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _log("weights: huggingface_hub not installed; cannot resolve Chronos-2")
        return None

    # The cache first, and explicitly offline, so a machine that already holds the weights never
    # depends on the network being up.
    try:
        path = snapshot_download(MODEL_ID, revision=PINNED_REVISION, local_files_only=True)
        _log(f"weights: local HF cache at {path} (rev {PINNED_REVISION[:8]})")
        return path
    except Exception:
        pass

    if not allow_download:
        _log("weights: not cached and downloads disabled")
        return None

    try:
        path = snapshot_download(MODEL_ID, revision=PINNED_REVISION)
        _log(f"weights: downloaded to {path} (rev {PINNED_REVISION[:8]})")
        return path
    except Exception as exc:  # offline sandbox, hub outage, rate limit
        _log(f"weights: download failed ({type(exc).__name__}: {exc})")
        return None


def history_from_bundle(nf) -> pd.DataFrame:
    """Rebuild the observed history as a long frame from the checkpoint's stored dataset.

    ``nf.save(save_dataset=True)`` keeps the full training frame — 96 x 4320 rows and every future
    covariate — which is what makes an offline cascade possible at all: Chronos needs a context and
    the input dir carries no target history whatsoever.

    Storing it is *required*, not a convenience, and is not the thing that ruling forbids: this
    is the model's own conditioning window, not provided validation or test data. A windowed model
    that ships without its lookback cannot forecast anything.
    """
    ds = nf.dataset
    temporal = ds.temporal
    arr = temporal.numpy() if hasattr(temporal, "numpy") else np.asarray(temporal)
    arr = np.array(arr, copy=True)
    cols = [str(c) for c in ds.temporal_cols]

    # *** UNDO THE LOCAL SCALER, OR EVERY CONSUMER OF THIS FRAME IS SILENTLY WRONG. ***
    # `NeuralForecast` with `local_scaler_type` stores the SCALED temporal array and applies the
    # inverse only inside `nf.predict()`. This function reads `ds.temporal` DIRECTLY, so without the
    # loop below it hands back a robust-normalised frame: measured on a scaled checkpoint,
    # y arrived at mean 0.200 / min -3.119 against the raw 9.913 / 0.164, `queue_pressure_forecast`
    # at 0.068 against 5.071, and `nominal_capacity` at exactly 0.000 -- a per-series CONSTANT, so
    # `(x - median) / MAD` collapses it to zero.
    #
    # WHAT THAT BREAKS, and none of it is theoretical: this frame is the CHRONOS CASCADE CONTEXT,
    # the TREE'S LAG AND ROLLING FEATURES, and the source of `nominal_capacity` for A9's capacity
    # weights. Only A9 had a NaN guard (0/0 from an all-zero weight vector), and it is the sole
    # reason a scaled checkpoint failed loudly on 2026-09-04 instead of emitting a plausible CSV
    # built from a zero-crossing target that neither Chronos nor LightGBM ever saw in training.
    #
    # EXACT, NOT APPROXIMATE, AND CHECKED AS SUCH. `coreforecast`'s scalers keep per-series stats
    # and invert to float32 precision: round-tripped against `data/raw/train.csv`, y returns to
    # 9.9128 / 5.5482 / 0.1640 and every covariate matches to <= 4e-6. See
    # `tests/test_history_unscale.py`, which fails on the un-inverted frame.
    #
    # A NO-OP FOR AN UNSCALED CHECKPOINT. `nf.scalers_` is empty when `local_scaler_type` is None,
    # so every checkpoint shipped before this change is byte-identical through here.
    scalers = getattr(nf, "scalers_", None) or {}
    if scalers:
        from coreforecast.grouped_array import GroupedArray

        indptr = np.asarray(ds.indptr).astype(np.int32)
        for j, col in enumerate(cols):
            sc = scalers.get(col)
            if sc is None:
                continue
            arr[:, j] = sc.inverse_transform(GroupedArray(arr[:, j].astype(np.float64), indptr))

    frames = []
    for i, uid in enumerate(nf.uids):
        lo, hi = int(ds.indptr[i]), int(ds.indptr[i + 1])
        block = pd.DataFrame(arr[lo:hi], columns=cols)
        block.insert(0, NF_ID, uid)
        idx = pd.date_range(end=pd.Timestamp(nf.last_dates[i]), periods=hi - lo, freq=nf.freq)
        block.insert(1, NF_TIME, idx)
        frames.append(block)

    hist = pd.concat(frames, ignore_index=True)
    return hist.drop(columns=[c for c in ("available_mask",) if c in hist.columns])


def generate_channel(
    nf,
    futr_df: pd.DataFrame,
    *,
    source: str,
    device: str = "cuda",
    batch_series: int = 0,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast ``chronos2_forecast`` over exactly ``futr_df``'s rows, conditioned on the history.

    Mirrors ``chronos2_oof.generate_inference``: ONE block anchored at the end of the observed
    history, so the context stops at the origin and the frame is gap-honest by construction rather
    than by a check applied afterwards.

    The block length is the full horizon (``gap + span``, i.e. 672 for the graded run), matching
    what the CV frames did for their scored halves — ``chronos2_oof --cut-idx c --reuse`` forecasts
    ``[c, c+672)`` in one call from context ``[0, c)``. S4 measured that keeping the 336h *training*
    grid against this longer inference lead is load-bearing rather than a defect: matching them
    costs 10.8%. So the mismatch here is deliberate and is the shipped condition.

    ``history`` overrides the bundle's stored context. Only the recursive-rollout arm passes it
    (``predict.rollout_predict``), where the context has been extended with the model's OWN earlier
    forecasts, so a later block's channel is conditioned on the same information the TFT will see.
    Defaulting to ``None`` leaves every normal call reading the bundle, unchanged.
    """
    from src.models.chronos2_eval import _pick_pred_column, _predict
    from src.models.chronos2_oof import base_exog

    exog = [c for c in base_exog() if c in futr_df.columns]
    hist = history_from_bundle(nf) if history is None else history
    missing_ctx = [c for c in exog if c not in hist.columns]
    if missing_ctx:
        raise ValueError(f"history lacks covariates Chronos conditions on: {missing_ctx}")

    ctx = hist[[NF_ID, NF_TIME, NF_TARGET, *exog]]
    fut = futr_df[[NF_ID, NF_TIME, *exog]]
    h = int(fut.groupby(NF_ID)[NF_TIME].size().max())
    _log(f"generating {CHRONOS_COL}: h={h}, {fut[NF_ID].nunique()} series, {len(exog)} covariates")

    pipe = _load_pipeline_from(source, device)
    try:
        raw = _predict(pipe, ctx, fut, h, batch_series)
    finally:
        _release_pipeline(pipe, device)
        del pipe
    pcol = _pick_pred_column(raw)
    out = raw[[NF_ID, NF_TIME, pcol]].rename(columns={pcol: CHRONOS_COL})
    out[CHRONOS_COL] = out[CHRONOS_COL].clip(lower=0.0)
    return out


def _release_pipeline(pipe, device: str) -> None:
    """Move a finished pipeline off the GPU and drop its cached blocks.

    S10 needs this and S9 did not: the archive now runs TWO Chronos-2 forward passes per inference
    — the zero-shot cascade channel, then the fine-tuned blend member — and each set of weights is
    455.8 MB. Without an explicit release the first stays resident, and the clean-room run measured
    the consequence exactly: 5.94 GiB in use, a 1.02 GiB allocation refused, and the member falling
    back to rung 2 on an 8 GB card. S9.0b had measured 5688 MiB with ONE pipeline, so the second is
    precisely what tips it over.

    The failure mode this prevents is the quiet one. An OOM here does not crash the submission — it
    degrades it — so a grading box with a modest GPU would have shipped the previous model while
    every log line still looked healthy. Moving the weights to CPU (rather than only ``del``-ing)
    releases the VRAM even if something else still holds a reference to the pipeline.
    """
    import gc

    before = _vram_mb(device)
    moved = []
    # `Chronos2Pipeline.__init__` does `super().__init__(inner_model=model); self.model = model`,
    # so both names point at the same module — but naming both means a future refactor that keeps
    # only one still releases. A silent no-op here is exactly the failure this function exists to
    # prevent, so an attribute that is missing gets LOGGED rather than skipped quietly.
    for attr in ("model", "inner_model"):
        obj = getattr(pipe, attr, None)
        if obj is not None and hasattr(obj, "to"):
            try:
                obj.to("cpu")
                moved.append(attr)
            except Exception as exc:  # never let cleanup break a forecast that already succeeded
                _log(f"release: {attr}.to('cpu') failed ({type(exc).__name__}: {exc})")
    if not moved:
        _log(f"release: NOTHING MOVED — no .model/.inner_model on {type(pipe).__name__}")
    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    after = _vram_mb(device)
    if before is not None:
        _log(f"release: moved={moved or None} VRAM {before:.0f} -> {after:.0f} MiB allocated")


def _vram_mb(device: str) -> float | None:
    """Allocated VRAM in MiB, or None off-GPU. Diagnostics that SHIP: the archive now runs two
    Chronos passes and the difference between rung 1 and rung 2 is whether the second one fits."""
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        return torch.cuda.memory_allocated() / 2**20
    except Exception:
        return None


def _load_pipeline_from(source: str, device: str):
    """``_load_pipeline`` hardcodes the hub id; this takes a resolved path or id instead."""
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(str(source), device_map=device)


def attach_channel(
    futr_df: pd.DataFrame,
    values: pd.DataFrame | None,
    fill_stats: dict,
    *,
    channel: str = CHRONOS_COL,
) -> pd.DataFrame:
    """Add ``channel`` (+ its ``*_missing`` flag) to ``futr_df``. ``values=None`` -> degraded mode.

    Two arms, and the flag is what tells them apart to the model:

    - **values supplied** — the real forecast, ``*_missing = 0``. The shipped condition.
    - **values None** — the per-series median the bundle already carries, ``*_missing = 1``. Costs
      no extra bytes and is a state the model trained on (the 288h warm-up prefix), so the VSN
      down-gates the channel rather than trusting a fabricated number.

    Never silent: the degraded arm warns, because it changes the graded metric.
    """
    out = futr_df.copy()
    flag = channel + MISSING_SUFFIX

    if values is not None:
        merged = out.merge(values[[NF_ID, NF_TIME, channel]], on=[NF_ID, NF_TIME], how="left")
        if len(merged) != len(out):
            raise ValueError(
                f"attaching {channel} changed the row count ({len(out)} -> {len(merged)}); "
                "the generator emitted duplicate (series, timestamp) pairs."
            )
        gaps = int(merged[channel].isna().sum())
        if gaps:
            raise ValueError(
                f"{gaps} of {len(merged)} horizon rows got no {channel}; the generated block does "
                "not cover the requested hours."
            )
        merged[flag] = 0.0
        return merged

    stats = fill_stats.get(channel) or {}
    if not stats:
        raise ValueError(
            f"cannot degrade gracefully: the checkpoint carries no fill statistic for {channel}."
        )
    fallback = float(np.median([v for v in stats.values() if v is not None]))
    out[channel] = out[NF_ID].map(stats).astype(float).fillna(fallback)
    out[flag] = 1.0
    warnings.warn(
        f"{channel} could not be generated; falling back to the stored per-series median with "
        f"{flag}=1. This is the OFFLINE arm and it scores materially worse than the cascade.",
        stacklevel=2,
    )
    _log(f"DEGRADED: {channel} filled from stored medians, {flag}=1")
    return out


def cascade_enabled() -> bool:
    """Escape hatch so the offline arm can be exercised deliberately (tests, rehearsal)."""
    return os.environ.get("DISABLE_CASCADE", "").strip() not in ("1", "true", "TRUE")
