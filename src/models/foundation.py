"""Zero-shot foundation forecasters — one harness, four ~30-line backends.

What this is for (plan S2)
--------------------------
The error-correlation matrix has two clusters, not a gradient: ``{tft, tft_cascade}`` at one end,
``{chronos_ft, lgbm, lstm, bitcn, lgbm_s24_unitcat}`` at the other, minimum 0.916 *within* either
block and maximum 0.813 *across*. Five architectures that share nothing but a training set are, as
far as the errors go, one member — which is why the blend saturates at two. The one untried category
that could sit outside both blocks is a model **pretrained on an outside corpus and never fitted to
this data**.

Those models are worth screening on **two** axes, and the project's own evidence says the second one
is where the value is. The same zero-shot Chronos-2 forecast is worth 0.190 as a standalone member,
+0.0007 (null) as a covariate to the tree, and **0.1463 -> 0.1341** as a covariate to the TFT — the
largest single gain in the project. Value is realised by a model that must traverse the 336h gap and
has a Variable Selection Network that can learn when to trust a prior; the tree never traverses the
gap, which is exactly the row that came back null.

The standing architectural rule (2026-07-31) is what makes this legitimate: **a model FITTED ON
OUR DATA enters only as a blend member, never as a covariate.** The line is *fitted-on-our-data*,
not "any model" — these weights never saw this dataset, which is the same reason ``tft_cascade``
exists.

Why one harness and four small functions
----------------------------------------
All four candidates run the *same procedure*: take ``y`` history up to the cutoff, forecast
``horizon`` steps from there, write the result as a per-window derived frame with a provenance
sidecar. Only the model call differs. So the windowing, the frame write and the sidecar live here
once, and each model contributes a :data:`ForecastFn` of 20-40 lines.

Two consequences that are the point of the design:

1. **The harness is testable locally with a stub backend** (``--dry-run``), so ``pytest`` stays
green
   and the repo venv gains *zero* dependencies. Only the four backends need their real package, and
   those only ever run on Modal, in per-model images — TimesFM, TabPFN and xLSTM pin mutually
   incompatible stacks, and the isolation means a model that will not install costs one image and
   blocks nothing.
2. **One forward pass yields both artifacts.** The 672-step block anchored at the cutoff *is* the
   member prediction (its last 336 steps) and *is* the covariate over the horizon. So the member
   axis is a free by-product of the covariate screen, and nothing has to be decided in advance.

Gap honesty is structural here, not inspected --------------------------------------------- Every
block is generated from a context that stops before it starts, and the horizon block's context stops
at the cutoff — the forecast actually obtainable at test time. A ``.provenance.json`` sidecar
records the grid and ``run_member`` refuses a channel it cannot verify. That check is not
decoration: a covariate whose context ran past the cutoff scores **better**, so an unchecked one
fails in the flattering direction. See ``src.models.cascade_provenance`` for the leak this fences,
and plan 3.10a for what it was worth once measured (0.0018 of a 0.0209 gap — real, and nearly
worthless).

Two modes, and the second one is the Stage-2 cost the plan under-counts
----------------------------------------------------------------------- ``--horizon-only`` (the
default, and all Stage 1 needs) generates the single block ``[cut, cut+horizon)``. That is ~1
GPU-minute per (model, window) and yields the member cube plus the covariate over the scored block.

``--train-region`` additionally rolls the origin back across ``[0, cut)`` in ``block``-hour steps,
so a TFT can *train* on the channel. Stage 2 needs it and Stage 1 does not, which matters for the
budget: the plan costs Stage 2 as "1 TFT train x 3 windows", and the honest figure is that plus one
train-region pass per window for the winning model (the analogue of the original Chronos rolling
pass). Named here so it is a planned cost rather than a surprise.

    python -m src.models.foundation --model toto --cut-idx 3648            # Stage 1
    python -m src.models.foundation --model toto --cut-idx 3648 --train-region   # Stage 2 prereq
    python -m src.models.foundation --model toto --cut-idx 3648 --dry-run  # CPU, no package
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import ID, TARGET, TIME
from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import HOUR_IDX, SCORE_LEN, add_hour_index
from src.models.cascade_provenance import check_gap_honest, write_provenance

# The rolling block size for the train region. 336 matches the graded horizon and the grid
# ``chronos2_oof`` already used, so a foundation channel and the Chronos channel are on the same
# k-ahead grid — which is what makes a two-channel (ALONGSIDE) cascade a comparison of *models*
# rather than of forecast leads.
TRAIN_BLOCK = SCORE_LEN
GAPPED_HORIZON = 2 * SCORE_LEN  # 672 = gap + scored block
WEEK = 168

# Toto 2.0's checkpoint, and how many series it forecasts at once.
#
# The chunk is DELIBERATELY NOT the 12 that was calibrated against Toto 1.0. That number was
# measured on a different package, a different parameter count and a different attention
# implementation, so carrying it over would be a guess wearing a measurement's clothes. 32 is a
# starting point; the full-width smoke is what adjudicates it, which is the entire reason that gate
# now runs 96 series instead of 3. If it OOMs, halve it — the backend is univariate per series, so
# this constant changes throughput and nothing else about the numbers.
TOTO_REPO = "Datadog/Toto-2.0-313m"
TOTO_SERIES_CHUNK = 32
TIREX_REPO = "NX-AI/TiRex"  # v1: v2 caps its horizon at 320 and we need 672

# A backend takes the observed history as a long frame (``unique_id, ds, y``), a horizon, and a
# device; it returns one row per series in the frame's own ``unique_id`` sort order, ``h`` columns
# wide. Target-only, deliberately: Stage 1 compares four pretrained families like with like, and
# TabPFN-TS's covariate variant is the obvious S3 follow-up precisely because it is the one
# candidate whose forecast would *move* with a better gap fill.
ForecastFn = Callable[[pd.DataFrame, int, str], np.ndarray]

# A COVARIATE-AWARE backend takes the context AND the known-future frame, both carrying the exog
# columns, and returns the same (n_series, h) array. Separate signature rather than an optional
# argument so a target-only backend can never be handed a frame it will silently ignore.
#
# THIS EXISTS BECAUSE THE FIRST SCREEN WAS CONFOUNDED. `chronos2_oof` feeds Chronos-2 all 29
# known-future covariates via `base_exog()`, while `run_one` sliced the candidates to
# [unique_id, ds, y] before `generate` saw the rest. The screen then ranked four covariate-BLIND
# models against a covariate-FED control and reported them "far below" it — which measured the
# covariates, not the models. Matching the control is the whole point, so these backends reuse its
# exact recipe: the shared imputation over NAN_COLS, then `futr_exog_list()` minus our own column.
CovForecastFn = Callable[[pd.DataFrame, pd.DataFrame, int, str], np.ndarray]

_BACKENDS: dict[str, ForecastFn] = {}
_COV_BACKENDS: dict[str, CovForecastFn] = {}


def register_backend(name: str, fn: ForecastFn) -> ForecastFn:
    if name in _BACKENDS or name in _COV_BACKENDS:
        raise ValueError(f"foundation backend {name!r} is already registered")
    _BACKENDS[name] = fn
    return fn


def register_cov_backend(name: str, fn: CovForecastFn) -> CovForecastFn:
    """Register a backend that consumes known-future covariates alongside the target."""
    if name in _BACKENDS or name in _COV_BACKENDS:
        raise ValueError(f"foundation backend {name!r} is already registered")
    _COV_BACKENDS[name] = fn
    return fn


def available_backends() -> list[str]:
    return sorted(set(_BACKENDS) | set(_COV_BACKENDS))


def is_covariate_backend(name: str) -> bool:
    return name in _COV_BACKENDS


def foundation_exog(name: str) -> list[str]:
    """Known-future covariates for a covariate-aware backend — the control's list, minus our own.

    Deliberately `futr_exog_list()` rather than a hand-picked subset, because the ONLY reason this
    exists is to put a candidate on the same footing as `chronos2_oof.base_exog()`. A different list
    would re-introduce the confound in a smaller size.
    """
    from src.data.features import MISSING_SUFFIX, futr_exog_list

    col = forecast_column(name)
    drop = {col, col + MISSING_SUFFIX}
    # `_cov` variants generate the SAME channel as their target-only twin (toto_cov -> toto_cov_
    # forecast), but a two-channel ALONGSIDE frame may carry the twin, so strip both spellings.
    if name.endswith("_cov"):
        base = forecast_column(name[: -len("_cov")])
        drop |= {base, base + MISSING_SUFFIX}
    return [c for c in futr_exog_list() if c not in drop]


def forecast_column(name: str) -> str:
    """The channel name a model contributes: ``toto`` -> ``toto_forecast``."""
    return f"{name}_forecast"


def derived_frame(name: str, cut_idx: int, out_dir: str | Path = "data/derived") -> Path:
    """Per-window derived frame path. Per window because the covariate is honest for ONE cutoff."""
    return Path(out_dir) / f"train_{name}_cut{cut_idx}.csv"


# --------------------------------------------------------------------------- the shared harness


def _series_order(long_df: pd.DataFrame) -> list[str]:
    """The canonical series order a backend's output rows are matched against."""
    return sorted(long_df[NF_ID].unique())


def _seasonal_naive(history: pd.DataFrame, h: int, device: str) -> np.ndarray:
    """Stub backend: per-series weekly-naive (last 168h tiled). NOT a result — plumbing only.

    Exists so the harness, the frame write, the sidecar and the member runner are all exercised by
    ``pytest`` with no package installed and no GPU. Mirrors ``chronos2_oof``'s dry-run backend, so
    a plumbing bug surfaces identically on both covariate paths.
    """
    del device
    out = []
    for sid in _series_order(history):
        hist = history.loc[history[NF_ID] == sid, NF_TARGET].to_numpy(dtype=float)
        period = hist[-WEEK:] if hist.size >= WEEK else hist
        out.append(np.resize(period, h) if period.size else np.zeros(h))
    return np.vstack(out)


def _block(
    long_df: pd.DataFrame,
    forecast_fn: ForecastFn | CovForecastFn,
    start: int,
    h: int,
    device: str,
    label: str,
    exog: list[str] | None = None,
) -> pd.DataFrame:
    """Forecast ``[start, start+h)`` from context ``[0, start)``. The only place a backend is
    called.

    The two assertions are the leakage guarantee in code rather than in a docstring — which is the
    lesson 3.10 charged us for. ``chronos2_oof``'s docstring claimed "all consumers inherit the
    honesty for free" while its grid handed the scored block a context ending 336 hours past the
    cutoff.
    """
    ctx = long_df[long_df[HOUR_IDX] < start]
    fut = long_df[(long_df[HOUR_IDX] >= start) & (long_df[HOUR_IDX] < start + h)]
    assert int(ctx[HOUR_IDX].max()) == start - 1, "context overruns the origin"
    assert fut.empty or int(fut[HOUR_IDX].min()) >= start, "forecast hour precedes the origin"

    order = _series_order(long_df)
    t0 = time.perf_counter()
    if exog is None:
        preds = np.asarray(forecast_fn(ctx[[NF_ID, NF_TIME, NF_TARGET]], h, device), dtype=float)
    else:
        # Same split, same assertions — the covariates ride along on frames already proven not to
        # overrun the origin. `fut` carries NO target: these are known-FUTURE covariates, and
        # handing a backend `y` over the horizon is the leak this whole module exists to prevent.
        missing = [c for c in exog if c not in long_df.columns]
        if missing:
            raise KeyError(f"covariate backend needs {missing}, absent from the frame")
        # A short future frame is the one failure a covariate backend cannot survive: it needs a
        # covariate row per forecast hour. The target-only path tolerates it (predictions running
        # the end are dropped by the hour-index merge), so this has to be checked here, not there.
        per_series = fut.groupby(NF_ID)[HOUR_IDX].count()
        if per_series.empty or int(per_series.min()) < h:
            got = 0 if per_series.empty else int(per_series.min())
            raise ValueError(
                f"covariate backend asked for {h} future hours from {start} but the frame supplies "
                f"only {got} — the timeline ends at {int(long_df[HOUR_IDX].max())}."
            )
        preds = np.asarray(
            forecast_fn(
                ctx[[NF_ID, NF_TIME, NF_TARGET, *exog]], fut[[NF_ID, NF_TIME, *exog]], h, device
            ),
            dtype=float,
        )
    if preds.shape != (len(order), h):
        raise ValueError(
            f"backend returned {preds.shape}, expected {(len(order), h)} — one row per series in "
            f"sorted unique_id order, {h} columns wide."
        )
    print(
        f"  [{label}] hours [{start}..{start + h - 1}] "
        f"({time.perf_counter() - t0:.1f}s{_gpu_note()})"
    )

    # Map back onto real timestamps by hour index, so a series shorter than the block (or a horizon
    # running off the end of the frame) drops rows rather than silently shifting them.
    hours = long_df[[NF_ID, NF_TIME, HOUR_IDX]]
    wide = pd.DataFrame(preds, index=order, columns=range(start, start + h))
    tidy = wide.stack().rename("value").reset_index()
    tidy.columns = [NF_ID, HOUR_IDX, "value"]
    return tidy.merge(hours, on=[NF_ID, HOUR_IDX], how="inner")[[NF_ID, NF_TIME, "value"]]


def generate(
    name: str,
    long_df: pd.DataFrame,
    cut_idx: int,
    *,
    forecast_fn: ForecastFn | None = None,
    horizon: int = GAPPED_HORIZON,
    device: str = "cuda",
    train_region: bool = False,
    block: int = TRAIN_BLOCK,
) -> tuple[pd.DataFrame, list[tuple[int, int]], int]:
    """All of one model's forecasts for one cutoff, plus the provenance grid they were made on.

    Returns ``(pred_long, blocks, n_hours)`` where ``pred_long`` is
    ``[unique_id, ds, <name>_forecast]`` and ``blocks`` is ``[(start, length), ...]`` in the shape
    ``src.models.cascade_provenance`` records.

    The horizon block is generated **last and separately** from the train region, and anchored at
    the cutoff rather than spliced from a rolling grid. That is the whole correction of plan 3.10: a
    336h rolling grid covers ``[c, c+336)`` honestly and ``[c+336, c+672)`` from a context 336 hours
    past the cutoff, which is the half that is graded.
    """
    fn = forecast_fn or _BACKENDS.get(name) or _COV_BACKENDS.get(name)
    if fn is None:
        raise KeyError(f"no backend for {name!r}; registered: {available_backends()}")
    # An explicit `forecast_fn` (the dry-run stub, the tests) is target-only unless the registered
    # name says otherwise — the stub takes three arguments and must not be handed four.
    exog = foundation_exog(name) if (forecast_fn is None and is_covariate_backend(name)) else None

    df = add_hour_index(long_df)
    n_hours = int(df.groupby(NF_ID)[HOUR_IDX].max().min()) + 1
    if not 0 < cut_idx <= n_hours:
        raise ValueError(f"cut_idx={cut_idx} outside the timeline (n={n_hours})")

    frames, blocks = [], []
    if train_region:
        # Aligned to the cutoff and rolled backwards, so every train block sits on the same k-ahead
        # grid as the horizon block and the train-fill / inference-fill distributions match.
        starts = list(range(cut_idx - block, 0, -block))
        for i, s in enumerate(starts, 1):
            frames.append(_block(df, fn, s, block, device, f"train {i}/{len(starts)}", exog))
            blocks.append((s, block))
        warmup = cut_idx - len(starts) * block
        print(f"  warm-up (no prior origin): first {warmup} hours/series stay NaN")

    frames.append(_block(df, fn, cut_idx, horizon, device, "horizon", exog))
    blocks.append((cut_idx, horizon))

    pred = pd.concat(frames, ignore_index=True).rename(columns={"value": forecast_column(name)})
    pred[forecast_column(name)] = pred[forecast_column(name)].clip(lower=0.0)
    return pred, blocks, n_hours


def attach(
    raw_csv: str | Path, pred_long: pd.DataFrame, column: str, out_csv: Path
) -> pd.DataFrame:
    """Left-merge the forecast column onto the raw CSV — original schema plus exactly one column.

    One column and a sidecar, rather than a richer format, because every downstream loader
    (``load_window_long``, ``src.data.impute``, the tree's ``build_design``) reads the raw schema
    and must stay unchanged. The uncovered warm-up prefix stays NaN and the shared imputation fills
    it with a per-series median plus a ``*_missing`` flag, exactly as for the planning signals.
    """
    raw = pd.read_csv(raw_csv)
    raw[TIME] = pd.to_datetime(raw[TIME])
    keep = set(pred_long[NF_ID].unique())
    raw = raw[raw[ID].isin(keep)]
    p = pred_long.rename(columns={NF_ID: ID, NF_TIME: TIME})
    merged = raw.merge(p[[ID, TIME, column]], on=[ID, TIME], how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def _sanity(merged: pd.DataFrame, column: str, label: str) -> None:
    """The leakage tell-tales: NaN fraction, and corr(forecast, target) on observed rows.

    ``corr ~= 1.0`` means the "forecast" is reading its own answer. It is a smell test, not a proof
    — 3.10's leak sat at corr 0.99 and was still only worth 0.0018 — so it is printed, never gated
    on. The gate is the provenance grid, which is a fact about how the numbers were made.
    """
    col = merged[column]
    msg = f"[{label}] {column}: {float(col.isna().mean()):.1%} NaN (warm-up)"
    if TARGET in merged.columns:
        ok = merged.dropna(subset=[column, TARGET])
        if not ok.empty:
            msg += (
                f"; corr(forecast, target)={float(np.corrcoef(ok[column], ok[TARGET])[0, 1]):.3f}"
            )
    print(msg)


def run_one(
    name: str,
    train_csv: str | Path,
    cut_idx: int,
    *,
    out_dir: str | Path = "data/derived",
    horizon: int = GAPPED_HORIZON,
    device: str = "cuda",
    train_region: bool = False,
    forecast_fn: ForecastFn | None = None,
    limit_series: int = 0,
) -> Path:
    """Generate, attach, write the sidecar, and VERIFY — the whole per-(model, window) unit of work.

    The verification is the last step deliberately: ``check_gap_honest`` re-reads what was just
    written rather than trusting the in-memory grid, so a frame that reaches the volume has been
    checked in the same way ``run_member`` will check it before spending a GPU on it.
    """
    column = forecast_column(name)
    raw = pd.read_csv(train_csv)
    raw[TIME] = pd.to_datetime(raw[TIME])

    covariate_run = forecast_fn is None and is_covariate_backend(name)
    if covariate_run:
        # The control's recipe, step for step (`chronos2_oof.load_long`): fit the shared fill and
        # apply it over NAN_COLS, so ~4.5% NaN in the planning signals becomes a per-series median
        # plus a `*_missing` flag. A DIFFERENT imputation here would reintroduce the confound this
        # variant exists to remove — the candidate would be scored against a control that saw the
        # same covariates filled another way. Note this runs BEFORE the rename, because
        # `fit_fill_stats` groups by the raw `ID` column.
        from src.data.features import NAN_COLS
        from src.data.impute import apply_fill, fit_fill_stats

        nan_cols = [c for c in NAN_COLS if c != forecast_column(name)]
        raw = apply_fill(raw, fit_fill_stats(raw, nan_cols=nan_cols), nan_cols=nan_cols)

    long_df = raw.rename(columns={ID: NF_ID, TIME: NF_TIME, TARGET: NF_TARGET})
    if limit_series > 0:
        keep = sorted(long_df[NF_ID].unique())[:limit_series]
        long_df = long_df[long_df[NF_ID].isin(keep)]

    frame = long_df if covariate_run else long_df[[NF_ID, NF_TIME, NF_TARGET]]

    pred, blocks, n_hours = generate(
        name,
        frame,
        cut_idx,
        forecast_fn=forecast_fn,
        horizon=horizon,
        device=device,
        train_region=train_region,
    )

    out_csv = derived_frame(name, cut_idx, out_dir)
    merged = attach(train_csv, pred, column, out_csv)
    write_provenance(
        out_csv,
        column=column,
        blocks=blocks,
        n_hours=n_hours,
        generator=f"{__name__}:{name}",
        zero_shot=True,
        note=(
            f"Gap-honest for cut_idx={cut_idx} ONLY — the horizon block [{cut_idx}, "
            f"{cut_idx + horizon}) is one call from context [0, {cut_idx}). Do not use for another "
            f"cutoff. train_region={train_region}."
        ),
    )
    check_gap_honest(out_csv, column, cut_idx, horizon)
    _sanity(merged, column, f"{name} cut{cut_idx}")
    print(f"Wrote {out_csv} + sidecar (verified gap-honest)")
    return out_csv


def merge_frames(
    frames: dict[str, str | Path], out_csv: str | Path, cut_idx: int, horizon: int = GAPPED_HORIZON
) -> Path:
    """Build the ALONGSIDE (two-channel) frame: one CSV carrying both forecast columns.

    ``frames`` maps ``column -> the single-channel derived frame that holds it``. The first entry
    supplies the base schema; every other contributes exactly its own forecast column.

    Each source frame is re-verified against ``cut_idx`` **before** it is merged, so a channel that
    is honest alone cannot be laundered into a merged frame that nothing rechecks. The merged
    sidecar keeps both grids (:func:`src.models.cascade_provenance.merge_provenance`), which is
    what lets ``run_member`` check each channel separately — a two-channel member is two
    independent honesty claims, not one.
    """
    from src.models import cascade_provenance as cp

    items = list(frames.items())
    base_col, base_path = items[0]
    cp.check_gap_honest(base_path, base_col, cut_idx, horizon)
    merged = pd.read_csv(base_path)
    merged[TIME] = pd.to_datetime(merged[TIME])
    parts = {base_col: cp.read_provenance(base_path, base_col)}

    for col, path in items[1:]:
        cp.check_gap_honest(path, col, cut_idx, horizon)
        other = pd.read_csv(path, usecols=[ID, TIME, col])
        other[TIME] = pd.to_datetime(other[TIME])
        before = len(merged)
        merged = merged.merge(other, on=[ID, TIME], how="left", validate="one_to_one")
        assert len(merged) == before, f"merging {col} changed the row count"
        parts[col] = cp.read_provenance(path, col)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    cp.merge_provenance(
        out_csv,
        parts,
        note=f"Two-channel ALONGSIDE frame for cut_idx={cut_idx}: {sorted(parts)}.",
    )
    for col in parts:
        cp.check_gap_honest(out_csv, col, cut_idx, horizon)
    print(f"Wrote {out_csv} + merged sidecar ({', '.join(sorted(parts))}), verified gap-honest")
    return out_csv


# --------------------------------------------------------------------------- the four backends
#
# Each is the ONLY model-specific code in this module, and each runs only inside its own remote
# image. They are written against each library's documented interface and are validated by a
# per-model smoke gate on the remote runner, not by anything that runs locally — a model whose
# image will not build or whose smoke fails is recorded with its failure and dropped, and it does
# not hold up the other three.


def toto_context_offset(n_hours: int, patch: int) -> int:
    """How many of the OLDEST hours to drop so a Toto context divides into whole patches.

    Module level and dependency-free on purpose: the bug this prevents is pure arithmetic, so the
    test for it must run in the repo venv where `toto2` is not installed. Everything else in
    `toto_forecast` needs an L4 and a 1.25 GB checkpoint to exercise; this needs neither, and it is
    the part that was actually wrong.
    """
    if patch <= 0:
        raise ValueError(f"patch_size must be positive, got {patch}")
    offset = n_hours % patch
    if offset >= n_hours:
        raise ValueError(f"context of {n_hours}h is shorter than one {patch}h patch")
    return offset


_MODEL_SLOT: dict[str, object] = {}


def cached_model(key: str, factory):
    """Keep ONE model resident, keyed by ``key``; loading a different one frees the previous first.

    Every backend used to build its model inside the per-block call. With a horizon-only frame that
    is one load per (model, window) and invisible. Under ``--train-region`` it is ~11, and the
    resident models accumulate: the 2026-08-04 timesfm_cov run reached ``[train 8/8]`` and then died
    on the horizon block with 5.20 GiB of live torch tensors — roughly seven 200M models — inside a
    process already holding 22.02 GiB. **Loading a model in the hot loop is the bug; the OOM is the
    symptom.**

    One slot rather than an unbounded cache, because the key that varies is the compiled horizon
    (``h``): train blocks all use 336 and the horizon block uses 672 exactly once, and it runs last.
    So a single slot gives two loads per window instead of eleven — a memory fix and a speed fix in
    the same line — while an LRU of two would keep a model resident that is never asked for again.
    """
    if key in _MODEL_SLOT:
        return _MODEL_SLOT[key]
    _MODEL_SLOT.clear()
    _release_gpu()
    model = factory()
    _MODEL_SLOT[key] = model
    return model


def _gpu_note() -> str:
    """Per-block GPU memory, so a hogging allocator shows up in the log BEFORE it OOMs.

    ``free`` is the whole device as the driver sees it; ``torch`` is only what torch has allocated.
    The gap between them is everything else in the process, and on the timesfm_cov image that gap is
    JAX — 22.02 GiB held against 5.20 GiB of torch tensors on the run this was added for. A single
    number would have hidden exactly the thing worth seeing, so both are printed.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info()
        gb = 1024**3
        return (
            f", gpu free {free / gb:.1f}/{total / gb:.1f} GiB, "
            f"torch {torch.cuda.memory_allocated() / gb:.1f} GiB"
        )
    except Exception:  # noqa: BLE001 — diagnostics must never break a generation run
        return ""


def _release_gpu() -> None:
    """Drop what the evicted model held. Import-guarded: not every image ships torch."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def toto_forecast(history: pd.DataFrame, h: int, device: str) -> np.ndarray:
    """Datadog Toto 2.0 (313M) — pretrained on **observability telemetry**.

    The closest public analogue to an "operational load index" that exists, which is the single
    strongest a-priori reason to expect a prior Chronos-2 does not already carry: a corpus of
    machine-generated load/latency/throughput series rather than of economics and web traffic.

    Toto 2.0 is a different PACKAGE from the 1.0 this first ran against — `toto2.Toto2Model`, not
    `toto.model.Toto` — with disjoint config schemas, so the old class cannot load the new weights
    and this is a rewrite rather than a new repo string. 313m of the 4m/22m/313m/1B/2.5B family:
    S9 ships weights, and 2.5B is 9.8 GB fp32 against Chronos-2's 456 MB.
    """
    import torch
    from toto2 import Toto2Model

    model = cached_model(f"toto:{device}", lambda: Toto2Model.from_pretrained(TOTO_REPO).to(device))
    model.eval()
    # Measured on the installed package, not read off a doc page: `forecast` returns
    # (n_quantiles, batch, n_var, horizon) with `knots` = [0.1 .. 0.9], so the median is a lookup
    # rather than an assumption about ordering.
    knots = list(model.output_head.knots)
    median_idx = knots.index(0.5)

    # Toto patchifies the context, so its length must be an exact multiple of `patch_size` --
    # otherwise `forecast` dies inside einops with "can't divide axis of length 3312 in chunks of
    # 32". This cost a fan-out window: our cutoffs are [3648, 3312, 2976], and 3648 and 2976 both
    # happen to divide by 32 while 3312 does not, so the ONE cutoff the smoke runs is one of the two
    # that work. A gate that exercises a single point of a parameter the fan-out varies cannot see
    # this class of bug -- the same lesson `--limit-series 3` taught about width.
    #
    # Trim from the LEFT, i.e. drop the OLDEST hours. That is the only side that is safe: the
    # right-hand end IS the cutoff, and moving it is exactly the gap-honesty violation the sidecar
    # exists to catch. Dropping <=31 hours off the head of a >=2976-hour context is immaterial, and
    # it is a no-op at the two cutoffs that already divide -- so frames generated before this fix
    # remain bit-identical rather than needing regeneration.
    patch = int(model.config.patch_size)

    order = sorted(history[NF_ID].unique())
    out_blocks = []
    # Each series is its OWN univariate item (n_var=1), batched only for throughput. Toto is
    # multivariate and could take the 96 units as variates of one panel — but then a unit's forecast
    # would depend on which other units share its slice, making an operational memory constant
    # silently change the numbers. Univariate keeps the chunk a pure memory knob, and it is also
    # what compares like-with-like against the other three, which are all univariate.
    for i in range(0, len(order), TOTO_SERIES_CHUNK):
        part = order[i : i + TOTO_SERIES_CHUNK]
        ctx = np.vstack(
            [history.loc[history[NF_ID] == s, NF_TARGET].to_numpy(dtype=float) for s in part]
        )
        ctx = ctx[:, toto_context_offset(ctx.shape[1], patch) :]
        target = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(1)
        inputs = {
            "target": target,
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(target.shape[0], 1, dtype=torch.long, device=device),
        }
        with torch.inference_mode():
            quantiles = model.forecast(inputs, horizon=h, has_missing_values=False)
        out_blocks.append(np.asarray(quantiles[median_idx][:, 0, :].float().cpu()))
        del inputs, target, quantiles
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    # `order` is sorted and the slices are consecutive, so stacking restores the row order the
    # harness's shape check matches against.
    return np.vstack(out_blocks)


def timesfm_forecast(history: pd.DataFrame, h: int, device: str) -> np.ndarray:
    """Google TimesFM (~200M) — pretrained on Google Trends, Wikipedia pageviews and synthetics.

    On the list precisely because that corpus has *nothing* in common with this one. If a prior
    from an unrelated domain still helps, the cascade's gain is about the shape of the mechanism
    (a global prior a VSN can gate) rather than about corpus match — which is a claim the write-up
    can make only if this candidate is in the comparison.
    """
    del device  # selected at construction, not per call
    import timesfm

    order = sorted(history[NF_ID].unique())
    ctx = [history.loc[history[NF_ID] == s, NF_TARGET].to_numpy(dtype=float) for s in order]

    # The installed package is TimesFM **2.5**, whose API is not 2.0's. The first version of this
    # function used `timesfm.TimesFm(hparams=..., checkpoint=...)` and died with
    # `module 'timesfm' has no attribute 'TimesFm'`. The shape below was read off the installed
    # module by a CPU probe on the remote runner, not off documentation.
    #
    # `DEFAULT_REPO_ID` is the class's own — so the checkpoint is never named here. That matters:
    # the repo this file previously named (`google/timesfm-2.0-500m-pytorch`) is a 2.0 checkpoint
    # that 2.5 does not load, and prefetching it cached 3.99 GB nothing reads.
    cls = timesfm.TimesFM_2p5_200M_torch

    def _load():
        m = cls.from_pretrained(cls.DEFAULT_REPO_ID, torch_compile=False)
        m.compile(
            timesfm.ForecastConfig(
                max_context=2048,
                max_horizon=h,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
            )
        )
        return m

    # Keyed on `h` because `max_horizon` is compiled in: train blocks share 336, the horizon block
    # asks for 672 once and last. See `cached_model` for what loading this per block cost.
    model = cached_model(f"timesfm:{h}", _load)
    point, _ = model.forecast(horizon=h, inputs=ctx)
    return np.asarray(point)[:, :h]


def tabpfn_ts_forecast(history: pd.DataFrame, h: int, device: str) -> np.ndarray:
    """TabPFN-TS — TabPFN **v3** as a tabular regressor over calendar features.

    **Not a sequence model at all**, which is why it is on the list: it is the most structurally
    different candidate available, and structural difference is what a third cluster would require.
    It is also the only one that could eventually eat the 13 planning signals directly, making its
    covariate variant the obvious S3 follow-up once the gap fill improves.

    The checkpoint is still never named here, and that pays off twice over: ``tabpfn`` 8.x made
    **TabPFN-3** the default, and ``tabpfn-time-series`` 1.2 then overrides it with a
    time-series-specialised v3 checkpoint. Letting the library resolve its own weights put us on
    both without a line changing — where a named repo id would have pinned us to v2.
    """
    del device
    from autogluon.timeseries import TimeSeriesDataFrame
    from tabpfn_time_series import TabPFNMode, TabPFNTimeSeriesPredictor

    # `DefaultFeatures` was a PRE-1.0 name and is exported by no 1.x release — this raised
    # ImportError the moment the licence stopped masking it. The generators are classes now, so
    # they are instantiated rather than passed as unbound functions.
    from tabpfn_time_series.features import (
        CalendarFeature,
        FeatureTransformer,
        RunningIndexFeature,
    )

    order = sorted(history[NF_ID].unique())
    hist = history.rename(columns={NF_ID: "item_id", NF_TIME: "timestamp", NF_TARGET: "target"})
    tsdf = TimeSeriesDataFrame.from_data_frame(
        hist, id_column="item_id", timestamp_column="timestamp"
    )
    train_tsdf, test_tsdf = (
        tsdf,
        TimeSeriesDataFrame(
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "item_id": s,
                            "timestamp": pd.date_range(
                                hist.loc[hist["item_id"] == s, "timestamp"].max()
                                + pd.Timedelta(hours=1),
                                periods=h,
                                freq="h",
                            ),
                            "target": np.nan,
                        }
                    )
                    for s in order
                ],
                ignore_index=True,
            ).set_index(["item_id", "timestamp"])
        ),
    )
    transformer = FeatureTransformer([RunningIndexFeature(), CalendarFeature()])
    train_tsdf, test_tsdf = transformer.transform(train_tsdf, test_tsdf)
    pred = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL).predict(train_tsdf, test_tsdf)
    col = "target" if "target" in pred.columns else pred.columns[0]
    return np.vstack([pred.loc[s, col].to_numpy(dtype=float)[:h] for s in order])


def tirex_forecast(history: pd.DataFrame, h: int, device: str) -> np.ndarray:
    """NX-AI TiRex (35M, xLSTM) — recurrent rather than attention, and by far the smallest.

    **The one candidate deliberately NOT upgraded**, because TiRex-2 cannot reach our horizon:
    it caps at 320 steps and *truncates with a warning* rather than raising, and we need 672
    (336h gap + 336h scored block). Only this module's own shape check turned that into an error
    instead of a silently short frame. Rolling 320x3 would reach 672 and stay gap-honest, and is
    still wrong here — the S2 gate is err-corr <= 0.85, and rollout noise is uncorrelated with
    everything else, so a rolled-out v2 would read as orthogonal *because* it is noisy.

    The repo is **no longer gated** (the HF API reports `gated: false`), so the token is now a
    rate-limit convenience rather than the access requirement this docstring used to assert. Its
    sLSTM CUDA kernels still have a slow Python fallback, so a build that "works" can be too slow.

    Its size is a live consideration rather than trivia: S9 ships weights, Chronos-2 is already
    456 MB, and a two-channel cascade ships both. The specification sets no numeric cap, so size
    is a tiebreaker on a comparable gain, not a gate.
    """
    del device
    import torch
    from tirex import load_model

    order = sorted(history[NF_ID].unique())
    ctx = torch.tensor(
        np.vstack(
            [history.loc[history[NF_ID] == s, NF_TARGET].to_numpy(dtype=float) for s in order]
        ),
        dtype=torch.float32,
    )
    model = cached_model("tirex", lambda: load_model(TIREX_REPO))
    _, mean = model.forecast(context=ctx, prediction_length=h)
    return np.asarray(mean.float().cpu())[:, :h]


# --------------------------------------------------------- covariate-aware variants (the re-screen)
#
# Registered as SEPARATE models (`<name>_cov`) rather than replacing their twins, for three reasons.
# The target-only frames are already generated and gap-honest, so nothing is invalidated. Keeping
# both makes the covariate lift a direct paired A/B on identical rows, a result the write-up
# wants either way. And `tirex` has no covariate path at all (v1 exposes only
# `forecast(context, ...)`; v2 added `future_covariates` but truncates our 672-step horizon to 320),
# so a replace-in-place design would have needed a per-model exception anyway.


def toto_cov_forecast(
    context: pd.DataFrame, future: pd.DataFrame, h: int, device: str
) -> np.ndarray:
    """Toto 2.0 with known-future covariates — through `Toto2Model.forecast`, not the GluonTS path.

    THE VERSION AUDIT WAS WRONG ABOUT THIS. It recorded "Toto 2.0 has no exogenous-variable support
    yet" off the upstream README, and the installed package has it. Reading
    `Toto2GluonTSModel.forward` is what settled it, and the shape of that method is why this backend
    stays 6 lines away from its target-only twin rather than importing GluonTS: `forward` is a thin
    wrapper that puts three extra keys into the SAME `inputs` dict our twin already builds, then
    calls the SAME `model.forecast`. The GluonTS predictor is a batching convenience on top, not the
    covariate mechanism.

        inputs |= {"known_dynamic": ..., "known_dynamic_mask": ..., "known_dynamic_series_ids": ...}

    `known_dynamic` spans **ctx+horizon** — known-future means the model sees the covariate over the
    block it is forecasting, which is the whole point. Covariates ride in that key rather than being
    concatenated onto `target`, so `n_var` stays 1 and the output axes are unchanged.
    """
    import torch
    from toto2 import Toto2Model

    model = cached_model(f"toto:{device}", lambda: Toto2Model.from_pretrained(TOTO_REPO).to(device))
    model.eval()
    knots = list(model.output_head.knots)
    median_idx = knots.index(0.5)
    patch = int(model.config.patch_size)

    exog = [c for c in context.columns if c not in (NF_ID, NF_TIME, NF_TARGET)]
    order = sorted(context[NF_ID].unique())
    out_blocks = []
    for i in range(0, len(order), TOTO_SERIES_CHUNK):
        part = order[i : i + TOTO_SERIES_CHUNK]
        ctx = np.vstack(
            [context.loc[context[NF_ID] == s, NF_TARGET].to_numpy(dtype=float) for s in part]
        )
        off = toto_context_offset(ctx.shape[1], patch)
        ctx = ctx[:, off:]
        target = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(1)

        # (n_series, n_cov, ctx+horizon) — the context half trimmed to match the target exactly, or
        # the two would describe different hours. `h` is a multiple of the patch and so is the
        # trimmed context, so their sum is too.
        kd = np.stack(
            [
                np.concatenate(
                    [
                        context.loc[context[NF_ID] == s, exog].to_numpy(dtype=float)[off:].T,
                        future.loc[future[NF_ID] == s, exog].to_numpy(dtype=float).T,
                    ],
                    axis=1,
                )
                for s in part
            ]
        )
        known = torch.tensor(kd, dtype=torch.float32, device=device)
        inputs = {
            "target": target,
            "target_mask": torch.ones_like(target, dtype=torch.bool),
            "series_ids": torch.zeros(target.shape[0], 1, dtype=torch.long, device=device),
            "known_dynamic": known,
            "known_dynamic_mask": torch.ones_like(known, dtype=torch.bool),
            "known_dynamic_series_ids": torch.zeros(
                known.shape[0], known.shape[1], dtype=torch.long, device=device
            ),
        }
        with torch.inference_mode():
            quantiles = model.forecast(inputs, horizon=h, has_missing_values=False)
        out_blocks.append(np.asarray(quantiles[median_idx][:, 0, :].float().cpu()))
        del inputs, target, known, quantiles
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return np.vstack(out_blocks)


def tabpfn_ts_cov_forecast(
    context: pd.DataFrame, future: pd.DataFrame, h: int, device: str
) -> np.ndarray:
    """TabPFN-TS over calendar features **plus the 29 known-future covariates**.

    The natural fit of the four: TabPFN is a tabular regressor, so a covariate is just another
    column and no architectural channel has to exist for it. The plan called this "the obvious S3
    follow-up"; it arrives early because the screen's control was never target-only.
    """
    del device
    from autogluon.timeseries import TimeSeriesDataFrame
    from tabpfn_time_series import TabPFNMode, TabPFNTimeSeriesPredictor
    from tabpfn_time_series.features import (
        CalendarFeature,
        FeatureTransformer,
        RunningIndexFeature,
    )

    order = sorted(context[NF_ID].unique())
    exog = [c for c in context.columns if c not in (NF_ID, NF_TIME, NF_TARGET)]
    ren = {NF_ID: "item_id", NF_TIME: "timestamp", NF_TARGET: "target"}

    train_tsdf = TimeSeriesDataFrame.from_data_frame(
        context.rename(columns=ren), id_column="item_id", timestamp_column="timestamp"
    )
    # The future frame carries the covariates and a NaN target — the same shape the target-only
    # variant builds, except the columns are real instead of absent. `target` must be present and
    # NaN: it is what marks these rows as the ones to predict.
    fut = future.rename(columns=ren).copy()
    fut["target"] = np.nan
    test_tsdf = TimeSeriesDataFrame.from_data_frame(
        fut[["item_id", "timestamp", "target", *exog]],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    transformer = FeatureTransformer([RunningIndexFeature(), CalendarFeature()])
    train_tsdf, test_tsdf = transformer.transform(train_tsdf, test_tsdf)
    pred = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL).predict(train_tsdf, test_tsdf)
    col = "target" if "target" in pred.columns else pred.columns[0]
    return np.vstack([pred.loc[s, col].to_numpy(dtype=float)[:h] for s in order])


def timesfm_cov_forecast(
    context: pd.DataFrame, future: pd.DataFrame, h: int, device: str
) -> np.ndarray:
    """TimesFM with `forecast_with_covariates` — the target-only forecast plus an xreg correction.

    Worth stating plainly because it changes what a win would MEAN: `xreg_mode="xreg + timesfm"`
    not attend over the covariates inside the transformer. It fits a ridge regression on them and
    adds it to the base forecast, so this measures "TimesFM plus a linear covariate model", not "a
    covariate-native TimesFM". A gain here is therefore a weaker claim than the same gain from
    TabPFN, and the write-up should say so rather than pooling the two.
    """
    del device
    import timesfm

    order = sorted(context[NF_ID].unique())
    exog = [c for c in context.columns if c not in (NF_ID, NF_TIME, NF_TARGET)]
    ctx = [context.loc[context[NF_ID] == s, NF_TARGET].to_numpy(dtype=float) for s in order]

    # Dynamic covariates span context + horizon: TimesFM wants the whole window, not just the tail.
    # Concatenating here (rather than passing two frames) is what the signature asks for.
    dyn = {
        c: [
            np.concatenate(
                [
                    context.loc[context[NF_ID] == s, c].to_numpy(dtype=float),
                    future.loc[future[NF_ID] == s, c].to_numpy(dtype=float),
                ]
            )
            for s in order
        ]
        for c in exog
    }

    cls = timesfm.TimesFM_2p5_200M_torch

    def _load():
        m = cls.from_pretrained(cls.DEFAULT_REPO_ID, torch_compile=False)
        m.compile(
            timesfm.ForecastConfig(
                max_context=2048,
                max_horizon=h,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                # REQUIRED for the covariate path and for that path only — the target-only twin
                # must not set it. `forecast_with_covariates` fits its ridge on the in-sample
                # residual, so it needs the backcast over the context, and without this it raises
                # "For XReg, `return_backcast` must be set to True ... Please recompile the model."
                # It is a compile-time flag, so it cannot be passed at the call and has to be here.
                return_backcast=True,
            )
        )
        return m

    # Separate slot from the target-only twin: same class, different compile (return_backcast).
    model = cached_model(f"timesfm_cov:{h}", _load)
    point, _ = model.forecast_with_covariates(
        inputs=ctx,
        dynamic_numerical_covariates=dyn,
        xreg_mode="xreg + timesfm",
        normalize_xreg_target_per_input=True,
    )
    return np.asarray(point)[:, :h]


register_backend("toto", toto_forecast)
register_backend("timesfm", timesfm_forecast)
register_backend("tabpfn_ts", tabpfn_ts_forecast)
register_backend("tirex", tirex_forecast)
register_cov_backend("toto_cov", toto_cov_forecast)
register_cov_backend("tabpfn_ts_cov", tabpfn_ts_cov_forecast)
register_cov_backend("timesfm_cov", timesfm_cov_forecast)


# --------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gap-honest zero-shot foundation forecast for one CV window."
    )
    ap.add_argument("--model", required=True, help=f"one of: {', '.join(available_backends())}")
    ap.add_argument("--cut-idx", type=int, required=True, help="train-end _hidx for this window")
    ap.add_argument("--train_csv", default="data/raw/train.csv")
    ap.add_argument("--out-dir", default="data/derived")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizon", type=int, default=GAPPED_HORIZON)
    ap.add_argument(
        "--train-region",
        action="store_true",
        help="also roll the origin back across [0, cut) so a TFT can TRAIN on the channel. Stage 2 "
        "needs this; Stage 1 does not, and it is the larger half of the cost.",
    )
    ap.add_argument("--limit-series", type=int, default=0, help="first N series only (smoke)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="seasonal-naive backend (no package, no GPU) to validate plumbing only",
    )
    args = ap.parse_args()

    fn = _seasonal_naive if args.dry_run else None
    if args.dry_run:
        print("DRY RUN: seasonal-naive backend (placeholder values, NOT a real forecast)")
    run_one(
        args.model,
        args.train_csv,
        args.cut_idx,
        out_dir=args.out_dir,
        horizon=args.horizon,
        device=args.device,
        train_region=args.train_region,
        forecast_fn=fn,
        limit_series=args.limit_series,
    )


if __name__ == "__main__":
    main()
