"""Leakage-free Chronos-2 (zero-shot) forecast as a TFT future covariate.

Feature cascading: produce a ``chronos2_forecast`` column holding Chronos-2's out-of-sample
point forecast of ``target`` at every hour, then feed it to TFT as a known-future covariate
(see ``configs/tft_chronos.yaml`` + the ``chronos2_forecast`` entry in ``src.data.features``).
TFT's Variable Selection Network then learns when to trust the Chronos global prior vs. the
raw covariates — an intelligent, non-linear blend instead of a fixed output average.

LEAKAGE CONTROL — the whole point of this script, and it has TWO levels, not one:
Chronos-2 (``amazon/chronos-2``) is a *pretrained* foundation model; it never trained on THIS
dataset's targets, so there is no fit to leak. What remains is a question about contexts.

* **Context causality** — every ``chronos2_forecast[t]`` is produced from a context ending strictly
  before ``t``, so it never saw ``target[t]``. The default rolling grid below gives this, and it is
  what a model *training* on the train region needs.
* **Gap honesty** — for a gapped window with cutoff ``c``, no covariate value over the forecast
  horizon ``[c, c+672)`` saw ``target`` at or after ``c``. This is what the *graded* task requires,
  and **the 336h rolling grid does not provide it.**

  An earlier version of this docstring claimed "one honest pass; all consumers inherit the honesty
  for free". That was wrong, and expensively so. With ``H=336`` and ``n=4320`` the blocks start at
  3984, 3648, ...; at cutoff ``c=3648`` the scored block ``[3984, 4320)`` draws its covariate from
  block ``s=3984``, whose context ends at 3983 — **336 hours past the cutoff**. Every CV cutoff sits
  on the same grid and leaks identically, over its far half only. The cascade's entire measured
  advantage lived in that half; see ``src.models.cascade_provenance`` for the numbers.

To generate a **gap-honest** covariate for one cutoff, pass ``--cut-idx c``: the train region is
rolled as usual (honest for training) and the whole horizon ``[c, c+horizon)`` is produced as a
SINGLE block from context ``[0, c)`` — which is the forecast actually available at test time.
Either way a ``*.provenance.json`` sidecar records the grid, and ``run_member`` refuses a cascade
member whose sidecar does not clear the window it is being asked to run.

The earliest ``n % 336`` hours of each series have no prior origin and are left NaN — the
shared imputation (``src.data.impute``) fills them with a per-series median and flags them via
``chronos2_forecast_missing``, exactly as for the other planning signals.

THE TRAIN/INFERENCE LEAD MISMATCH, and the ``--train-block`` flag that closes it (plan S4 / 7.2).
The rolling grid is what a model *trains* on, and at ``H=336`` every training row sees a Chronos-2
forecast at lead **1-336**. At inference the scored block sits at lead **337-672** from the same
anchor, so the Variable Selection Network calibrates how far to trust that channel on a quality it
never meets at test time. ``--train-block 672`` rolls the train region in 672h blocks instead, so
training rows span the same 1-672 lead range the horizon block does.

Three properties make that the right way to match rather than merely a convenient one, and all
three depend on ``6 * 672 == 12 * 336 == 4032``:

* **the warm-up is identical** — both grids start at hour 288, so the ``chronos2_forecast_missing``
  pattern is bit-identical across arms and the ablation isolates the lead profile and nothing else;
* **the total step count is identical**, so neither arm is advantaged by more Chronos compute;
* the 672 grid puts a block start **exactly** on cutoffs 3648 and 2976, so those two windows are
  already gap-honest and need no extra call at all (see :func:`stale_hours`).

    python -m src.models.chronos2_oof                  # full run (needs a GPU + chronos)
    python -m src.models.chronos2_oof --limit-series 3 --dry-run   # CPU smoke (no chronos)
    python -m src.models.chronos2_oof --train-block 672 --skip-inference   # the matched grid
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.features import ID, MISSING_SUFFIX, NAN_COLS, TARGET, TIME, futr_exog_list
from src.data.impute import apply_fill, fit_fill_stats
from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import HOUR_IDX, SCORE_LEN, add_hour_index
from src.models import cascade_provenance as cp
from src.models.cascade_provenance import write_provenance

CHRONOS_COL = "chronos2_forecast"
H = SCORE_LEN  # 336 — rolling block size, matches the forecast horizon
WEEK = 168  # weekly period, used by the dry-run seasonal-naive backend


def base_exog() -> list[str]:
    """Covariates fed to Chronos — the futr list minus the column we are generating.

    Robust to whether ``features.py`` has already registered ``chronos2_forecast``: we always
    strip it (and its ``*_missing`` flag) so the generator never conditions on its own output.
    """
    drop = {CHRONOS_COL, CHRONOS_COL + MISSING_SUFFIX}
    return [c for c in futr_exog_list() if c not in drop]


def base_nan() -> list[str]:
    """NaN-prone columns to impute for the Chronos context/future — excludes our own column."""
    return [c for c in NAN_COLS if c != CHRONOS_COL]


def frame_tag(nan_fill: str = "median", block: int = H) -> str:
    """Filename suffix encoding the two properties that make a derived frame a DIFFERENT artifact.

    Both belong in the *name* rather than in a note, because both are silent when wrong. A frame
    generated under a different scattered fill than the member trains under is the provenance
    mismatch that confounded S3's first cascade x interp run; a frame rolled on a different block
    grid is the ablation S4 exists to measure, and writing one over the other would destroy the
    control it is compared against.
    """
    fill = "" if nan_fill == "median" else f"_{nan_fill}"
    return fill if block == H else f"{fill}_h{block}"


def load_long(
    csv_path: str | Path, fill_stats: dict | None, nan_cols: list[str], strategy: str = "median"
):
    """Read a raw CSV -> neuralforecast long format, imputing the base planning signals.

    Mirrors ``src.data.loader.load_long`` but with an explicit ``nan_cols`` so it never touches
    ``chronos2_forecast`` (absent from the raw CSVs). ``validation_input.csv`` has no target
    column; that is fine — the future frame never needs ``y``.
    """
    df = pd.read_csv(csv_path)
    df[TIME] = pd.to_datetime(df[TIME])
    if fill_stats is None:
        fill_stats = fit_fill_stats(df, nan_cols=nan_cols)
    df = apply_fill(df, fill_stats, nan_cols=nan_cols, strategy=strategy)
    rename = {ID: NF_ID, TIME: NF_TIME}
    if TARGET in df.columns:
        rename[TARGET] = NF_TARGET
    return df.rename(columns=rename), fill_stats


def _seasonal_naive(context: pd.DataFrame, future: pd.DataFrame, h: int) -> pd.DataFrame:
    """Dry-run backend: per-series weekly-naive forecast (last 168h tiled). NOT for real results."""
    out = []
    for sid, fg in future.groupby(NF_ID, sort=False):
        hist = context.loc[context[NF_ID] == sid, NF_TARGET].to_numpy()
        if hist.size == 0:
            fill = np.zeros(len(fg))
        else:
            period = hist[-WEEK:] if hist.size >= WEEK else hist
            fill = np.resize(period, len(fg))
        g = fg[[NF_ID, NF_TIME]].copy()
        g[CHRONOS_COL] = fill
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _forecast_block(pipe, context, future, exog, h, batch_series, dry_run) -> pd.DataFrame:
    """Forecast one block -> long df (unique_id, ds, chronos2_forecast), clipped >= 0."""
    if dry_run:
        pred = _seasonal_naive(context, future, h)
    else:
        from src.models.chronos2_eval import _pick_pred_column, _predict

        ctx = context[[NF_ID, NF_TIME, NF_TARGET, *exog]]
        fut = future[[NF_ID, NF_TIME, *exog]]
        raw = _predict(pipe, ctx, fut, h, batch_series)
        pcol = _pick_pred_column(raw)
        pred = raw[[NF_ID, NF_TIME, pcol]].rename(columns={pcol: CHRONOS_COL})
    pred[CHRONOS_COL] = pred[CHRONOS_COL].clip(lower=0.0)
    return pred


def generate_train(pipe, train_long, exog, batch_series, dry_run, block: int = H) -> pd.DataFrame:
    """Rolling-origin OOF forecast over the whole train timeline (origin rolled back in ``block``).

    Block with start index ``s`` forecasts hours ``[s, s+block)`` from context hours ``[0, s)`` — so
    the context ends strictly before every forecast hour (the leakage guarantee). Aligning the
    grid to the train end (``range(n-block, 0, -block)``) puts the train blocks on the same k-ahead
    grid as the validation block, matching the train-fill / inference-fill distributions.

    ``block`` is the LEAD PROFILE the consuming model trains on, and it is S4's whole experiment.
    At the default 336 every training row sits at lead 1-336 while the scored block sits at lead
    337-672; at 672 the training rows span the same range the horizon block does. See the module
    docstring for why 672 is the matched grid and why a strictly-337-672 grid is not.
    """
    df = add_hour_index(train_long)
    n = int(df.groupby(NF_ID)[HOUR_IDX].max().min()) + 1
    starts = list(range(n - block, 0, -block))
    blocks = []
    for i, s in enumerate(starts, 1):
        ctx = df[df[HOUR_IDX] < s]
        fut = df[(df[HOUR_IDX] >= s) & (df[HOUR_IDX] < s + block)]
        # Leakage guard: context ends at s-1, every forecast hour is >= s.
        assert int(ctx[HOUR_IDX].max()) == s - 1, "context overruns the origin"
        assert int(fut[HOUR_IDX].min()) >= s, "forecast hour precedes the origin"
        t0 = time.perf_counter()
        blocks.append(_forecast_block(pipe, ctx, fut, exog, block, batch_series, dry_run))
        print(
            f"  [train {i}/{len(starts)}] block hours [{s}..{s + block - 1}] "
            f"({time.perf_counter() - t0:.1f}s)"
        )
    warmup_hours = n - len(starts) * block if starts else n
    print(f"  warm-up (no prior origin): first {warmup_hours} hours/series stay NaN")
    return (
        pd.concat(blocks, ignore_index=True),
        {"blocks": [(s, block) for s in starts], "n_hours": n},
    )


def generate_horizon(pipe, train_long, exog, batch_series, dry_run, cut_idx, horizon):
    """ONE block: ``[cut, cut+horizon)`` forecast from context ``[0, cut)``.

    This is the forecast actually obtainable at the origin, and therefore the only honest covariate
    for a gapped window. Chronos cannot skip the gap — you forecast all ``horizon`` steps and the
    tail is the scored block — but what matters is the **anchor**, not the length: every value here
    conditions on ``y`` up to ``cut-1`` and nothing after.
    """
    df = add_hour_index(train_long)
    n = int(df.groupby(NF_ID)[HOUR_IDX].max().min()) + 1
    if not 0 < cut_idx <= n:
        raise ValueError(f"cut_idx={cut_idx} outside the train timeline (n={n})")
    ctx = df[df[HOUR_IDX] < cut_idx]
    fut = df[(df[HOUR_IDX] >= cut_idx) & (df[HOUR_IDX] < cut_idx + horizon)]
    assert int(ctx[HOUR_IDX].max()) == cut_idx - 1, "context overruns the cutoff"
    print(
        f"  [horizon] one {horizon}h block [{cut_idx}..{cut_idx + horizon - 1}] "
        f"from ctx [0,{cut_idx})"
    )
    t0 = time.perf_counter()
    pred = _forecast_block(pipe, ctx, fut, exog, horizon, batch_series, dry_run)
    print(f"    ({time.perf_counter() - t0:.1f}s)")
    return pred, n


def stale_hours(base_meta, cut_idx: int, horizon: int, n_hours: int) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` of horizon hours no *keepable* block covers. ``lo >= hi`` = nothing.

    A block starting at ``s`` conditions on ``[0, s)``, so it is legitimate for cutoff ``c``
    whenever ``s <= c``. Everything else over ``[c, c+horizon)`` has to be recomputed. Splitting
    this out of :func:`rehost_gapped` is what lets the caller decide **before spending the GPU**:
    on the 672 grid the block starts land exactly on cuts 3648 and 2976, so both windows come back
    empty here and need no Chronos call at all. Only cut 3312 splices.
    """
    kept = [(s, length) for s, length in cp._blocks(base_meta) if s <= cut_idx]
    stale = [
        t
        for t in range(cut_idx, min(cut_idx + horizon, n_hours))
        if cp.context_end_for_hour({"blocks": kept}, t) is None
    ]
    return (min(stale), max(stale) + 1) if stale else (cut_idx + horizon, cut_idx + horizon)


def rehost_gapped(base_csv, base_meta, new_block, cut_idx, horizon, n_hours):
    """Splice one honest horizon block into an existing frame, keeping every block it may keep.

    **Most of the shipped frame is already honest and does not need recomputing.** A block starting
    at ``s`` conditions on ``[0, s)``, so it is legitimate for a window with cutoff ``c`` whenever
    ``s <= c`` — which covers the entire train region *and* the gap half of the horizon, since the
    grid has a block starting exactly at each CV cutoff. Only the hours whose covering block starts
    **after** ``c`` are contaminated, and those are precisely the scored block.

    So one Chronos call per window replaces ~336 hours per series, instead of three full passes
    regenerating ~4300. The kept gap block is not merely cheaper, it is *better*: a 336-step
    forecast issued at the cutoff, which is exactly what would be available at test time and is a
    shorter lead than the tail of a 672-step one.

    ``new_block`` may be ``None`` exactly when nothing is stale — the case the 672 grid produces at
    two of the three cutoffs. The frame still travels this path so it gets the same normalisation
    and the same sidecar, and the returned block list contains only blocks that were really
    generated: fabricating a ``(cut, horizon)`` entry for a call we never made would put a claim in
    the provenance that the frame cannot support.

    Returns ``(frame, provenance_blocks)``.
    """
    kept = [(s, length) for s, length in cp._blocks(base_meta) if s <= cut_idx]
    lo, hi = stale_hours(base_meta, cut_idx, horizon, n_hours)

    out = base_csv.copy()
    out[TIME] = pd.to_datetime(out[TIME])
    out = out.sort_values([ID, TIME])
    if lo >= hi:
        print(f"  [splice] nothing stale at cut={cut_idx}: all {len(kept)} kept blocks are honest")
        assert new_block is None, "a horizon block was generated for a window that needed none"
        return out, kept
    assert new_block is not None, f"hours [{lo},{hi}) are stale but no horizon block was generated"
    print(f"  [splice] replacing hours [{lo},{hi}) — {hi - lo}/series; keeping {len(kept)} blocks")
    hidx = out.groupby(ID).cumcount()
    new = new_block.rename(columns={NF_ID: ID, NF_TIME: TIME})
    new[TIME] = pd.to_datetime(new[TIME])
    repl = out[[ID, TIME]].merge(new, on=[ID, TIME], how="left")[CHRONOS_COL].to_numpy()
    mask = ((hidx >= lo) & (hidx < hi)).to_numpy()
    out.loc[mask, CHRONOS_COL] = repl[mask]
    assert not out.loc[mask, CHRONOS_COL].isna().any(), "the horizon block did not cover every row"
    return out, [*kept, (cut_idx, horizon)]


def generate_inference(pipe, context_long, future_long, exog, batch_series, dry_run):
    """Forecast the future block (validation/test) from the full observed history as context."""
    n = int(add_hour_index(context_long).groupby(NF_ID)[HOUR_IDX].max().min()) + 1
    pred = _forecast_block(pipe, context_long, future_long, exog, H, batch_series, dry_run)
    # One block starting where the observed history ends: context is everything before it, so this
    # frame is honest by construction for the real submission origin.
    return pred, {"blocks": [(n, H)], "n_hours": n}


def attach(raw_csv: Path, pred_long: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Left-merge the chronos column onto the raw CSV (original schema + 1 col) and write it."""
    raw = pd.read_csv(raw_csv)
    raw[TIME] = pd.to_datetime(raw[TIME])
    keep = set(pred_long[NF_ID].unique())
    raw = raw[raw[ID].isin(keep)]
    p = pred_long.rename(columns={NF_ID: ID, NF_TIME: TIME})
    merged = raw.merge(p[[ID, TIME, CHRONOS_COL]], on=[ID, TIME], how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def _sanity(merged: pd.DataFrame, label: str) -> None:
    """Print the leakage tell-tales: NaN fraction + corr(chronos, target) on observed rows."""
    col = merged[CHRONOS_COL]
    msg = f"[{label}] chronos2_forecast: {float(col.isna().mean()):.1%} NaN (warm-up)"
    if TARGET in merged.columns:
        ok = merged.dropna(subset=[CHRONOS_COL, TARGET])
        if not ok.empty:
            corr = float(np.corrcoef(ok[CHRONOS_COL], ok[TARGET])[0, 1])
            msg += f"; corr(chronos, target)={corr:.3f} (expect ~0.7-0.9 honest, NOT ~1.0)"
    print(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the leakage-free Chronos-2 OOF covariate.")
    ap.add_argument("--config", type=Path, default=Path("configs/chronos2.yaml"))
    ap.add_argument("--train_csv", type=Path, default=Path("data/raw/train.csv"))
    ap.add_argument("--val_csv", type=Path, default=Path("data/raw/validation_input.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/derived"))
    ap.add_argument("--device", default="cuda", help="cuda | cpu | mps")
    ap.add_argument(
        "--nan-fill",
        default="median",
        help="scattered-NaN imputation for the covariates Chronos-2 CONDITIONS ON "
        "(src.data.gap_fill). The generated frame inherits this, so it must match the imputation "
        "the consuming member trains under — a covariate built from median-imputed inputs and fed "
        "to a model trained on interp-imputed ones is a silent provenance mismatch.",
    )
    ap.add_argument(
        "--batch-series", type=int, default=0, help="series per predict_df call (0=all)"
    )
    ap.add_argument("--limit-series", type=int, default=0, help="first N series only (smoke test)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="seasonal-naive backend (no chronos/GPU) to validate plumbing only",
    )
    ap.add_argument(
        "--cut-idx",
        type=int,
        default=0,
        help="generate a GAP-HONEST frame for this cutoff: the horizon [cut, cut+H) becomes one "
        "block from context [0, cut). Omit for the rolling grid, which is context-causal but "
        "NOT gap-honest (see the module docstring). Writes train_chronos_cut<CUT>.csv.",
    )
    ap.add_argument(
        "--reuse",
        type=Path,
        default=Path("data/derived/train_chronos.csv"),
        help="existing frame whose gap-honest blocks (start <= cut) are kept verbatim; only the "
        "contaminated hours are recomputed. Needs its .provenance.json sidecar.",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=2 * H,
        help="forecast horizon for --cut-idx (default 672 = gap + scored block)",
    )
    ap.add_argument(
        "--train-block",
        type=int,
        default=H,
        help="rolling block length for the TRAIN region (default 336). 672 gives the "
        "matched-lead grid of plan S4/7.2: training rows then span the same 1-672 lead range the "
        "scored block sits in. Part of the frame's name (see frame_tag), because the two grids are "
        "different artifacts and the A/B needs both.",
    )
    ap.add_argument(
        "--skip-inference",
        action="store_true",
        help="do not regenerate validation_input_chronos.csv. It is a single block from the full "
        "history and does not depend on the train grid at all, so a --train-block run would write "
        "a byte-equivalent duplicate under a name implying otherwise.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    device = cfg.get("device", args.device)
    batch_series = int(cfg.get("batch_series", args.batch_series))

    nan_cols, exog = base_nan(), base_exog()
    train_long, stats = load_long(args.train_csv, None, nan_cols, args.nan_fill)
    val_long, _ = load_long(args.val_csv, stats, nan_cols, args.nan_fill)

    if args.limit_series > 0:
        keep = sorted(train_long[NF_ID].unique())[: args.limit_series]
        train_long = train_long[train_long[NF_ID].isin(keep)]
        val_long = val_long[val_long[NF_ID].isin(keep)]

    # Resolve the reuse frame BEFORE the model is loaded. On the matched 672 grid the block starts
    # land exactly on two of the three cutoffs, so two windows are already gap-honest and must not
    # pay a GPU container's worth of weight-loading to forecast nothing.
    base_meta, lo, hi, n_hours = None, 0, 0, 0
    if args.cut_idx:
        base_meta = cp.read_provenance(args.reuse)
        if base_meta is None:
            raise SystemExit(
                f"--cut-idx needs {args.reuse} and its provenance sidecar "
                f"({cp.provenance_path(args.reuse).name}) so the honest blocks can be identified. "
                "Generate the rolling frame first, or backfill a sidecar."
            )
        n_hours = int(base_meta["n_hours"])
        lo, hi = stale_hours(base_meta, args.cut_idx, args.horizon, n_hours)

    pipe = None
    if args.dry_run:
        print("DRY RUN: seasonal-naive backend (placeholder values, not real Chronos forecasts)")
    elif args.cut_idx and lo >= hi:
        print("Nothing stale at this cutoff — skipping the Chronos-2 load entirely.")
    else:
        from src.models.chronos2_eval import _load_pipeline

        print(f"Loading amazon/chronos-2 on {device} ...")
        pipe = _load_pipeline(device)

    common = {"column": CHRONOS_COL, "generator": __name__, "zero_shot": True}

    if args.cut_idx:
        # One Chronos call at most. Everything with a block start <= cut is already honest for this
        # window and is kept verbatim — see `rehost_gapped`.
        print(f"Gap-honest covariate for cut_idx={args.cut_idx}, reusing {args.reuse}")
        new_block = None
        if lo < hi:
            new_block, n_seen = generate_horizon(
                pipe, train_long, exog, batch_series, args.dry_run, args.cut_idx, args.horizon
            )
            assert n_seen == n_hours, f"reuse frame says n={n_hours}, train_csv says {n_seen}"
        else:
            print(f"  [horizon] SKIPPED — {args.reuse} is already gap-honest at this cutoff")
        base = pd.read_csv(args.reuse)
        merged, blocks = rehost_gapped(
            base, base_meta, new_block, args.cut_idx, args.horizon, n_hours
        )
        # The imputation and the train grid are both part of the frame's identity: writing an
        # interp or a matched frame over the median 336 one would destroy the control the A/B
        # exists to measure against.
        tag = frame_tag(args.nan_fill, args.train_block)
        train_out = args.out_dir / f"train_chronos{tag}_cut{args.cut_idx}.csv"
        train_out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(train_out, index=False)
        write_provenance(
            train_out,
            **common,
            blocks=blocks,
            n_hours=n_hours,
            note=f"Gap-honest for cut_idx={args.cut_idx} ONLY. Do not use for another cutoff.",
        )
        cp.check_gap_honest(train_out, CHRONOS_COL, args.cut_idx, args.horizon)
        _sanity(merged, f"cut{args.cut_idx}")
        print(f"Wrote {train_out} + sidecar (verified gap-honest)")
        return

    tag = frame_tag(args.nan_fill, args.train_block)
    print(f"Generating train-timeline OOF covariate (rolling grid, block={args.train_block}) ...")
    train_pred, train_meta = generate_train(
        pipe, train_long, exog, batch_series, args.dry_run, args.train_block
    )
    train_out = args.out_dir / f"train_chronos{tag}.csv"
    train_merged = attach(args.train_csv, train_pred, train_out)
    write_provenance(
        train_out,
        **common,
        **train_meta,
        note=f"Rolling grid, block={args.train_block}: context-causal, but NOT gap-honest for any "
        "cutoff on the grid.",
    )
    _sanity(train_merged, "train")
    print(f"Wrote {train_out} + sidecar")

    if args.skip_inference:
        print("Skipped the validation frame: it does not depend on the train grid.")
        return

    print("Generating validation future covariate ...")
    val_pred, val_meta = generate_inference(
        pipe, train_long, val_long, exog, batch_series, args.dry_run
    )
    val_out = args.out_dir / f"validation_input_chronos{tag}.csv"
    val_merged = attach(args.val_csv, val_pred, val_out)
    write_provenance(val_out, **common, **val_meta, note="Single block from the full history.")
    _sanity(val_merged, "validation")
    print(f"Wrote {val_out} + sidecar")


if __name__ == "__main__":
    main()
