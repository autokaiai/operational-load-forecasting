"""Inference entrypoint — the frozen submission contract.

Fixed CLI — do NOT change the arguments or the output schema:

    python predict.py \
        --input_dir /data/input \
        --output_file /output/predictions.csv \
        --checkpoint /submission/checkpoint.pt

Writes ``series_id,timestamp,prediction`` for every row of the forecast index
(``forecast_index_test.csv`` for private eval, else ``forecast_index_validation.csv``).

No internet at inference, and the input dir carries NO target history: the checkpoint is a
self-contained bundle (model weights + scalers + per-series last-window history + imputation
stats — see ``src.bundle``). Covariates are read from ``--input_dir``; the recent target
history comes from the checkpoint. The same code path serves validation and the private test.

The checkpoint is our own trusted, offline artifact, loaded via ``NeuralForecast.load`` (a full
unpickle) rather than ``torch.load(weights_only=True)``; that is safe here by construction.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# Set BEFORE torch is imported anywhere (every torch import in this file is lazy, so this is the
# only place it can take effect). The archive now makes TWO Chronos-2 forward passes per inference,
# and the clean-room run measured the failure mode they create on a small card: 3799 MiB ALLOCATED
# but only 644 MiB free on a 7.6 GiB device — PyTorch had reserved far more than it was using, so a
# 1.02 GiB request failed with ~3 GiB nominally unused. That is fragmentation, not capacity, and
# expandable segments is the allocator mode CUDA's own OOM message recommends for it.
#
# It matters because the failure is SILENT in the worst way: an OOM here does not crash the
# submission, it drops it to rung 2, so a modest grading GPU would quietly ship the previous model.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd  # noqa: E402

from src.data.features import ID, TIME  # noqa: E402
from src.data.loader import NF_ID, NF_TARGET, NF_TIME, build_futr_df
from src.models.fullft_inference import FULLFT_COL

# ``src.bundle`` (and thus neuralforecast) is imported lazily inside main() so the pure helpers
# below — and their schema/contract tests — stay importable without the heavy forecasting stack.

_PRED_COL = "prediction"

# Variance-expansion constant for the rung-1 blend. See the block that applies it, near the end of
# `main`, for the measurement and for why it is 1.04 and not our own frame's optimum of 1.06.
DESMOOTH_GAMMA = 1.04


def _first_match(input_dir: Path, exact: list[str], glob: str) -> Path | None:
    """Schema-driven discovery: prefer the known names, else glob (never bank on private names)."""
    for name in exact:
        if (input_dir / name).exists():
            return input_dir / name
    hits = sorted(p for p in input_dir.glob(glob) if p.is_file())
    return hits[0] if hits else None


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    """Load the rows that need predictions. Globs ``forecast_index*.csv`` so we never depend on
    the private filename (#32 task 3): test/validation names are tried first, then any match."""
    path = _first_match(
        input_dir,
        ["forecast_index_test.csv", "forecast_index_validation.csv"],
        "forecast_index*.csv",
    )
    if path is None:
        raise FileNotFoundError(f"No forecast_index*.csv in {input_dir}.")
    return pd.read_csv(path)


def load_covariates(input_dir: Path) -> pd.DataFrame | None:
    """Load the future covariate table, globbing ``*_input.csv`` so private names don't matter."""
    path = _first_match(input_dir, ["test_input.csv", "validation_input.csv"], "*_input.csv")
    return pd.read_csv(path) if path is not None else None


def horizon_plan(last_observed, forecast_index: pd.DataFrame) -> dict:
    """How far the model must forecast to REACH the requested hours. Derived, never assumed.

    Returns ``gap`` (unobserved hours between the last target and the first requested hour),
    ``span`` (distinct hours requested) and ``h_required = gap + span``.

    The graded run and the leaderboard run differ ONLY in ``gap`` — 336 and 0 respectively — so a
    hardcoded 336 gets the leaderboard wrong and a hardcoded 0 gets the graded run wrong. The
    specification's *"the timeframe might differ"* rules out pinning either, so this reads the
    two files it is given. ``src.data.gap_fill.infer_gap_len`` does the arithmetic; it existed
    long before it was finally wired into the submission path.
    """
    from src.data.gap_fill import infer_gap_len  # pure; no forecasting stack

    ts = pd.to_datetime(forecast_index[TIME])
    gap = infer_gap_len(pd.Timestamp(last_observed), ts.min())
    span = int(ts.nunique())
    return {
        "gap": gap,
        "span": span,
        "h_required": gap + span,
        "first_forecast": ts.min(),
        "last_forecast": ts.max(),
    }


def bag_columns(preds: pd.DataFrame, model_col: str) -> list[str]:
    """Every seed's output column for ``model_col``, in bag order: ``TFT, TFT1, ... TFTN``.

    **The shipped member is a 5-seed bag, and ``nf.predict`` returns one column per seed.** All
    five models live in ONE ``NeuralForecast``, so neuralforecast aliases them by appending an
    index to the class name — and reading ``sidecar["model"]`` alone silently takes the FIRST seed
    and discards the other four. That was worth 13.483 -> 15.794 on the leaderboard: the bag means
    10.6458 over the scored block and seed 0 alone means 10.0345.

    S6 measured what the discarded seeds are worth. Bagging beats the *mean* seed by +0.00574 and
    the *best* of the five by +0.00231 (2.9 SE), and it is the difference between shipping an
    expectation (0.13162) and shipping one draw (0.13517 +- 0.00226).

    The ``\\d*$`` anchor is what keeps quantile columns out: a point head emits ``TFT``, but a
    distributional one would add ``TFT-lo-90`` / ``TFT-median``, and averaging a level into a
    point forecast would be silent rather than loud.
    """
    pat = re.compile(rf"^{re.escape(model_col)}(\d*)$")
    hits = []
    for col in preds.columns:
        m = pat.match(str(col))
        if m:
            hits.append((int(m.group(1) or 0), str(col)))
    return [c for _, c in sorted(hits)]


def assign_predictions(forecast_index: pd.DataFrame, preds: pd.DataFrame, model_col: str):
    """Map predictions onto the index rows **by timestamp**, and prove every row is covered.

    This replaced a positional match, and the replacement is the actual bug fix (#32). Position was
    described as *"robust to absolute-timestamp offsets"*; it is robust to the LABELLING offset and
    blind to the HORIZON one. Sorting both frames and zipping them makes a forecast of the wrong
    hours look perfectly correct — the row count matches, every series is present, the timestamps
    written out are the requested ones — so the only symptom is a silently bad score.

    A timestamp join cannot do that. The model's predictions carry the hours it actually forecast,
    so if those are not the hours that were asked for, the join comes up short and this raises.
    The horizon is now a superset of the request (see ``horizon_plan``), so the join also does the
    slicing: the gap block simply finds no partner and drops out.
    """
    fi = forecast_index[[ID, TIME]].copy()
    fi[TIME] = pd.to_datetime(fi[TIME])
    fi["_order"] = range(len(fi))

    pr = preds.rename(columns={NF_ID: ID, NF_TIME: TIME, model_col: _PRED_COL})[
        [ID, TIME, _PRED_COL]
    ].copy()
    pr[TIME] = pd.to_datetime(pr[TIME])

    merged = fi.merge(pr, on=[ID, TIME], how="left")
    if len(merged) != len(fi):
        raise ValueError(
            f"prediction join changed the row count ({len(fi)} -> {len(merged)}); "
            "the model emitted duplicate (series, timestamp) pairs."
        )
    if merged[_PRED_COL].isna().any():
        miss = merged[merged[_PRED_COL].isna()]
        have = pd.to_datetime(pr[TIME])
        raise ValueError(
            f"{len(miss)} of {len(fi)} requested rows got no prediction — the forecast does not "
            f"cover the requested hours. Requested {fi[TIME].min()}..{fi[TIME].max()}, "
            f"forecast {have.min()}..{have.max()}. First unmatched: "
            f"{miss.iloc[0][ID]} @ {miss.iloc[0][TIME]}."
        )
    return merged.sort_values("_order")[_PRED_COL].to_numpy()


# Retained under the old private name so existing imports keep working; the behaviour is the new
# timestamp join, because the positional one was the defect.
_assign_predictions = assign_predictions


def run_tree(nf, tree: dict, cov, future, fill_stats: dict):
    """Forecast the LightGBM member from its persisted booster. Returns ``[id, ds, lgbm]``.

    The tree is **direct multi-horizon, not a recursive roll**: its lags are origin-anchored
    (``y[o-(lag-1)]``), so every feature is available from the observed history and the whole
    horizon is one ``booster.predict``. That is what makes shipping it cheap.

    Two things must match training exactly or the design is silently wrong rather than loudly:
    the scattered-NaN fill is **interp** (S3's tree-only adoption, +0.00401 3/3), and the unit
    categorical codes come from ``lgbm.unit_codes``, which sorts — so the same 96 series produce
    the same codes here as they did at fit time.
    """
    import lightgbm as lgb
    import numpy as np

    from src.data.features import active_aggregate_columns, futr_exog_list, stat_exog_list
    from src.models import cascade_inference as ci
    from src.models.lgbm import forecast_gapped

    booster = lgb.Booster(model_str=tree["booster"])
    futr_cols, stat_cols = futr_exog_list(), stat_exog_list()

    # The tree's OWN future frame: same rows as the cascade's, different imputation.
    # THE GAP'S FILL STAYS AT THE DEFAULT `median+exact`, AND THAT IS A DECISION (2026-09-04).
    # A handoff proposed `how168+exact+guard` here, worth ~0.0013 at TREE level. It is NOT adopted,
    # for two reasons that compound:
    #   1. It is UNMEASURABLE on this artifact. `predict.py` builds a 672-row horizon even at gap 0,
    #      so the gap fill feeds the model in every regime — but there are no labels for the
    #      validation window, so the only test of the change is the leaderboard itself.
    #   2. Its +0.0013 was a TREE-level number inside a package built around the EWMA tree, which
    #      this project measured and rejected (SPRINT 2C-2: +0.01369 as a member, -0.00001 in the
    #      blend). Tree-level gains discount hard here; that one discounted to zero.
    # `v-prod-1.1` scored 12.141 with the default. An unmeasured input change to the graded artifact
    # on the last submission is the trade this project spent 2026-09-04 undoing.
    futr = build_futr_df(cov, future, fill_stats, nan_strategy=tree.get("nan_fill", "median"))

    hist = ci.history_from_bundle(nf)
    statics = hist.groupby(NF_ID, as_index=False)[stat_cols].first()
    fut = futr.merge(statics, on=NF_ID, how="left")
    fut[NF_TARGET] = np.nan

    # SPRINT 2 — CROSS-SERIES AGGREGATES, materialised HERE or the next line raises KeyError.
    #
    # `futr_exog_list()` above already contains them whenever the bundle's channel is active (it is
    # read at call time, so no change to this file was needed to SEE them). What this file must do
    # is BUILD them, and there are two rules that make the difference between a correct feature and
    # a silently degraded one:
    #
    #  1. AFTER IMPUTATION, NEVER BEFORE. `validation_input.csv` carries 4.4-4.7% NaN in exactly the
    #     covariates being aggregated, and pandas' `skipna=True` default would average ~92 series at
    #     inference against 96 in training WITHOUT RAISING. `build_futr_df` has already filled `fut`
    #     with the tree's own `nan_fill`, and `hist` comes imputed off the bundle, so building here
    #     — after both — is what keeps train and inference on the same construction.
    #  2. PER FRAME, NOT ON THE CONCATENATION. Each aggregate is a within-hour statistic over the 96
    #     series, and both frames are complete 96-series rectangles at every hour they cover
    #     (checked below), so building on each separately gives the identical answer and avoids
    #     depending on the concat order.
    xs_names = active_aggregate_columns()

    # *** ATTACH AFTER THE CONCAT, NEVER PER FRAME. THIS FAILS SILENTLY IF DONE WRONG. ***
    # The A-blocks are WITHIN-HOUR statistics, so building them on `hist` and `fut` separately gives
    # the identical answer and the old per-frame attach was correct for them. **The EWMA blocks walk
    # TIME.** Attached per frame they restart at the history/horizon boundary, every horizon hour
    # loses its accumulation, and what ships is the strict variant — with NO exception raised and a
    # perfectly well-formed CSV. The only way to catch it is to check that a horizon-block EWMA
    # column is non-constant and matches what the training construction gives for the same hour.
    #
    # So: concatenate on the BASE columns, then build the block ONCE over the full timeline.
    # `hist` already carries the aggregates (they were in `futr_exog_list()` at fit time, so
    # `history_from_bundle` returns them baked in) while `fut` does not, so both frames are reduced
    # to the base set first and the block is rebuilt for the joined frame.
    base_cols = [c for c in futr_cols if c not in xs_names]
    cols = [NF_ID, NF_TIME, NF_TARGET, *base_cols, *stat_cols]
    long_df = (
        pd.concat([hist[cols], fut[cols]], ignore_index=True)
        .sort_values([NF_ID, NF_TIME])
        .reset_index(drop=True)
    )
    if xs_names:
        from src.data.xs_attach import attach_active_aggregates

        long_df = attach_active_aggregates(long_df, where="tree frame (history+horizon)")
        # `attach_active_aggregates` adds `_hidx` on its way through; the tree's design does not
        # expect it and `feature_columns` is order-sensitive, so restore the exact column set.
        long_df = long_df[[NF_ID, NF_TIME, NF_TARGET, *futr_cols, *stat_cols]]
    cut_idx = int(hist.groupby(NF_ID)[NF_TIME].size().max())
    horizon = int(fut.groupby(NF_ID)[NF_TIME].size().max())
    print(f"[tree] {tree['member']} fill={tree.get('nan_fill')} cut={cut_idx} h={horizon}")

    out = forecast_gapped(
        long_df,
        cut_idx,
        booster,
        tree["feature_names"],
        horizon=horizon,
        fc_lags=tree.get("fc_lags"),
        categorical_unit=bool(tree.get("categorical_unit", False)),
    )
    return out[[NF_ID, NF_TIME, "lgbm"]]


def _inference_device() -> str:
    """CUDA when there is one. The eval box is assumed to have a GPU; a CPU box must still work."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def attach_cascade_channels(nf, futr_df, sidecar: dict, history=None):
    """Add every cascade covariate the shipped model requires, generating it here and now.

    The channels are read from the MODEL (``src.bundle`` derives them from its ``futr_exog_list``),
    never hardcoded and never taken from the sidecar's ``config`` — the shipped checkpoint has no
    ``cascade`` key at all, and its sidecar records 29 covariates against the models' 31.

    Failure is contained rather than fatal. ``chronos2_forecast`` is worth +8.3%, but a submission
    that raises produces no CSV, and no CSV fails the bonus's own minimum condition ("beats the
    naive last-value baseline"). So an unreachable generator degrades to the stored-median arm with
    ``*_missing = 1`` — loudly, and only for the channel that could not be built.
    """
    from src.models import cascade_inference as ci

    channels = list(sidecar.get("cascade_channels") or [])
    if not channels:
        return futr_df

    unknown = [c for c in channels if c != ci.CHRONOS_COL]
    if unknown:
        # A channel with no producer cannot be degraded honestly: we would be inventing a column
        # the model was trained to trust, with no measured fallback behind it.
        raise ValueError(
            f"the checkpoint requires cascade channel(s) {unknown} for which this build has no "
            "generator; ship a checkpoint whose channels src.models.cascade_inference can produce."
        )

    for channel in channels:
        values = None
        if ci.cascade_enabled():
            source = ci.resolve_source(allow_download=True)
            if source is not None:
                try:
                    values = ci.generate_channel(
                        nf, futr_df, source=source, device=_inference_device(), history=history
                    )
                except Exception as exc:
                    # Includes CUDA OOM: `batch_series=0` sends all 96 series in one call, which is
                    # the archive's peak allocation and the one number S9.0b probes. Retry narrow
                    # before giving the channel up.
                    print(f"[cascade] full-width generation failed ({exc}); retrying batched")
                    try:
                        values = ci.generate_channel(
                            nf,
                            futr_df,
                            source=source,
                            device=_inference_device(),
                            batch_series=16,
                            history=history,
                        )
                    except Exception as exc2:
                        print(f"[cascade] batched generation also failed ({exc2})")
        else:
            print("[cascade] DISABLE_CASCADE set — taking the offline arm deliberately")
        futr_df = ci.attach_channel(futr_df, values, sidecar["fill_stats"], channel=channel)
    return futr_df


def choose_rung(
    *,
    neural_values,
    tree_values,
    fullft_values,
    tree_weight: float,
    fullft_weights: dict | None,
    tree_name: str = "tree",
    bag_name: str = "bag",
):
    """Pick the highest rung whose members are all present, and combine them.

    Pure arithmetic over the three member vectors, extracted from ``main`` ON PURPOSE. The blend is
    the last thing that touches a graded number, and while it lived inline no test could reach it —
    the same shape as the ~300 lines of S8 panel logic that sat in a Modal launcher, and as the bag
    bug that shipped one seed of five past every check the artifact carried.

    Returns ``(rung, values, message)``.

    | rung | arm | CV pooled |
    |------|-----|-----------|
    | 1 | full-FT Chronos + cascade bag + tree | **0.12722** |
    | 2 | cascade bag + tree — TODAY'S SHIPPED MODEL, unchanged | 0.13163 |
    | 4 | tree alone | 0.15200 |

    Rung 3 — the cascade channel blanked to its stored median with ``*_missing = 1`` — is chosen
    inside ``attach_cascade_channels`` rather than here, because by this point it is indistinguish-
    able from rung 2: same members, same weights, a degraded channel inside the bag.

    THE RULE THAT MATTERS: each rung uses ITS OWN measured weights. Rung 2 keeps S6's 0.2438 /
    0.7562
    rather than rung 1's three weights re-spread over two members, because the latter is a model
    nothing has scored. Trading a measured quantity for an invented one is exactly what the
    blanked-channel arm already refuses to do.
    """
    if neural_values is None:
        # Materially worse than any blend (the tree alone is ~0.152 against 0.13163), but the
        # bonus's minimum condition is beating the naive baseline at 0.5471 and the tree clears
        # that comfortably. Producing nothing does not.
        return 4, tree_values, f"!! DEGRADED: {tree_name} ALONE, the neural bag produced nothing"
    if tree_values is None:
        return 2, neural_values, f"{bag_name} alone; no tree in the checkpoint"
    if fullft_values is not None and fullft_weights:
        wf = float(fullft_weights["chronos_full_ft"])
        wb = float(fullft_weights["cascade_bag"])
        wt = float(fullft_weights["tree"])
        total = wf + wb + wt
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rung 1 weights sum to {total!r}, not 1.0: {fullft_weights!r}")
        values = wf * fullft_values + wb * neural_values + wt * tree_values
        return 1, values, f"{wf:.4f}*{FULLFT_COL} + {wb:.4f}*{bag_name} + {wt:.4f}*{tree_name}"
    values = tree_weight * tree_values + (1.0 - tree_weight) * neural_values
    return 2, values, f"{tree_weight:.2f}*{tree_name} + {1 - tree_weight:.2f}*{bag_name}"


def _free_cuda(device: str) -> None:
    """Drop collectable CUDA blocks between two attempts at the same forecast."""
    import gc

    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def run_fullft(nf, futr_df, sidecar: dict):
    """RUNG 1's third member: the full fine-tuned Chronos-2, or ``None`` to drop to rung 2.

    Never fatal, and never renormalised. If the weights cannot be fetched we return ``None`` and
    the caller falls back to rung 2's OWN measured weights (0.2438 tree / 0.7562 bag = 0.13163).
    Re-spreading rung 1's three weights across two members would be a model no measurement
    describes, which is exactly the trade the cascade's blank-channel arm already refuses.
    """
    from src.models import fullft_inference as ff

    if not ff.fullft_enabled():
        print("[fullft] DISABLE_FULLFT set — taking rung 2 deliberately")
        return None
    source = ff.resolve_source(allow_download=True)
    if source is None:
        return None

    device = _inference_device()
    err = None
    try:
        return ff.generate_member(nf, futr_df, source=source, device=device)
    except Exception as exc:
        # Keep the MESSAGE, drop the EXCEPTION. `except ... as exc` holds `exc` alive for the rest
        # of the block, and `exc.__traceback__` references the frames that reference the CUDA
        # activations — so retrying INSIDE this block runs with the failed attempt's memory still
        # pinned. Measured on an 8 GB card: the first attempt left 3342 MiB allocated after its own
        # release, and the batched retry then failed by 1.3 MiB (854.69 free, 856.00 needed).
        err = f"{type(exc).__name__}: {exc}"

    # `exc` is out of scope here, so its traceback and the tensors it pinned are collectable.
    _free_cuda(device)
    print(f"[fullft] full-width generation failed ({err}); retrying batched")
    try:
        return ff.generate_member(nf, futr_df, source=source, device=device, batch_series=16)
    except Exception as exc2:
        print(f"[fullft] batched generation also failed ({type(exc2).__name__}: {exc2})")
        return None


def rollout_predict(nf, sidecar: dict, cov, needed: int, *, needs_futr: bool, model_col: str):
    """Reach PAST the checkpoint's fixed ``h`` by feeding the model its own forecasts back.

    **This is a fallback and never runs on the graded path.** The TFT is a *direct* multi-horizon
    model: one forward pass emits exactly ``h`` steps, so a request needing ``gap + span > h``
    cannot be served by it. The arithmetic says that cannot arise here — train ends at hour 4319,
    the private block is 4656-4991, so ``gap + span = 672`` exactly, and the dataset's 4992-hour
    timeline has nothing after it. But *"the timeframe might differ"* is a statement about a harness
    we cannot inspect, and the minimum condition is beating the naive baseline (0.5471) — **which
    no CSV at all fails outright.** So an impossible-but-catastrophic case gets
    a degraded forecast rather than an exception, exactly like the cascade's offline arm and the
    tree-alone rung below it.

    Each block conditions on the observed history PLUS the blocks already predicted, including a
    freshly generated cascade channel, so the TFT and Chronos see the same context. Errors compound
    across blocks — this is a recursive rollout of a model never trained for one, and **its accuracy
    is unmeasured**. It is insurance, not a second ship path.
    """
    import math

    from src.data.features import stat_exog_list
    from src.models import cascade_inference as ci

    h = int(nf.models[0].h)
    blocks = math.ceil(needed / h)
    print(
        f"[predict] !! ROLLOUT: need {needed}h from a checkpoint with h={h}; "
        f"rolling {blocks} blocks, feeding forecasts back as history. UNMEASURED ACCURACY."
    )

    hist = ci.history_from_bundle(nf)
    # Intersect rather than assume: a bundle fitted without static covariates carries none of these
    # columns, and `groupby(...)[missing]` raises KeyError instead of returning an empty frame.
    stat_cols = [c for c in stat_exog_list() if c in hist.columns]
    statics = hist.groupby(NF_ID, as_index=False)[stat_cols].first() if stat_cols else None
    fill_stats = sidecar["fill_stats"]
    collected = []

    for b in range(blocks):
        last = hist.groupby(NF_ID)[NF_TIME].max()
        step = pd.Timedelta(hours=1)
        idx = pd.concat(
            [
                pd.DataFrame(
                    {
                        ID: uid,
                        TIME: pd.date_range(t + step, periods=h, freq="h"),
                    }
                )
                for uid, t in last.items()
            ],
            ignore_index=True,
        )

        futr = None
        if needs_futr:
            from src.data.features import active_aggregate_columns
            from src.data.xs_attach import attach_active_aggregates

            futr = build_futr_df(cov, idx, fill_stats)
            # SPRINT 2. The ROLLOUT path — this is the one the leaderboard block (gap 0) takes, and
            # it needs the aggregates for the same reason the direct path does: the bag was TRAINED
            # with them. `build_futr_df` no longer raises on their absence (it cannot build derived
            # columns), so without this the frame would be quietly short of covariates the models
            # were fitted on. `statics` is already in scope here and carries `nominal_capacity`,
            # which A9 weights by.
            if active_aggregate_columns():
                _cols = list(futr.columns)
                futr = futr.merge(statics, on=NF_ID, how="left")
                futr = attach_active_aggregates(futr, where="rollout horizon")
                futr = futr[_cols + [c for c in active_aggregate_columns() if c not in _cols]]
            futr = attach_cascade_channels(nf, futr, sidecar, history=hist)

        preds = nf.predict(df=hist, static_df=statics, futr_df=futr)
        if NF_ID not in preds.columns:
            preds = preds.reset_index()
        seeds = bag_columns(preds, model_col)
        preds = preds.assign(**{model_col: preds[seeds].mean(axis=1)})
        block = preds[[NF_ID, NF_TIME, model_col]].copy()
        block[NF_TIME] = pd.to_datetime(block[NF_TIME])
        collected.append(block)
        lo, hi = block[NF_TIME].min(), block[NF_TIME].max()
        print(f"[predict]    block {b + 1}/{blocks}: {lo}..{hi}")

        if b + 1 == blocks:
            break
        # Feed it back: the block's own covariates plus the forecast standing in for the target.
        nxt = futr if futr is not None else idx.rename(columns={ID: NF_ID, TIME: NF_TIME})
        if statics is not None:
            nxt = nxt.merge(statics, on=NF_ID, how="left")
        nxt = nxt.merge(block.rename(columns={model_col: NF_TARGET}), on=[NF_ID, NF_TIME])
        hist = pd.concat([hist, nxt[hist.columns]], ignore_index=True).sort_values([NF_ID, NF_TIME])

    return pd.concat(collected, ignore_index=True).drop_duplicates(subset=[NF_ID, NF_TIME])


def main() -> None:
    """Load the checkpoint bundle and write predictions for every forecast-index row."""
    parser = argparse.ArgumentParser(description="Generate forecast predictions.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    from src import bundle  # lazy: pulls neuralforecast only when actually running inference
    from src.models.registry import supports_futr  # same — reads the model's capability flags

    forecast_index = load_forecast_index(args.input_dir)
    nf, sidecar = bundle.load(args.checkpoint)
    # SPRINT 2. Reactivate the derived cross-series aggregates the bundle was FITTED with, before
    # anything calls `futr_exog_list()`. Same contract as `cascade_channels`: the sidecar records
    # what the models' own column list implies, and this restores it. `set_aggregate_columns([])`
    # for a bundle that has none, so an older archive behaves exactly as before.
    from src.data.features import set_aggregate_columns

    _xs = list(sidecar.get("aggregate_columns") or [])
    set_aggregate_columns(_xs)
    if _xs:
        print(f"[xs] bundle declares {len(_xs)} cross-series aggregate(s): {_xs}")
    model_col = sidecar["model"]
    needs_futr = supports_futr(model_col)  # does this architecture consume known-future covariates?

    # The hours the model will ACTUALLY forecast, straight from the checkpoint's stored history.
    # Everything below is derived from this rather than from a constant: `future` is h steps from
    # the last observed target, and the requested block may sit anywhere inside it.
    future = nf.make_future_dataframe()
    future = future.rename(columns={NF_ID: ID, NF_TIME: TIME})
    future[TIME] = pd.to_datetime(future[TIME])
    last_observed = future[TIME].min() - pd.Timedelta(hours=1)

    plan = horizon_plan(last_observed, forecast_index)
    h_available = int(future.groupby(ID)[TIME].size().max())
    print(
        f"[predict] last observed {last_observed} | requested "
        f"{plan['first_forecast']}..{plan['last_forecast']} | gap {plan['gap']}h | "
        f"span {plan['span']}h | h required {plan['h_required']} | h available {h_available}"
    )
    # h < h_required is the #32 condition: with h=336 and a 336h gap the model reaches only halfway
    # to the graded block, and the old positional match wrote those near-block numbers onto
    # far-block labels without a murmur. It must never be answered by forecasting the wrong hours —
    # but it need not be answered by producing nothing either, so it routes to a recursive rollout
    # (see rollout_predict). Unreachable on the graded run: gap 336 + span 336 = h exactly.
    needs_rollout = h_available < plan["h_required"]

    # The TREE needs no rollout even when the bag does: it is DIRECT multi-horizon with
    # origin-anchored lags (`y[o-(lag-1)]`, constant across the horizon), so any number of steps is
    # one `booster.predict` from the same origin. Only its `horizon_step` feature runs off the end
    # of what it was trained on, where a tree saturates at its last split rather than failing.
    future_tree = future
    if needs_rollout:
        extra = plan["h_required"] - h_available
        step = pd.Timedelta(hours=1)
        tail = pd.concat(
            [
                pd.DataFrame({ID: uid, TIME: pd.date_range(t + step, periods=extra, freq="h")})
                for uid, t in future.groupby(ID)[TIME].max().items()
            ],
            ignore_index=True,
        )
        future_tree = pd.concat([future, tail], ignore_index=True).sort_values([ID, TIME])

    cov = None
    if needs_futr:
        cov = load_covariates(args.input_dir)
        if cov is None:
            raise FileNotFoundError(
                f"{model_col} needs covariates: expected test_input.csv or "
                f"validation_input.csv in {args.input_dir}."
            )

    # ---- THE TREE FIRST. It is the one member that CANNOT fail for an environmental reason: a
    # serialised booster restored from the checkpoint, a design matrix, one predict call. No
    # download, no network, no GPU. Running it before the neural bag is deliberate — it means a
    # catastrophic failure there still leaves a shippable forecast rather than no CSV at all.
    tree = sidecar.get("tree")
    weight = float((sidecar.get("blend") or {}).get("tree_weight", 0.0)) if tree else 0.0
    tree_values = None
    if tree and weight > 0:
        tree_preds = run_tree(nf, tree, cov, future_tree, sidecar["fill_stats"])
        tree_values = assign_predictions(forecast_index, tree_preds, "lgbm")
    elif tree:
        print(f"[predict] tree present but weight={weight:.2f}; not blending it")

    # ---- THE NEURAL BAG. Its cascade channel may degrade (see attach_cascade_channels); this
    # guard is for the tier below that, where the bag cannot run at all.
    neural_values = None
    base_futr = None
    try:
        if needs_rollout:
            preds = rollout_predict(
                nf, sidecar, cov, plan["h_required"], needs_futr=needs_futr, model_col=model_col
            )
        elif needs_futr:
            # Built ONCE and kept: the full-FT member must be forecast over exactly the rows the
            # TFT was handed, or the two arms are blended across different futures.
            # SPRINT 2 — SAME ORDERING TRAP AS THE TREE PATH, and it is fatal here rather than
            # silent: `build_futr_df` validates its output against the LIVE `futr_exog_list()`
            # (loader.py:192), which contains the aggregates whenever the bundle declares them, and
            # it builds from `cov`, which does not. Build with the channel off, then attach.
            #
            # THE BAG WAS TRAINED WITH THESE COLUMNS, so this is not optional dressing: without
            # them `nf.predict` is handed a frame missing covariates the models were fitted on.
            from src.data.features import (
                active_aggregate_columns,
                aggregate_columns,
                stat_exog_list,
            )
            from src.data.xs_attach import attach_active_aggregates
            from src.models import cascade_inference as ci

            with aggregate_columns([]):
                base_futr = build_futr_df(cov, future, sidecar["fill_stats"])
            if active_aggregate_columns():
                # A9 weights by `nominal_capacity`, a STATIC absent from the horizon frame; take it
                # off the checkpoint's own history, then drop the statics again so the frame handed
                # to `nf.predict` carries exactly the model's futr columns.
                _stat_cols = stat_exog_list()
                _hist = ci.history_from_bundle(nf)
                _statics = _hist.groupby(NF_ID, as_index=False)[_stat_cols].first()
                _cols = list(base_futr.columns)
                base_futr = base_futr.merge(_statics, on=NF_ID, how="left")
                base_futr = attach_active_aggregates(base_futr, where="neural horizon")
                base_futr = base_futr[
                    _cols + [c for c in active_aggregate_columns() if c not in _cols]
                ]
            preds = nf.predict(futr_df=attach_cascade_channels(nf, base_futr.copy(), sidecar))
        else:
            preds = nf.predict()
        if NF_ID not in preds.columns:
            preds = preds.reset_index()
        # AVERAGE THE BAG. `nf.predict` returns one column per seed and the member that ships is
        # the mean of all five, not the first one — see bag_columns.
        seeds = bag_columns(preds, model_col)
        if not seeds:
            raise ValueError(
                f"no output column for model {model_col!r} in {list(preds.columns)}; "
                "the checkpoint's sidecar disagrees with what the models emit."
            )
        print(f"[predict] {model_col} bag of {len(seeds)}: {', '.join(seeds)}")
        preds = preds.assign(**{model_col: preds[seeds].mean(axis=1)})
        neural_values = _assign_predictions(forecast_index, preds, model_col)
    except Exception as exc:
        if tree_values is None:
            raise  # nothing to fall back to; a wrong CSV is worse than a loud failure
        print(f"[predict] !! THE NEURAL BAG FAILED ({type(exc).__name__}: {exc})")

    # ---- RUNG 1's THIRD MEMBER. Attempted only when the bag and the tree both stand and the
    # rung-1 weights travel in the checkpoint: a member without its measured weight is unusable.
    fullft_values = None
    fullft_w = (sidecar.get("blend") or {}).get("fullft")
    if neural_values is not None and tree_values is not None and fullft_w and base_futr is not None:
        member = run_fullft(nf, base_futr, sidecar)
        if member is not None:
            fullft_values = assign_predictions(forecast_index, member, FULLFT_COL)

    predictions = forecast_index[[ID, TIME]].copy()
    rung, values, msg = choose_rung(
        neural_values=neural_values,
        tree_values=tree_values,
        fullft_values=fullft_values,
        tree_weight=weight,
        fullft_weights=fullft_w,
        tree_name=tree["member"] if tree else "tree",
        bag_name=f"{model_col}_bag{len(nf.models)}",
    )
    # *** A DEGRADED RUNG MUST BE LOUD, NOT A LINE IN A SCROLLBACK. ***
    # Rung 1 is the model every reported number describes. Rungs 2 and 4 are FALLBACKS: they still
    # emit a perfectly well-formed CSV of the right shape with plausible values, so nothing
    # downstream can tell that a member went missing. This project has already been bitten once by
    # exactly that — a dead HuggingFace revision pin silently dropped the run to rung 2 and the
    # output looked entirely normal. So the fallback announces itself in a form an evaluator cannot
    # miss, names the member that failed, and states the accuracy cost from our own CV.
    print(f"[predict] RUNG {rung} — {msg}")
    if rung != 1:
        _cost = {
            2: "the full fine-tuned Chronos-2 member is MISSING. CV pooled WAPE 0.12029 -> 0.13163 "
            "(+9.4% relative error).",
            4: "BOTH neural members are MISSING; this is the tree alone. CV pooled WAPE "
            "0.12029 -> ~0.152 (+26% relative error). It still beats the naive baseline "
            "(0.5471), which is why it emits rather than aborts.",
        }.get(rung, "a member is missing.")
        bar = "!" * 78
        print(
            f"\n{bar}\n"
            f"!! WARNING — DEGRADED PREDICTION. This run fell back to RUNG {rung}.\n"
            f"!! {_cost}\n"
            f"!! The output below is a VALID, correctly-shaped forecast, but it is NOT the\n"
            f"!! model the report describes. Most likely causes: the `chronos` package is\n"
            f"!! unavailable here, or the pinned HuggingFace revision could not be fetched.\n"
            f"{bar}\n",
            flush=True,
        )
        import sys as _sys

        print(
            f"[predict] WARNING: DEGRADED — rung {rung}, not rung 1. See stdout for details.",
            file=_sys.stderr,
            flush=True,
        )

    predictions[_PRED_COL] = values

    # ---- DE-SMOOTHING (gamma). Blending three members, one of them a 5-seed bag, shrinks the
    # prediction's dispersion toward each unit's mean: measured at 0.870 of the target's own
    # within-unit spread, against 0.947 for a single unshrunk model. WAPE is minimised by the
    # conditional MEDIAN, so an under-dispersed blend is systematically biased against the metric --
    # visible as +9.1% vs consensus in the lowest quintile and -3.4% in the highest.
    #
    #     pred' = unit_mean + gamma * (pred - unit_mean)
    #
    # MEASURED on the three CV windows, fitted on `blk < 224` and scored on `blk >= 224`:
    # 0.12029 -> 0.11807, +0.00222 at SE 0.00017 (13.1 SE), 3/3 windows, and a PLATEAU over
    # gamma in [1.02, 1.08] rather than a knife edge. An independent refit of gamma on our own fit
    # region returns 1.040 to three decimals. Per-series gamma was tried and is a NULL (+0.00002
    # for 96 free parameters, spread 0.995-1.105): the over-smoothing is a global property of the
    # blending operation, so one scalar is the right model rather than a simplification.
    #
    # 1.04 AND NOT 1.06. Our own frame's optimum is 1.06 (+0.00260) and taking it would be
    # selecting the parameter on the region we then report. 1.04 was fixed on a DIFFERENT cube and
    # applied blind here, which is why it is the defensible value.
    #
    # RUNG 1 ONLY. gamma was measured on the three-member blend. Rung 2 is a different mixture with
    # its own weights and its own dispersion, and no measurement covers it -- an unmeasured
    # correction on the degraded fallback path is exactly the risk this project does not take.
    #
    # NO LABELS ARE READ. The anchor is the mean of OUR OWN prediction over that unit's forecast
    # block, so this is legal at inference and needs nothing the scorer holds.
    # EVERY RUNG, not rung 1 only. Over-smoothing is a property of AVERAGING, and every rung
    # averages: rung 2 blends two members, and even rung 4's tree-alone output is a 1181-tree
    # boosted mean. The correction is applied to the BLEND OUTPUT rather than per member because
    # the per-unit mean is linear — sum_c w_c (b_c + g(p_c - b_c)) = B + g(P - B) — so the two are
    # algebraically identical and this is one parameter instead of three.
    _anchor = predictions.groupby(ID)[_PRED_COL].transform("mean")
    predictions[_PRED_COL] = _anchor + DESMOOTH_GAMMA * (predictions[_PRED_COL] - _anchor)
    print(f"[predict] de-smoothed at gamma={DESMOOTH_GAMMA} (rung {rung})")

    # The target is strictly positive; never emit NaN/Inf (the scorer hard-rejects them).
    predictions[_PRED_COL] = predictions[_PRED_COL].clip(lower=0.0)
    fallback = float(predictions[_PRED_COL].median())
    predictions[_PRED_COL] = pd.to_numeric(predictions[_PRED_COL], errors="coerce").fillna(fallback)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
