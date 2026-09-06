"""The member **runner** registry — one uniform way to produce any member's window predictions.

Why this module exists
----------------------
``src/eval/protocol.py`` is the grading layer and it is fully member-agnostic: every function takes
``member: str`` and nothing in it knows what a LightGBM is. The *runner* layer was not. Producing a
member's predictions meant an ``if member == "lgbm" / elif member == "chronos_ft" / elif member in
NF_MEMBERS`` dispatch in ``scripts/member_preds_window.py``, and anything that needed a variation on
it — a specific seed, a parameter override — re-implemented the call by hand. That is how a seed
probe came to carry its own copy of the LightGBM call, and how the copy in
``member_preds_window`` came to strip ``seed`` while the copy in the probe set it.

The asymmetry mattered because Phase 5 Tier A adds four members (XGBoost, CatBoost, LightGBM
``dart`` and a seed-bagged LightGBM) through that same dispatch. Four members meant four new
branches in three scripts. Here they are four ``register(...)`` calls, and two of them are pure
parameter overrides with no new code at all.

The contract a member runner satisfies
--------------------------------------
Given a :class:`RunContext` (a window's imputed long frame + its train-end ``cut_idx``), return the
scored block as::

    unique_id, ds, cutoff, y, <member>

``run_member`` enforces that contract on the way out — see :func:`validate_member_frame`. It is a
tripwire, not a repair function: a member that returns ragged series or NaN predictions fails here
rather than three scripts downstream, where it would surface as a quietly wrong WAPE.

``cutoff`` is emitted here, which is what ``src.eval.protocol.validate_prediction_df`` has always
required and no artifact carried. Concatenating windows now yields a frame that validator accepts
directly. Existing consumers are unaffected: they select ``[unique_id, ds, y, <member>]``
explicitly before merging.

Registry vs registry
--------------------
``src/models/registry.py`` maps a *neuralforecast architecture name* to its class (``build_nf``).
This maps a *member name* to the procedure that produces its predictions for one CV window. A single
architecture can back several members (``lgbm`` and ``lgbm_dart`` share ``src.models.lgbm``), so the
two are deliberately separate.

Lazy imports are load-bearing: ``neuralforecast`` must not be imported at module scope, because the
Chronos Modal image does not have it. Every runner imports its own dependencies inside the call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.features import CASCADE_FORECASTS_ALL, ID, TIME, cascade_channels
from src.data.impute import fit_fill_stats
from src.data.loader import (
    NF_ID,
    NF_TARGET,
    NF_TIME,
    add_volume_sample_weight,
    load_long,
    static_frame,
)
from src.eval.protocol import CUTOFF
from src.eval.splits import (
    BLOCK_REGIMES,
    HOUR_IDX,
    SCORE_LEN,
    add_hour_index,
    gapped_horizon,
    take_block,
)
from src.models.cascade_provenance import check_gap_honest
from src.models.lgbm import WEEKLY_LAGS

MEMBER_COLUMNS = [NF_ID, NF_TIME, CUTOFF, NF_TARGET]  # + the member's own prediction column

# Gap (336) + scored block (336). The window's forecast horizon, and therefore the span a cascade
# covariate has to be honest over.
GAPPED_HORIZON = 2 * SCORE_LEN

# neuralforecast architectures that can be driven as members; each needs a `configs/<name>.yaml`.
NF_MEMBERS = ("tft", "lstm", "bitcn", "nhits", "tide", "dlinear")

# Keys in a tree config that are consumed by the runner, not by the booster. Anything left here
# would reach LightGBM as an unknown parameter — warned about, then hidden by `verbosity: -1`.
_TREE_RUNNER_KEYS = {
    "model",
    "name",
    "origin_stride",
    "early_stopping_rounds",
    "max_boost_round",
    "fc_lags",
    "categorical_unit",
    "normalise_level",
    "recency_halflife",
}


# --------------------------------------------------------------------------- context + spec


@dataclass(frozen=True)
class RunContext:
    """Everything a member runner needs for one (member, window) unit of work.

    ``long_df`` must already be imputed on the window's OWN train slice — use
    :func:`load_window_long`, never ``src.data.loader.load_long`` directly, or the fill statistics
    peek past the cutoff.
    """

    long_df: pd.DataFrame
    cut_idx: int
    seed: int | None = None
    gap_cov: str = "impute"
    adapter: str | None = None
    device: str = "cuda"
    batch_series: int = 0
    max_steps: int | None = None
    # How the 336h gap block's absent covariates are reconstructed (src.data.gap_fill). Only
    # meaningful when gap_cov="impute"; "real" supplies the genuine rows and reconstructs nothing.
    # Inference-only — the fit never sees it (see _nf_runner) — so N strategies cost N predicts,
    # not N trains, and every arm shares identical weights.
    gap_fill: str = "median"
    # Which half of the 672h forecast becomes the cube: "far" (steps 337-672, the graded private-
    # test scenario) or "near" (steps 1-336, the public leaderboard's scenario). See
    # src.eval.splits.take_block. **The default is far and must stay far** — a near cube is an
    # opt-in diagnostic (final-push lane 1E), never something a training run, a config or a CI
    # path selects on its own, because every weight this project holds was fitted on far.
    regime: str = "far"


@dataclass(frozen=True)
class MemberSpec:
    """A member's identity and capabilities, plus the callable that runs it.

    The capability flags exist so callers can *ask* rather than assume. ``seedable`` in particular
    is checked rather than ignored: reporting a ``mean ± std`` over five runs of a deterministic
    member would fabricate a precision that does not exist, so ``run_member`` refuses a seed a
    member cannot honour instead of silently dropping it. That silent drop is exactly the defect
    plan 3.6 found in ``configs/lgbm.yaml``.
    """

    name: str
    kind: str  # tree | neural | foundation
    run: Callable[[RunContext], pd.DataFrame]
    seedable: bool
    needs_gpu: bool
    needs_adapter: bool = False
    honours_gap_cov: bool = True
    # measured  — a number exists for it in results/ and is quotable
    # untested  — registered, never run; `available_members(status="measured")` excludes it so a
    #             sweep cannot pick it up by accident
    # cut       — deliberately not run, and the note says why. Kept rather than deleted: a decision
    #             not to spend compute is a result, and a deleted registration cannot carry it.
    status: str = "measured"
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    # A member that conditions on a cascade covariate needs BOTH of these, and needs them
    # together: `cascade` puts the column in the conditioning set, `train_csv` points at the only
    # frame that actually contains it. Declaring one without the other is the silent failure —
    # the cascade column present but unused, or requested but absent.
    cascade: tuple[str, ...] = field(default_factory=tuple)
    train_csv: str | None = None


_REGISTRY: dict[str, MemberSpec] = {}


def register(spec: MemberSpec) -> MemberSpec:
    """Add a member to the registry. Re-registering the same name is an error, not an override."""
    if spec.name in _REGISTRY:
        raise ValueError(f"member {spec.name!r} is already registered")
    if spec.kind not in ("tree", "neural", "foundation"):
        raise ValueError(f"member {spec.name!r}: unknown kind {spec.kind!r}")
    unknown = set(spec.cascade) - set(CASCADE_FORECASTS_ALL)
    if unknown:
        raise ValueError(f"member {spec.name!r}: unknown cascade channel(s) {sorted(unknown)}")
    if spec.cascade and not spec.train_csv:
        raise ValueError(
            f"member {spec.name!r} declares cascade channels {list(spec.cascade)} but no "
            "train_csv. The raw CSVs do not carry cascade columns, so the member would train on "
            "an all-NaN covariate. Point train_csv at the derived frame that has it."
        )
    _REGISTRY[spec.name] = spec
    return spec


def get_member(name: str) -> MemberSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown member {name!r}; registered: {available_members()}")
    return _REGISTRY[name]


def available_members(kind: str | None = None, *, status: str | None = None) -> list[str]:
    """Registered member names, optionally filtered by ``kind`` and/or ``status``."""
    return sorted(
        n
        for n, s in _REGISTRY.items()
        if (kind is None or s.kind == kind) and (status is None or s.status == status)
    )


# --------------------------------------------------------------------------- contract


def validate_member_frame(df: pd.DataFrame, member: str, cut_idx: int) -> pd.DataFrame:
    """Assert one window's member output is well-formed and return it in canonical column order.

    Raises ``AssertionError`` on any violation. Deliberately strict about *equal* per-series row
    counts: a ragged frame pools to a WAPE that is silently weighted by whichever series happened
    to return more rows.
    """
    cols = [*MEMBER_COLUMNS, member]
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"{member}: missing column(s) {missing}; have {sorted(df.columns)}"
    assert len(df) > 0, f"{member}: empty prediction frame at cut_idx={cut_idx}"

    for col in (NF_TARGET, member):
        n_nan = int(df[col].isna().sum())
        assert n_nan == 0, f"{member}: {n_nan} NaN in column {col!r} at cut_idx={cut_idx}"

    per_series = df.groupby(NF_ID).size()
    assert per_series.nunique() == 1, (
        f"{member}: ragged output at cut_idx={cut_idx} — per-series row counts range "
        f"{per_series.min()}..{per_series.max()} (expected {SCORE_LEN} for every series). "
        "Pooled WAPE would silently over-weight the longer series."
    )

    dupes = int(df.duplicated(subset=[NF_ID, NF_TIME]).sum())
    assert dupes == 0, f"{member}: {dupes} duplicate (unique_id, ds) rows at cut_idx={cut_idx}"

    return df[cols].sort_values([NF_ID, NF_TIME]).reset_index(drop=True)


def run_member(member: str, ctx: RunContext) -> pd.DataFrame:
    """Run one member for one window and return its validated scored-block frame.

    The single entry point every script should use. Checks the context against the member's
    declared capabilities *before* spending the compute, then checks the output against the member
    contract after.
    """
    spec = get_member(member)
    if spec.needs_adapter and not ctx.adapter:
        raise ValueError(f"member {member!r} requires an adapter (the window's LoRA dir)")
    if ctx.seed is not None and not spec.seedable:
        raise ValueError(
            f"member {member!r} is deterministic — it cannot honour seed={ctx.seed}. "
            "Check `MemberSpec.seedable` before looping over seeds; a mean +- std over identical "
            "runs would claim a precision the member does not have."
        )
    if ctx.gap_cov not in ("real", "impute"):
        raise ValueError(f"gap_cov must be 'real' or 'impute', got {ctx.gap_cov!r}")
    # The near block is a DIAGNOSTIC and it has to be opted into by name, per member. A runner that
    # has not been taught the axis would silently return its far block under a near label — the
    # worst possible failure, because the number looks fine and is answering a different question.
    if ctx.regime not in BLOCK_REGIMES:
        raise ValueError(f"regime must be one of {BLOCK_REGIMES}, got {ctx.regime!r}")
    if ctx.regime != "far" and not getattr(spec.run, "supports_regime", False):
        raise ValueError(
            f"member {member!r} has no regime-aware runner, so regime={ctx.regime!r} would write "
            "its FAR block under a NEAR label. Teach its runner src.eval.splits.take_block first."
        )
    # Resolve the strategy BEFORE the fit: a typo or a scattered-only strategy must cost a second,
    # not a GPU-hour. get_strategy refuses one that does not support the gap surface rather than
    # letting it degrade silently (interp across 336 unanchored hours is a forward fill).
    from src.data.gap_fill import get_strategy

    get_strategy(ctx.gap_fill, "gap")
    if ctx.gap_fill != "median" and not spec.honours_gap_cov:
        raise ValueError(
            f"member {member!r} does not read the gap block (honours_gap_cov=False), so "
            f"gap_fill={ctx.gap_fill!r} would change nothing and report a difference of exactly "
            "zero as if it were a result. Its design matrix reads futr covariates at forecast "
            "hours [origin+337, origin+672], strictly after the gap. Vary the SCATTERED surface "
            "for this member instead (src.data.impute.apply_fill)."
        )

    # Scoped, so a cascade member cannot leak its extra covariate into the next member run in the
    # same process — which the paired A/B runner does by design.
    with cascade_channels(*spec.cascade):
        if spec.cascade:
            have = set(ctx.long_df.columns)
            absent = [c for c in spec.cascade if c not in have]
            assert not absent, (
                f"{member}: cascade column(s) {absent} are not in the frame. Load it from "
                f"{spec.train_csv!r} (see MemberSpec.train_csv), not the raw train.csv — "
                "otherwise this trains a plain model that silently ignores the cascade."
            )
            # Present is not the same as honest. A cascade column is a forecast, so its validity
            # depends entirely on which hours its context was allowed to see — a fact no column
            # carries. Verified against the recorded grid, before the GPU spend, and fail-closed:
            # a leaky covariate scores *better*, so an unchecked one fails in the flattering
            # direction. See src.models.cascade_provenance.
            frame = member_train_csv(member, spec.train_csv, ctx.cut_idx)
            for channel in spec.cascade:
                check_gap_honest(frame, channel, ctx.cut_idx, GAPPED_HORIZON)
        out = spec.run(ctx)
    if CUTOFF not in out.columns:
        out = out.assign(**{CUTOFF: ctx.cut_idx})
    return validate_member_frame(out, member, ctx.cut_idx)


# --------------------------------------------------------------------------- shared loading


def member_train_csv(member: str | None, default: str, cut_idx: int | None = None) -> str:
    """The labelled CSV a member needs — its own derived frame if it declares one.

    A cascade member's frame is **per window**, because a cascade covariate is only gap-honest for
    the cutoff it was anchored at (see ``src.models.cascade_provenance``). So ``train_csv`` may
    carry a ``{cut}`` placeholder, and a member that declares one cannot be resolved without a
    cutoff — asking for it is a programming error, not something to paper over with a default.
    """
    if not member:
        return default
    path = get_member(member).train_csv or default
    if "{cut}" not in path:
        return path
    if cut_idx is None:
        raise ValueError(
            f"member {member!r} needs a per-window frame ({path}) but no cut_idx was given. Its "
            "covariate is gap-honest for one cutoff only, so there is no window-agnostic answer."
        )
    return path.format(cut=cut_idx)


def load_window_long(
    train_csv: str, cut_idx: int, *, member: str | None = None, nan_fill: str = "median"
) -> pd.DataFrame:
    """Load the labelled CSV with imputation fitted ONLY on the window train slice (``_hidx<cut``).

    Mirrors ``src.data.loader.load_long`` but the fill medians come from the train slice alone, so
    no covariate statistic ever peeks at the scored block (plan pitfall #5).

    Pass ``member`` for a cascade member. Its channel has to be active *here*, not only inside the
    runner: imputation is what fills the cascade column's warm-up NaNs and adds its ``*_missing``
    flag, and that happens during loading. Load without it and the column reaches the model still
    NaN over its first 288 hours.

    ``nan_fill`` selects how the SCATTERED ~4.5% NaNs are reconstructed (``src.data.gap_fill``).
    This is the surface that reaches EVERY member including the tree — unlike the 336h gap, which
    the tree is provably invariant to. ``"median"`` is the incumbent and is bit-exact with the
    previous behaviour.
    """
    spec_cascade = get_member(member).cascade if member else ()
    with cascade_channels(*spec_cascade):
        return _load_window_long(train_csv, cut_idx, nan_fill=nan_fill)


def _load_window_long(train_csv: str, cut_idx: int, nan_fill: str = "median") -> pd.DataFrame:
    raw = pd.read_csv(train_csv)
    raw[TIME] = pd.to_datetime(raw[TIME])
    raw = raw.sort_values([ID, TIME])
    hidx = raw.groupby(ID).cumcount()
    fill_stats = fit_fill_stats(raw[hidx < cut_idx])
    long_df, _ = load_long(train_csv, fill_stats=fill_stats, strategy=nan_fill)
    return long_df


def _load_yaml(path: str) -> dict:
    p = Path(path)
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


# --------------------------------------------------------------------------- runners


def _lgbm_runner(name: str, *, config: str = "configs/lgbm.yaml", overrides: Mapping | None = None):
    """Build a runner for a LightGBM-family member. Variants are parameter overrides, not code.

    The config is read WITHOUT merging ``configs/base.yaml``. ``src.train.load_config`` merges it,
    which sends every neural-runner key (``h``, ``input_size``, ``max_steps``, ``accelerator``, …)
    into the booster's parameter dict. LightGBM ignores unknown parameters with a warning that
    ``verbosity: -1`` then hides, so this was invisible rather than harmless — and CatBoost raises
    on unknown parameters instead of warning, so it would not have stayed invisible for Tier A.
    Behaviour-preserving for LightGBM: the only base keys it recognises are ``seed`` and
    ``learning_rate``, and ``configs/lgbm.yaml`` sets both itself.

    ``seed`` is NOT stripped — it is a real LightGBM parameter. Stripping it (the old
    ``member_preds_window`` behaviour) made ``configs/lgbm.yaml`` declare a seed that never reached
    the booster; every run silently inherited ``DEFAULT_PARAMS['seed'] = 42``.
    """

    def run(ctx: RunContext) -> pd.DataFrame:
        from src.models import lgbm as lgbm_mod

        merged = {**_load_yaml(config), **(overrides or {})}
        params = {k: v for k, v in merged.items() if k not in _TREE_RUNNER_KEYS}
        if ctx.seed is not None:
            # LightGBM derives the bagging and feature-sampling streams from `seed`; pinning them
            # explicitly documents what `feature_fraction` and `bagging_fraction` actually consume.
            params["seed"] = int(ctx.seed)
            params["bagging_seed"] = int(ctx.seed)
            params["feature_fraction_seed"] = int(ctx.seed)
        report: dict = {}
        out = lgbm_mod.predict_gapped(
            ctx.long_df,
            ctx.cut_idx,
            params=params or None,
            origin_stride=int(merged.get("origin_stride", 168)),
            early_stopping_rounds=int(merged.get("early_stopping_rounds", 0)),
            max_boost_round=int(merged.get("max_boost_round", 3000)),
            tuning_report=report,
            fc_lags=merged.get("fc_lags"),
            categorical_unit=bool(merged.get("categorical_unit", False)),
            normalise_level=bool(merged.get("normalise_level", False)),
            recency_halflife=float(merged.get("recency_halflife", 0.0)),
            regime=ctx.regime,
        )
        if report:
            # The round count is a *result* of this run, not a setting. Print it: a silently
            # truncated fit (hit_ceiling) and a converged one look identical in the WAPE alone.
            print(
                f"[{name}] early stopping: best_iteration={report['best_iteration']} "
                f"(valid l1 {report['best_score']:.4f}, ceiling {report['max_boost_round']}, "
                f"hit_ceiling={report['hit_ceiling']}), refit on the full train region",
                flush=True,
            )
        return out.rename(columns={"lgbm": name})

    # What this runner's predictions actually depend on, declared rather than inferred. A cube
    # cache keyed on the member *name* would serve a stale hit the moment an override changed —
    # and a stale hit is not wasted time, it is a wrong number. See `src.models.cube_cache`.
    run.member_config = config
    run.member_overrides = dict(overrides or {})
    run.member_code = ("src/models/lgbm.py",)
    # One fit produces both halves of the 672h inference grid, so the near cube is a slice of the
    # same booster rather than a second model. Free on CPU (final-push lane 1E).
    run.supports_regime = True
    return run


def _catboost_runner(
    name: str, *, config: str = "configs/catboost.yaml", overrides: Mapping | None = None
):
    """Build a runner for a CatBoost-family member. Same shape as ``_lgbm_runner``, by design.

    The config is read WITHOUT merging ``configs/base.yaml``, for the reason 3.9 found and this
    member is the first to actually need: **CatBoost raises on an unknown parameter** where LightGBM
    ignores it with a warning that ``verbosity: -1`` then hides. Merging the base config would send
    ``h``, ``input_size``, ``max_steps``, ``accelerator`` … straight into the booster's constructor.
    The fix landed a phase before the member that would have died on it.

    ``seed`` is translated rather than passed: LightGBM's parameter is ``seed``, CatBoost's is
    ``random_seed``, and handing CatBoost the wrong spelling is a crash rather than a silent
    default — which is the better failure, but only if nobody writes the wrong spelling.
    """

    def run(ctx: RunContext) -> pd.DataFrame:
        from src.models import catboost_tree as cb

        merged = {**_load_yaml(config), **(overrides or {})}
        params = {k: v for k, v in merged.items() if k not in _TREE_RUNNER_KEYS}
        if ctx.seed is not None:
            params["random_seed"] = int(ctx.seed)
        report: dict = {}
        out = cb.predict_gapped(
            ctx.long_df,
            ctx.cut_idx,
            params=params or None,
            origin_stride=int(merged.get("origin_stride", 168)),
            early_stopping_rounds=int(merged.get("early_stopping_rounds", 0)),
            max_boost_round=int(merged.get("max_boost_round", 3000)),
            tuning_report=report,
            fc_lags=merged.get("fc_lags"),
            categorical_unit=bool(merged.get("categorical_unit", False)),
            normalise_level=bool(merged.get("normalise_level", False)),
            recency_halflife=float(merged.get("recency_halflife", 0.0)),
            regime=ctx.regime,
        )
        if report:
            print(
                f"[{name}] early stopping: best_iteration={report['best_iteration']} "
                f"(valid MAE {report['best_score']:.4f}, ceiling {report['max_boost_round']}, "
                f"hit_ceiling={report['hit_ceiling']}), refit on the full train region",
                flush=True,
            )
        return out.rename(columns={cb.MEMBER_COL: name})

    run.member_config = config
    run.member_overrides = dict(overrides or {})
    run.member_code = ("src/models/catboost_tree.py", "src/models/lgbm.py")
    # Same shape as the LightGBM runner, and it has to be: one fit, one inference grid, both halves.
    run.supports_regime = True
    return run


def vsn_importance(nf, futr: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Variable-selection weights off a fitted TFT. Empty dict for a model that has no VSN.

    Plan S4.3 / S7: the exposé promises a VSN importance figure and nothing extracts one, and it is
    also the single measurement that settles S3's open cascade hypothesis — whether better
    in-sample covariates make the VSN down-gate the Chronos-2 channel that is worth +8.3%.

    THE WEIGHTS LIVE ON THE MODEL AND DIE WITH IT. ``TFT.forward`` overwrites
    ``interpretability_params`` on every pass, so this reads whatever the LAST forward saw and
    cannot be recovered from a saved cube afterwards. That is why the export has to be threaded
    into the run rather than bolted on later.

    ``mean_on_batch`` then averages over that last batch only — with a small
    ``inference_windows_batch_size`` the figure would describe a SUBSET of units while looking like
    all of them. So this issues its own predict with the batch size raised past the series count,
    costs seconds and makes the row "averaged over all 96 units" true rather than approximately
    true.
    """
    model = nf.models[0]
    if not hasattr(model, "feature_importances"):
        return {}
    prev = getattr(model, "inference_windows_batch_size", None)
    n_series = int(futr[NF_ID].nunique()) if futr is not None else 0
    try:
        if prev is not None and n_series:
            model.inference_windows_batch_size = max(int(prev), n_series)
        nf.predict(futr_df=futr) if futr is not None else nf.predict()
        return model.feature_importances()
    finally:
        if prev is not None:
            model.inference_windows_batch_size = prev


def write_vsn_importance(nf, out_dir: Path, member: str, futr: pd.DataFrame | None = None) -> int:
    """Write each importance table to ``{out_dir}/{member}_vsn_<table>.csv``. Returns the count."""
    tables = vsn_importance(nf, futr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, frame in tables.items():
        slug = label.lower().replace(" ", "_")
        frame.to_csv(out_dir / f"{member}_vsn_{slug}.csv")
        print(f"[vsn] {member}: {label} {frame.shape} -> {out_dir}", flush=True)
    return len(tables)


# neuralforecast's own default. `BaseModel.__init__` runs `pl.seed_everything(self.random_seed)`,
# which OVERWRITES `registry.set_seed(cfg["seed"])` — and `seed` is a non-model key, so it was never
# forwarded. Every neural run in this project has therefore trained at an effective seed of 1.
HISTORICAL_NF_SEED = 1


def seeded_cfg(cfg: dict, seed: int) -> dict:
    """Make a seed actually reach a neuralforecast model. Pure, so it is directly testable.

    ``seed`` alone does NOT do it: ``registry.set_seed`` seeds python/numpy/torch and then
    ``BaseModel.__init__`` calls ``pl.seed_everything(self.random_seed, workers=True)`` and undoes
    it. TFT's ``random_seed`` defaults to 1, so an unseeded run — and every run this project has
    ever done — sits at seed 1 regardless of what the config says.

    Nothing recorded is wrong because of it: both arms of every paired A/B sat at the same
    effective seed. What it broke is the *capability* — ``seedable=True`` was a false claim for the
    nf members, ``member_seed_noise.py`` would have reported std = 0 as a measurement, and S5's
    Tier D seed-bagging would have averaged five copies of one model. Caught by
    ``tests/test_sweep_space.py`` before a container was spent on it.

    Only called when a caller EXPLICITLY supplies a seed, so an unseeded run keeps the historical
    regime and every recorded number still reproduces bit-for-bit.
    """
    return {**cfg, "seed": int(seed), "random_seed": int(seed)}


def nf_fit_and_sweep(
    name: str,
    cfg_path: str,
    ctx: RunContext,
    arms: list[tuple[str, str]],
    vsn_out: Path | None = None,
    overrides: Mapping | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit ONE model, then predict once per gap-fill arm. The whole point of plan S3 Finding 3.

    The gap-fill strategy is an **inference-only** change: ``nf.fit`` sees the train slice, whose
    covariates are real, and the withholding is applied only to ``futr_df`` at predict time. So N
    strategies cost ``1 fit + N predicts`` rather than N fits::

        naive    7 arms x 3 windows x 2 members = 42 TFT trains
        actual   3 fits per member, 21 predicts  -> 6 containers

    The efficiency is the smaller half of it. **The correctness is the point**: every arm shares
    *identical weights*, so the cross-run early-stopping variance that caveated S2 Stage 2 (~0.001,
    about the size of that phase's best deficit) is exactly ZERO here rather than merely small. A
    paired bootstrap over these arms measures the covariate fill and nothing else.

    ``arms`` is a list of ``(label, spec)``. ``spec`` is a gap-fill strategy name, or ``"real"`` to
    supply the genuine covariates — which makes the long-open true-vs-imputed delta for
    W1/W2 fall out free, as one more futr variant rather than another run.
    """
    import gc

    import torch

    from src.data.features import futr_exog_list
    from src.models.registry import build_nf, supports_futr, supports_stat
    from src.train import load_config  # lazy: pulls neuralforecast

    # S5: a trial's sampled hyperparameters enter as ordinary CONFIG VALUES, merged here rather
    # than patched onto the model later. That is deliberate — it means a swept dimension travels
    # the identical path a config key travels (`build_model` forwards it, or `_NON_MODEL_KEYS`
    # routes it to `build_loss`), so the totality guard in tests/test_sweep_space.py covers the
    # sweep and the configs at once instead of covering one and hoping about the other.
    cfg = {**load_config(cfg_path), **(overrides or {})}
    model_col = cfg["model"]
    df = add_hour_index(ctx.long_df)
    train = df[df[HOUR_IDX] < ctx.cut_idx].drop(columns=HOUR_IDX)
    # Fixed 672h gapped window [cut, cut+672) via the shared helper — NOT every row >= cut_idx,
    # which over-runs at earlier cutoffs and mis-scores the wrong block (see gapped_horizon).
    horizon = gapped_horizon(df, ctx.cut_idx).drop(columns=HOUR_IDX)
    h = int(horizon.groupby(NF_ID).size().min())

    wbs = int(cfg.get("windows_batch_size", 128))
    gcfg = {
        **cfg,
        "h": h,
        "windows_batch_size": max(8, wbs // 2),
        "inference_windows_batch_size": max(8, wbs // 2),
    }
    if ctx.seed is not None:
        gcfg = seeded_cfg(gcfg, ctx.seed)
        # BOTH, and the second one is the load-bearing half. `registry.set_seed` seeds
        # python/numpy/torch — and then `BaseModel.__init__` calls
        # `pl.seed_everything(self.random_seed, workers=True)`, which OVERWRITES it. `random_seed`
        # defaults to 1 and `seed` is a non-model key, so it was never forwarded: every neural run
        # in this project has trained at an effective seed of 1, and `ctx.seed` moved nothing.
        #
        # Nothing recorded is wrong because of it — every arm of every paired A/B sat at the same
        # effective seed — but it made `seedable=True` a false claim for the nf members, and it
        # would have made seed-bagging average five copies of one model and report std = 0 as a
        # measurement. Caught by tests/test_sweep_space.py before S5's Tier D spent a container.
        #
        # Forwarded ONLY when a caller explicitly asks for a seed, so an unseeded run keeps the
        # historical regime and every recorded number still reproduces bit-for-bit (S4 measured
        # that drift guard at max|Δpred| = 0 and it stays valid).
    if ctx.max_steps:  # smoke override: a few steps just to exercise the code path
        gcfg["max_steps"] = int(ctx.max_steps)
    # The volume-weighted (WAPE-aligned) loss, opt-in per config.
    #
    # THIS IS WHERE IT HAD TO GO, and its absence is why the backbone had never trained under the
    # aligned loss: `add_volume_sample_weight` was only ever called from `src/train.py`, and NO
    # member path goes through `src/train.py` — every neural member reaches the model via
    # run_member -> nf_fit_and_sweep -> build_nf -> nf.fit. So the flag existed, the helper
    # existed, and nothing connected them.
    #
    # neuralforecast consumes a `sample_weight` column natively (it scales `outsample_mask` per
    # window), in BOTH the training and validation steps — so this also makes the `ptl/val_loss`
    # early-stop monitor volume-aligned, which is #55's "monitor WAPE rather than the training
    # loss" delivered by fixing the loss instead of by writing a new monitor. Asserted rather than
    # trusted in tests/test_sweep_space.py, because a docstring has been the least reliable thing
    # in the room five times.
    if gcfg.get("sample_weight") == "volume":
        train = add_volume_sample_weight(train)

    # --- final-push lane 1G: residual target -------------------------------------------------
    #
    # Train on ``y - <channel>`` and add the channel back at inference, instead of handing the
    # channel to the VSN as a covariate to be trusted or distrusted. Same data, same backbone, same
    # horizon — a target transform, not architecture.
    #
    # THE ORDER OF THESE TWO BLOCKS IS LOAD-BEARING AND IS ASSERTED, NOT COMMENTED.
    # ``add_volume_sample_weight`` derives each series' weight from the MAD of ``NF_TARGET`` so
    # that neuralforecast's robust scaler cancels and plain MAE becomes the WAPE numerator
    # (src/data/loader.py). Replace the target first and that MAD is computed on the *residual*:
    # the WAPE alignment breaks silently and the lane returns a confident null with no error
    # message. So the weight is computed from the original ``y`` above, and the assertion below
    # proves it rather than trusting this comment.
    resid_channel = cfg.get("residual_target")
    if resid_channel:
        if resid_channel not in train.columns:
            raise AssertionError(
                f"{name}: residual_target={resid_channel!r} is not in the train frame — load the "
                "member's derived cascade frame, not the raw train.csv."
            )
        y_orig = train[NF_TARGET].to_numpy(dtype=float).copy()
        base = train[resid_channel].to_numpy(dtype=float)
        if np.isnan(base).any():
            raise AssertionError(f"{name}: residual_target {resid_channel!r} has NaN in train")
        train = train.copy()
        train[NF_TARGET] = y_orig - base
        if "sample_weight" in train.columns:
            # The trap, closed twice over: the weights must still describe the ORIGINAL series
            # scale, so they must NOT reproduce the residual's own per-series MAD.
            resid_w = add_volume_sample_weight(train[[NF_ID, NF_TIME, NF_TARGET]].copy())[
                "sample_weight"
            ].to_numpy(dtype=float)
            have = train["sample_weight"].to_numpy(dtype=float)
            assert not np.allclose(have, resid_w), (
                f"{name}: sample_weight matches the RESIDUAL's per-series MAD, i.e. it was "
                "computed after the target was replaced. Compute it from the original y."
            )
        print(
            f"[{name}] residual target: y - {resid_channel} "
            f"(train mean |y|={np.abs(y_orig).mean():.4f} -> "
            f"|resid|={np.abs(train[NF_TARGET].to_numpy(float)).mean():.4f})",
            flush=True,
        )

    nf = build_nf(gcfg)
    fit_kwargs = {"val_size": h}
    if supports_stat(model_col):
        fit_kwargs["static_df"] = static_frame(train)
    nf.fit(train, **fit_kwargs)

    labels = take_block(horizon, ctx.regime)[[NF_ID, NF_TIME, NF_TARGET]]
    gap_len = h - SCORE_LEN
    out: dict[str, pd.DataFrame] = {}
    for label, spec in arms:
        if supports_futr(model_col):
            futr = horizon[[NF_ID, NF_TIME, *futr_exog_list()]]
            # THE NEAR BLOCK **IS** THE WITHHELD BLOCK. `_withhold_gap_covariates` replaces the
            # first `gap_len` rows per series, which under regime="far" is the 336h gap nobody
            # scores — and under regime="near" is exactly the block being scored. Withholding there
            # measures "forecast 336h with every planning signal blanked", a scenario that does not
            # exist: the near/validation case supplies genuine known-future covariates for those
            # hours. Caught the hard way — the first 1E dispatch scored the near cascade at
            # 0.336/0.699/0.373 against a far block of ~0.13 on the SAME rows.
            if ctx.regime == "near" and spec != "real" and gap_len > 0:
                raise ValueError(
                    f"{name}: regime='near' with gap fill {spec!r} would withhold the covariates "
                    "of the very rows being scored. The near regime is the validation scenario — "
                    "no gap, real known-future covariates. Pass gap_cov='real'."
                )
            if spec != "real" and gap_len > 0:
                from src.models.chronos2_eval import _withhold_gap_covariates

                futr = _withhold_gap_covariates(
                    futr, train, gap_len, strategy=spec, cut_idx=ctx.cut_idx
                )
            preds = nf.predict(futr_df=futr)
        else:
            preds = nf.predict()
        if resid_channel:
            # Add the baseline back on the horizon's OWN rows, keyed on (unique_id, ds) rather
            # than positionally: `preds` comes back sorted by neuralforecast's own convention and
            # a positional add would silently mis-align one series against another.
            preds = preds.merge(
                horizon[[NF_ID, NF_TIME, resid_channel]], on=[NF_ID, NF_TIME], how="left"
            )
            assert not preds[resid_channel].isna().any(), (
                f"{name}: {resid_channel!r} is absent for some horizon rows — cannot add the "
                "residual baseline back."
            )
            preds[model_col] = preds[model_col] + preds[resid_channel]
        scored = take_block(preds, ctx.regime)[[NF_ID, NF_TIME, model_col]]
        merged = scored.merge(labels, on=[NF_ID, NF_TIME]).rename(columns={model_col: name})
        out[label] = merged[[NF_ID, NF_TIME, NF_TARGET, name]]
        # Inside the loop and after the predict, because the weights describe the frame THAT
        # predict was given — exporting once after the loop would silently attribute the last
        # arm's gating to every arm.
        if vsn_out is not None:
            write_vsn_importance(
                nf, Path(vsn_out) / label, name, futr if supports_futr(model_col) else None
            )
    del nf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _nf_runner(name: str, *, config: str | None = None):
    """Build a runner for a neuralforecast member: fit to the window's train slice, forecast, score.

    ``gap_cov`` controls the gap (cut → cut+336) known-future covariates fed to the rollout:
    ``impute`` median-imputes them (+ missing=1) — the #32 conservative floor; ``real`` feeds the
    genuine train.csv gap rows. All members must use the SAME setting in a given run, else the
    comparison is apples-to-oranges.
    """
    cfg_path = config or f"configs/{name}.yaml"

    def run(ctx: RunContext) -> pd.DataFrame:
        # ONE code path with the sweep, deliberately: a separate single-arm implementation would
        # drift from the swept one, and then a Wave-3 arm and its own control would differ by more
        # than the thing under test.
        spec = "real" if ctx.gap_cov == "real" else ctx.gap_fill
        return nf_fit_and_sweep(name, cfg_path, ctx, [("only", spec)])["only"]

    # Declared, not inferred: `tft_cascade` is driven by configs/tft_chronos.yaml, so anything
    # resolving a config from the member NAME would fit a different model than the registry says.
    run.member_config = cfg_path
    run.supports_regime = True  # nf_fit_and_sweep slices via take_block(ctx.regime)
    return run


def _chronos_ft_runner(name: str = "chronos_ft"):
    """FT-Chronos standalone preds using the window's LoRA adapter.

    ``gap_cov`` picks the run_mode: ``impute`` -> gapped-realistic (gap signals withheld),
    ``real`` -> gapped-optimistic (gap signals supplied) — matching the nf members' condition.
    Inference-only against a fixed adapter, so the member is deterministic: ``seedable=False``.
    """

    def run(ctx: RunContext) -> pd.DataFrame:
        from src.models.chronos2_eval import _load_pipeline, run_mode

        mode = "gapped-realistic" if ctx.gap_cov == "impute" else "gapped-optimistic"
        pipe = _load_pipeline(ctx.device, adapter=ctx.adapter)
        _, preds = run_mode(
            pipe,
            ctx.long_df,
            mode,
            None,
            ctx.batch_series,
            return_preds=True,
            cut_idx=ctx.cut_idx,
            regime=ctx.regime,
        )
        return preds.rename(columns={"prediction": name})[[NF_ID, NF_TIME, NF_TARGET, name]]

    run.supports_regime = True  # one 672h forward pass; the near block is its other half
    return run


# --------------------------------------------------------------------------- registrations

register(
    MemberSpec(
        name="lgbm",
        kind="tree",
        run=_lgbm_runner("lgbm"),
        seedable=True,
        needs_gpu=False,
        # predict_gapped never traverses the gap — its lags are origin-anchored and its covariates
        # are read at the forecast hour — so `real` and `impute` produce identical output. Plan 3.5
        # moves the neural members' numbers but NOT this one.
        honours_gap_cov=False,
        note="Direct-multistep LightGBM, L1 objective. Pooled FAR (late) 0.1628 at seed 42.",
    )
)

register(
    MemberSpec(
        name="lgbm_recency",
        kind="tree",
        run=_lgbm_runner(
            "lgbm_recency",
            overrides={
                # 8 weeks. The train region spans ~3600h, so the oldest origins keep ~0.16 weight
                # while the last month stays near 1 — a tilt, not a truncation.
                "recency_halflife": 1344,
                "early_stopping_rounds": 50,
                "max_boost_round": 3000,
            },
        ),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        status="untested",
        note=(
            "Plan 4.5a: exponential decay in the age of the training origin. The direct answer to "
            "the July-shift drift risk (#54) that linear_tree was meant to be, and unlike "
            "linear_tree it composes with L1. Compare against lgbm_es; retune the half-life "
            "alongside origin_stride if 4.4 lands."
        ),
        tags=("phase4", "weights"),
    )
)

register(
    MemberSpec(
        name="lgbm_norm",
        kind="tree",
        run=_lgbm_runner(
            "lgbm_norm",
            overrides={
                "normalise_level": True,
                "early_stopping_rounds": 50,
                "max_boost_round": 3000,
            },
        ),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        status="untested",
        note=(
            "Plan 4.6: model z = y/level, weight by level, multiply back. Both arms are "
            "WAPE-aligned, so the A/B against lgbm_es isolates the REPRESENTATION gain — trees "
            "split on absolute thresholds, so an unnormalised target burns early splits "
            "separating units by level instead of learning shared temporal shape."
        ),
        tags=("phase4", "target"),
    )
)

register(
    MemberSpec(
        name="lgbm_unitcat",
        kind="tree",
        run=_lgbm_runner(
            "lgbm_unitcat",
            overrides={
                "categorical_unit": True,
                "early_stopping_rounds": 50,
                "max_boost_round": 3000,
            },
        ),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        status="untested",
        note=(
            "Plan 4.2c: unit id as a genuine LightGBM categorical. The tree currently sees the 96 "
            "units only through 3 static covariates. Wired for CatBoost since the stacker, never "
            "for LightGBM. Compare against lgbm_es."
        ),
        tags=("phase4", "features"),
    )
)

register(
    MemberSpec(
        name="lgbm_stride24",
        kind="tree",
        run=_lgbm_runner(
            "lgbm_stride24",
            overrides={
                "origin_stride": 24,
                "early_stopping_rounds": 50,
                "max_boost_round": 3000,
            },
        ),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        status="untested",
        note=(
            "Plan 4.4: one training origin per day instead of per week — ~7x the rows. Compare "
            "against lgbm_es. The round count is measured in both arms, which matters here: more "
            "rows at a fixed budget is a different experiment from more rows."
        ),
        tags=("phase4", "data"),
    )
)

register(
    MemberSpec(
        name="lgbm_wlag",
        kind="tree",
        run=_lgbm_runner(
            "lgbm_wlag",
            overrides={
                "fc_lags": WEEKLY_LAGS,
                # Measured on top of the honest round count, not the stale 600 — otherwise the
                # extra features are judged at a budget chosen for a smaller basis, and a real gain
                # can read as a wash. Plan 4.1a exists so every later lever compounds cleanly.
                "early_stopping_rounds": 50,
                "max_boost_round": 3000,
            },
        ),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        status="measured",
        note=(
            "Plan 4.2a: forecast-hour-anchored lags 672/840/1008 (4/5/6 weeks) on top of lgbm_es. "
            "NULL RESULT — 0.1622 vs lgbm_es 0.1620, CI [-0.0012, +0.0009], 1 of 3 windows. The "
            "features do reach the model (round count moves 668->1186 on W0), so they are simply "
            "redundant with lag_504/lag_336 plus the calendar encodings. Kept registered as the "
            "reproducible negative result, not as a candidate."
        ),
        tags=("phase4", "features"),
    )
)

register(
    MemberSpec(
        name="lgbm_es",
        kind="tree",
        run=_lgbm_runner(
            "lgbm_es", overrides={"early_stopping_rounds": 50, "max_boost_round": 3000}
        ),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        status="measured",
        tags=("phase4-1a",),
        note="Plan 4.1a: round count measured on a held-out fold, then refit on the full train "
        "region. Pooled FAR (late) 0.1620 at seed 42 vs lgbm's 0.1629 — measured counts "
        "668/966/768 against a hand-set 600, so the incumbent was under-boosted in every window. "
        "THE PHASE-4 "
        "BASELINE: measure every later lever against this, not against lgbm.",
    )
)

# --- Phase 5 Tier A: CUT 2026-07-31, kept registered as the record ---------------------------
# These were the payoff for the registry: two Tier A members as pure parameter overrides rather than
# a dispatch branch each. They were never run, and S2 cut them.
#
# Correcting the record while cutting them: they were never *discarded*, they were never *measured*
# — `status` was "untested" throughout and no Phase-4 composite included them. The cut is on the
# two-cluster finding (plan S2.0), not on a result: `lgbm|lgbm_s24_unitcat` is err-corr **0.976**,
# and that is one library at two parameter settings. A third LightGBM variant is a cluster-B
# duplicate by construction, and S1 froze the tree, so there is nothing for a better one to be
# better *at*. `status="cut"` rather than deletion, because "we chose not to run this and here is
# why" is a result the write-up wants and a deleted registration cannot carry.

register(
    MemberSpec(
        name="lgbm_dart",
        kind="tree",
        run=_lgbm_runner("lgbm_dart", overrides={"boosting_type": "dart", "drop_rate": 0.1}),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        status="cut",
        tags=("phase5-tierA", "cut-s2"),
        note="DART dropout regularisation. NB: dart ignores early stopping — fix rounds by hand. "
        "CUT (2026-07-31): never measured, and cluster-B by construction.",
    )
)

register(
    MemberSpec(
        name="lgbm_extra_trees",
        kind="tree",
        run=_lgbm_runner("lgbm_extra_trees", overrides={"extra_trees": True}),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        status="cut",
        tags=("phase5-tierA", "cut-s2"),
        note="Extremely-randomised split thresholds — bagging-flavoured regularisation. "
        "CUT (2026-07-31): never measured, and cluster-B by construction.",
    )
)

for _nf_name in NF_MEMBERS:
    register(
        MemberSpec(
            name=_nf_name,
            kind="neural",
            run=_nf_runner(_nf_name),
            seedable=True,
            needs_gpu=True,
            status="measured" if _nf_name in ("tft", "lstm", "bitcn") else "untested",
            note=f"neuralforecast architecture; needs configs/{_nf_name}.yaml",
        )
    )

# --- Phase 4 composites: the baseline moved to lgbm_stride24, so later levers stack on THAT ---
# 4.4 swept 3/3 windows, so measuring anything against lgbm_es now would answer a question about a
# superseded model. Each composite is `stride24 + one lever`, compared against stride24 itself, so
# the delta still isolates a single change.
for _lever, _over, _plan, _why in [
    (
        "unitcat",
        {"categorical_unit": True},
        "4.2c",
        "unit id as a genuine categorical; the tree otherwise sees the 96 units through 3 statics",
    ),
    (
        "norm",
        {"normalise_level": True},
        "4.6",
        "z = y/level with weight=level; isolates the representation gain, both arms WAPE-aligned",
    ),
    (
        "recency",
        {"recency_halflife": 1344},
        "4.5a",
        "8-week decay in origin age; matters MORE at stride 24, which pulls in proportionally "
        "more old data",
    ),
    (
        "wlag",
        {"fc_lags": WEEKLY_LAGS},
        "4.2a",
        "weekly lags were null at stride 168; denser origins give the tree far more chances to "
        "learn the hour-of-week interaction, and the plan flagged the interaction in advance",
    ),
]:
    register(
        MemberSpec(
            name=f"lgbm_s24_{_lever}",
            kind="tree",
            run=_lgbm_runner(
                f"lgbm_s24_{_lever}",
                overrides={
                    "origin_stride": 24,
                    "early_stopping_rounds": 50,
                    "max_boost_round": 3000,
                    **_over,
                },
            ),
            seedable=True,
            needs_gpu=False,
            honours_gap_cov=False,
            status="untested",
            note=f"Plan {_plan} on the stride-24 baseline: {_why}. Compare against lgbm_stride24.",
            tags=("phase4", "composite"),
        )
    )


# --- S2: the second gradient-boosting library, as a BLEND MEMBER ----------------------------------

register(
    MemberSpec(
        name="catboost",
        kind="tree",
        run=_catboost_runner("catboost"),
        seedable=True,
        needs_gpu=False,
        # Same reason as the LightGBM members: `predict_gapped` never traverses the gap. Its lags
        # are origin-anchored and its covariates are read at the forecast hour, so `real` and
        # `impute` produce identical output.
        honours_gap_cov=False,
        status="measured",
        tags=("s2", "member"),
        note=(
            "Plan S2: ordered boosting + native categoricals on the SAME 41-feature basis and the "
            "same stride-24 rolling-origin grid as lgbm_s24_unitcat, so the A/B isolates the "
            "BOOSTER. Screened as a blend member only — the covariate axis is closed to anything "
            "fitted on our data (the standing architectural rule). "
            "MEASURED 2026-07-31, 3 windows, 96,768 rows: solo 0.1586 late / 0.1614 full, between "
            "the two LightGBMs, early stopping converged (761/812 rounds, no ceiling). And on the "
            "axis that decides, err-corr vs lgbm_s24_unitcat is 0.984 — THE HIGHEST PAIR IN THE "
            "MATRIX, above lgbm|lgbm_s24_unitcat's 0.976. A different library, oblivious trees, "
            "ordered boosting, native categoricals: more redundant with LightGBM than LightGBM is "
            "with itself at two settings. Its blend (0.19*catboost + 0.81*tft_cascade -> 0.1331) "
            "loses to 0.18*lgbm + 0.82*tft_cascade (0.1325) by +0.0006 late, CI [+0.0002, "
            "+0.0009], 3/3 windows, and the two BLENDS correlate at 0.9988; in a triple the "
            "simplex gives it weight 0.00. A drop-in substitute, not an addition. "
            "CONSEQUENCE: no further gradient-boosting members. XGBoost was cut on the PRIOR that "
            "another GBDT is cluster-B; this tested that prior on the most favourable evidence "
            "available and confirmed it. The negative result is worth more than the member would "
            "have been — it is the cleanest statement in the two-cluster argument: the shared "
            "training set determines the errors, not the algorithm."
        ),
    )
)

register(
    MemberSpec(
        name="lgbm_cascade",
        kind="tree",
        run=_lgbm_runner(
            "lgbm_cascade",
            overrides={
                "origin_stride": 24,
                "early_stopping_rounds": 50,
                "max_boost_round": 3000,
                "categorical_unit": True,
            },
        ),
        seedable=True,
        needs_gpu=False,
        honours_gap_cov=False,
        # The SAME zero-shot Chronos-2 covariate the TFT cascade eats, on the tuned tree instead.
        # Costs nothing but a registration: `run_member` enters `cascade_channels(*spec.cascade)`
        # before the runner, and `predict_gapped` calls `futr_exog_list()` inside that context, so
        # the channel arrives as a 30th future covariate with no change to the tree's code.
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_cut{cut}.csv",
        status="untested",
        tags=("phase4", "cascade"),
        # Why this is worth a run despite the tree's tuning gain not reaching the blend: the tuned
        # tree got BETTER (0.1658 -> 0.1585) and simultaneously LESS orthogonal to tft_cascade
        # (err-corr 0.782 -> 0.800), and those cancelled to 0.0002. Handing the tree the cascade's
        # own covariate pushes correlation the same way, so the pooled pair could easily get worse
        # — that is the point. It separates "the tree helps because it is a tree" from "the tree
        # helps because it lacks the Chronos prior". Compare on err-corr AND on the repriced pair,
        # not on solo WAPE alone.
        note="Tuned tree (stride24 + unit categorical) conditioned on the zero-shot Chronos-2 "
        "prior — the tree half of plan 7.1/S4. Compare against lgbm_s24_unitcat.",
    )
)

register(
    MemberSpec(
        name="tft_cascade",
        kind="neural",
        run=_nf_runner("tft_cascade", config="configs/tft_chronos.yaml"),
        seedable=True,
        needs_gpu=True,
        # ZERO-SHOT Chronos-2, deliberately — not the LoRA fine-tune. The cascade's job is to hand
        # the TFT a *global prior* its Variable Selection Network can learn when to trust; a
        # per-window fine-tune would make the covariate a second fitted model, and its errors would
        # correlate with `chronos_ft`, which is already an ensemble member (err-corr 0.93 to LGBM).
        # `src.models.chronos2_oof` calls `_load_pipeline(device)` with no adapter — zero-shot by
        # construction. Zero-shot Chronos-2 standalone is 0.190; the fine-tune is 0.169.
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_cut{cut}.csv",
        status="measured",
        note="Chronos-2 zero-shot OOF forecast as a TFT known-future covariate. MEASURED HONESTLY "
        "2026-07-31 on per-window gap-honest frames: pooled late 0.1341 vs plain TFT 0.1463, "
        "CI [+0.0093, +0.0151], 2 of 3 windows (W1 loses), err-corr 0.928 — all three admission "
        "checks pass, and it beats the incumbent convex blend's ~0.1427. The two historical "
        "numbers stay VOID (W0 0.1372, FAR 0.1464): both came from the rolling frame, whose "
        "context ran 336h past the cutoff. The leak was worth only 0.0018 of the W0 gap, so it "
        "flattered the result without creating it (plan 3.10a/3.10b).",
    )
)

# --- IDEAS #16: the selective state-space (Mamba) encoder swap ------------------------------------
#
# `tft_cascade` with its two LSTM encoders replaced by a Mamba stack (src/models/mamba_tft.py) and
# NOTHING else changed: same cascade frame, same backbone hyperparameters, same horizon, same
# gap-fill condition, same 5000-step ceiling.
#
# WHY THE CACHED `tft_cascade` CUBE IS AN HONEST CONTROL HERE, and why that is not true of the
# neighbouring #12 lane. The xvar arms had to run their own control because making the TFT
# multivariate changed the WINDOW SAMPLING, so a delta against the cached cube would have confounded
# the block with the batching. This swap changes neither the batching, the frame, nor the sampler:
# `MambaTFT` is univariate, its VSNs and gated skip are the objects the parent TFT already built,
# and `ssm_mode="lstm"` is pinned bit-for-bit to neuralforecast's own TFT in
# tests/test_mamba_tft.py. What remains is cross-run training stochasticity, measured at ~0.001 in
# S2 Stage 2 — a fifth of the gate below, which is why it is a caveat and not a control.
register(
    MemberSpec(
        name="tft_cascade_mamba",
        kind="neural",
        run=_nf_runner("tft_cascade_mamba", config="configs/tft_chronos_mamba.yaml"),
        seedable=True,
        needs_gpu=True,
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_cut{cut}.csv",
        status="measured",
        note="IDEAS #16. MEASURED 2026-09-03 and it LOSES: 0.16510 vs the cascade's 0.13901 on "
        "W0's 32,256 scored rows, -0.02610, CI [-0.02891, -0.02323], 17.9 SE, and -0.0248 in "
        "the near regime too. Early-stopped at ~849 of 5000 steps, so it converged rather than "
        "starved. Do not re-run without a NEW reason: the lane is closed at stage 1 — see "
        "docs/method.md (W5). Selective state-space (Mamba) encoders in place of the "
        "TFT's two LSTMs, "
        "parameter-matched at 98% of the LSTM's count so a win cannot be bought with capacity. "
        "Stop rule fixed before the first fit (docs/method.md, W5): stage 1 is "
        "window 0 at one seed against the cached tft_cascade cube, and it escalates to 3 windows "
        "only if it beats it by more than the backbone's cross-seed sigma 0.0055. It is a "
        "REPLACEMENT for casc_bag5, never an addition — its errors are cascade errors.",
    )
)

# --- Final push lane 1G: the residual-target cascade ---------------------------------------------
#
# The cascade's Chronos-2 channel, moved from the INPUT side to the TARGET side. `tft_cascade`
# feeds the frozen forecast to the VSN as a known-future covariate and lets the network learn when
# to trust it; this member subtracts it instead — trains on `y - chronos2_forecast`, adds the
# channel back at inference. Same frame, same backbone, same horizon, same hyperparameters: the
# only difference between the two registrations is one config key.
#
# It is a REPLACEMENT for `casc_bag5` in the blend, never an addition. Its errors are cascade
# errors; a pool holding both would be fitting two views of one model.
#
# The trap that had to be written down before the first fit (S10 law 5): `add_volume_sample_weight`
# derives its weights from the MAD of NF_TARGET so that neuralforecast's robust scaler cancels and
# plain MAE becomes the WAPE numerator. Replace the target before computing them and that MAD is
# the RESIDUAL's — the WAPE alignment breaks with no error message and the lane returns a confident
# null. `nf_fit_and_sweep` computes the weights first and then ASSERTS they are not the residual's.
# (The shipped arm, `elu_plain`, carries `sample_weight: none`, so the assertion guards the arm we
# do not currently run — which is exactly when a silent defect survives.)
register(
    MemberSpec(
        name="tft_cascade_resid",
        kind="neural",
        run=_nf_runner("tft_cascade_resid", config="configs/tft_chronos_resid.yaml"),
        seedable=True,
        needs_gpu=True,
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_cut{cut}.csv",
        status="untested",
        note="Lane 1G. Residual-target cascade: trains on y - chronos2_forecast and adds the "
        "channel back at inference, instead of gating it as a covariate. Stop rule fixed before "
        "the first fit: 1 seed x 3 windows, gap 336, no near fold; the gain must exceed the "
        "backbone's cross-seed sigma ~0.0055 AND win 3/3, or the lane closes with no second seed.",
    )
)

# --- S5 Tier D, finally run: the checkpoint-averaged cascade -------------------------------------
#
# S5 promised three post-hoc ensembling levers (SWA, seed-bagging, MC-dropout) and spawned none of
# them: gate 1 failed and Tier D was skipped by its own stop rule. Seed-bagging was paid for later
# anyway — `casc_bag5` averages 5 seeds — but checkpoint averaging has never been measured on any
# model in this project.
#
# WHY IT IS NOT THE SAME LEVER AS THE BAG, which is the whole reason it is worth a container.
# Seed-bagging averages FORECASTS from models that landed in different basins; checkpoint averaging
# averages WEIGHTS along one trajectory inside a single basin. Different variance components, and
# the second one is untouched by the first. If it pays, it stacks.
#
# THE CONTROL IS `casc_s892`, NOT `casc_bag5`, and that is not a detail. Comparing one averaged fit
# against a 5-seed bag would charge this arm for the bag's variance reduction and then report the
# difference as a mechanism — the exact error lane 1G was careful to avoid.
#
# THE TRAP THAT HAD TO BE WRITTEN DOWN FIRST (law 5). Lightning's own StochasticWeightAveraging is
# epoch-based and cannot initialise under neuralforecast's step-based trainer: `on_fit_start`
# asserts `trainer.max_epochs is not None`, `pl.Trainer(max_steps=...).max_epochs` IS None, and
# neuralforecast raises on the `max_epochs` kwarg that would fix it. Using it would have averaged
# nothing and returned a confident null. `src/models/swa.py` is the step-based replacement and
# `tests/test_swa.py` asserts the weights actually move — not that the callback ran.
register(
    MemberSpec(
        name="tft_cascade_swa",
        kind="neural",
        run=_nf_runner("tft_cascade_swa", config="configs/tft_chronos_swa.yaml"),
        seedable=True,
        needs_gpu=True,
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_cut{cut}.csv",
        status="untested",
        note="S5 Tier D. Checkpoint averaging (SWA): a rolling window of the last 5 parameter "
        "snapshots taken every 50 optimizer steps, averaged into the model at train end. Stop "
        "rule fixed before the first fit: 1 seed (892) x 3 windows, gap 336, no near fold, "
        "scored against casc_s892 (the SAME seed's stored cube, never the bag). The gain must "
        "exceed the backbone's cross-seed sigma ~0.0055 AND win 3/3, or the lane closes with no "
        "second seed. It is a REPLACEMENT for casc_bag5, never an addition - its errors are "
        "cascade errors.",
    )
)

# SPRINT 2 arm 2C-control. The SAME SWA backbone with the CHRONOS COVARIATE REMOVED (`cascade=()`,
# which also switches the training frame back to data/raw/train.csv). The question is a substitution
# one: `chronos2_forecast` is worth +8.3% and is the most expensive covariate we carry — a whole
# foundation model's rolling-origin forecast must exist before the TFT can train at all — while A9's
# capacity-weighted system mean is a different, nearly free route to "what is the rest of the panel
# doing". Paired against `tft_cascade_swa` + A9 at the SAME seed, the difference is the value
# chronos still adds ON TOP of the aggregates. One seed (892) by design: this reads a
# sign, it is not a candidate for admission.
register(
    MemberSpec(
        name="tft_swa",
        kind="neural",
        run=_nf_runner("tft_swa", config="configs/tft_swa.yaml"),
        seedable=True,
        needs_gpu=True,
        cascade=(),
        status="untested",
        note="Sprint 2 arm 2C-control. tft_cascade_swa with cascade=() — no chronos2_forecast "
        "covariate and no derived training frame. Run ONLY with --xs-block A9, paired against "
        "tft_cascade_swa+A9 at seed 892, to measure whether the Chronos covariate still earns "
        "its cost once cross-series aggregates are present.",
    )
)


# SPRINT 2. The scaler A/B — see configs/tft_chronos_swa_noscale.yaml. Paired against
# `tft_cascade_swa` + A9 at seed 892; `local_scaler_type: null` is the only difference.
register(
    MemberSpec(
        name="tft_cascade_swa_noscale",
        kind="neural",
        run=_nf_runner("tft_cascade_swa_noscale", config="configs/tft_chronos_swa_noscale.yaml"),
        seedable=True,
        needs_gpu=True,
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_cut{cut}.csv",
        status="untested",
        note="Sprint 2. Does the SUBMISSION path's missing per-series scaler help or hurt? Raw L1 "
        "minimises the pooled-WAPE numerator exactly; robust scaling weights series by 1/MAD and "
        "is misaligned by the 2.23x spread, but divides out the level the model would else learn.",
    )
)


# SPRINT 2. THE MATCHED-STEPS ISOLATION. The first scaler A/B moved TWO things: the unscaled arm
# early-stopped at ~800 steps against the control's 5000, because a raw-unit `valid_loss` is noisier
# and the same patience fires far sooner. So +0.06500 is scaler AND undertraining, in unknown
# proportion. These two members set `early_stop_patience_steps: -1` on BOTH sides -- the control's
# own stopping point was never verified either, so re-buying it is what makes the pair matched
# rather than assuming it ran the full ceiling.
for _nm, _cfg, _note in (
    (
        "tft_cascade_swa_full",
        "configs/tft_chronos_swa_full.yaml",
        "Sprint 2. Robust scaler, early stopping DISABLED -- the matched control for "
        "`tft_cascade_swa_noscale_full`. Not a ship candidate.",
    ),
    (
        "tft_cascade_swa_noscale_full",
        "configs/tft_chronos_swa_noscale_full.yaml",
        "Sprint 2. No scaler, early stopping DISABLED. Against `tft_cascade_swa_full` this "
        "isolates `local_scaler_type` from the stopping rule the first A/B confounded it with.",
    ),
):
    register(
        MemberSpec(
            name=_nm,
            kind="neural",
            run=_nf_runner(_nm, config=_cfg),
            seedable=True,
            needs_gpu=True,
            cascade=("chronos2_forecast",),
            train_csv="data/derived/train_chronos_cut{cut}.csv",
            status="untested",
            note=_note,
        )
    )


# --- S2: the four zero-shot foundation candidates -------------------------------------------------
#
# Three registrations per model, because S2 asks two different questions and the second has two
# modes. All three read the SAME artifact — one 672-step forward pass anchored at the cutoff — which
# is what makes the member axis a free by-product of the covariate screen rather than a second run.
#
#   <X>_zeroshot            the MEMBER. CPU: a filter over the derived frame, not a new forecast.
#   tft_cascade_<X>         INSTEAD OF — is X a BETTER prior than Chronos-2? Preferred when
#                           corr(X, chronos2_forecast) is HIGH: same information, possibly better
#                           rendered.
#   tft_cascade_chronos_<X> ALONGSIDE — do TWO priors beat one? Preferred when that correlation is
#                           LOW: genuinely different information, and the VSN gates each channel
#                           separately.
#
# Stage 1 hands over that correlation for free, so it SELECTS the mode rather than us guessing.
# Order is ALONGSIDE first, INSTEAD-OF as the ablation: alongside is the mode that can raise the
# ceiling, and if X is simply the better prior the VSN can down-gate chronos2_forecast by itself.
# Not airtight — a second channel adds input noise, so a swap can win where alongside loses.
#
# `status="untested"` throughout. Nothing here has been run, and `available_members(status=
# "measured")` excludes them so a sweep cannot pick one up as if it had a number.
#
# The `_cov` entries are the SAME models fed the control's own covariates. They exist because the
# first screen was confounded: `chronos2_oof` hands Chronos-2 all 29 known-future covariates
# `base_exog()`, while `foundation.run_one` sliced the candidates to [unique_id, ds, y] — so four
# covariate-blind models were ranked against a covariate-fed control and called "far below" it.
# That measured covariate access, not model quality. `tirex` has no `_cov` twin: v1
# exposes no covariate argument, and v2 (which takes `future_covariates`) truncates 672 to 320.
FOUNDATION_CANDIDATES = (
    "toto",
    "timesfm",
    "tabpfn_ts",
    "tirex",
    "toto_cov",
    "tabpfn_ts_cov",
    "timesfm_cov",
)

_FOUNDATION_WHY = {
    "toto_cov": (
        "Toto 2.0 with the 29 known-future covariates via `known_dynamic`. The version audit "
        "recorded 'no exogenous support yet' off the upstream README and the installed package "
        "has it — the fourth time a doc has been the least reliable thing in the room"
    ),
    "tabpfn_ts_cov": (
        "TabPFN-TS with the 29 known-future covariates — the control's own footing. The best fit "
        "of the four: a tabular regressor takes a covariate as another column, so no architectural "
        "channel has to exist for it"
    ),
    "timesfm_cov": (
        "TimesFM with `forecast_with_covariates`. NOTE `xreg_mode='xreg + timesfm'` fits a RIDGE "
        "on the covariates and adds it to the base forecast rather than attending over them, so a "
        "gain here is 'TimesFM plus a linear covariate model' — a weaker claim than TabPFN's"
    ),
    "toto": (
        "Datadog, 313M, pretrained on OBSERVABILITY TELEMETRY — the closest public analogue to an "
        "'operational load index' that exists, and the strongest a-priori reason to expect a prior "
        "Chronos-2 does not already carry"
    ),
    "timesfm": (
        "Google, ~200M, pretrained on Trends / Wikipedia / synthetic — a corpus with nothing in "
        "common with this one. If an unrelated-domain prior still helps, the cascade's gain is "
        "about the MECHANISM (a global prior a VSN can gate) rather than about corpus match"
    ),
    "tabpfn_ts": (
        "TabPFN v2 as a tabular regressor over time features — not a sequence model at all, so the "
        "most structurally different candidate available, and the only one that could eventually "
        "eat the 13 planning signals (the obvious S3 follow-up)"
    ),
    "tirex": (
        "NX-AI, 35M, xLSTM — recurrent rather than attention, and by far the smallest to ship. "
        "Highest install risk: gated HF repo (token via a MODAL SECRET, never a repo file) and "
        "sLSTM CUDA kernels with a slow fallback. One build attempt, then drop and record"
    ),
}


def _zeroshot_member_runner(name: str, column: str):
    """The MEMBER arm: slice the scored block out of the derived frame. No model call at all.

    The forward pass already happened when the covariate frame was generated, and its last 336 steps
    *are* the member's prediction — so this is CPU, seconds, and needs no image with the model in
    it. The same economy ``chronos2_zeroshot`` gets: a filter over the frames, not a new run.

    Deterministic, hence ``seedable=False``: a mean +- std over five identical slices would claim a
    precision that does not exist.
    """

    def run(ctx: RunContext) -> pd.DataFrame:
        df = add_hour_index(ctx.long_df)
        horizon = gapped_horizon(df, ctx.cut_idx).drop(columns=HOUR_IDX)
        scored = take_block(horizon, ctx.regime)
        missing = [c for c in (NF_TARGET, column) if c not in scored.columns]
        assert not missing, f"{name}: {missing} absent from the derived frame"
        return scored[[NF_ID, NF_TIME, NF_TARGET, column]].rename(columns={column: name})

    run.member_config = None
    run.member_overrides = {}
    run.member_code = ("src/models/foundation.py",)
    run.supports_regime = True  # a slice of an existing frame — both halves are already there
    return run


# S3's missing cell, made measurable. `interp` on the scattered surface gives the tree +0.00401
# and PLAIN TFT +0.00532 (both 3/3), but the cascade came back -0.00912 — and that run was
# CONFOUNDED, not a result: `chronos2_oof` imputes with the median, so the cascade channel encoded
# a forecast built from median-imputed covariates while the TFT trained on interp-imputed ones. The
# model saw two inconsistent views of the same signals, and the VSN weights that channel heavily.
#
# This member is the honest version: same architecture, same everything, but its covariate frame is
# regenerated under the SAME imputation the member trains under. Registered as a twin rather than
# an override so the median frame — and therefore the 0.13429 control — survives untouched.
register(
    MemberSpec(
        name="tft_cascade_interp",
        kind="neural",
        run=_nf_runner("tft_cascade_interp", config="configs/tft_chronos.yaml"),
        seedable=True,
        needs_gpu=True,
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_interp_cut{cut}.csv",
        status="untested",
        tags=("s3", "cascade", "imputation"),
        note="Plan S3 missing cell. `tft_cascade` with BOTH the training covariates and the "
        "Chronos-2 channel imputed by `interp`, so the covariate's provenance matches the data "
        "the member trains on. Compared against `tft_cascade` at 0.13429 late.",
    )
)


# S4 / plan 7.2 — the matched-distribution ablation, and the LAST open caveat on our best model.
#
# `tft_cascade` TRAINS on a covariate rolled in 336h blocks, so every training row sees a Chronos-2
# forecast at lead 1-336. At inference the scored block sits at lead 337-672 from the same anchor.
# The VSN therefore calibrates how far to trust that channel on a quality it never meets at test
# time. This twin trains on the 672-block grid instead, where the training rows span the same lead
# range the horizon block does.
#
# The two grids are otherwise identical BY ARITHMETIC, which is what makes this a clean ablation
# rather than two different experiments: 6*672 == 12*336 == 4032, so both start at hour 288, leave
# the same NaN warm-up (hence a bit-identical `chronos2_forecast_missing` pattern), and spend the
# same number of Chronos steps. Only the lead profile moves.
#
# S4.0's free screen predicts a NULL, and it is on the record before the run: period-matched, the
# covariate degrades only 0.1786 -> 0.1892 (+5.9%) when the lead doubles, and the two renderings
# correlate at 0.978-0.986. It runs anyway because a screen ranks covariate ACCURACY while the gate
# reads the VSN's learned TRUST — S3 is the worked example of those being different quantities.
register(
    MemberSpec(
        name="tft_cascade_matched",
        kind="neural",
        run=_nf_runner("tft_cascade_matched", config="configs/tft_chronos.yaml"),
        seedable=True,
        needs_gpu=True,
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_h672_cut{cut}.csv",
        status="untested",
        tags=("s4", "cascade", "matched"),
        note="Plan S4 / 7.2. `tft_cascade` trained on a Chronos-2 covariate rolled in 672h blocks "
        "instead of 336h ones, so the training lead distribution matches the 337-672 the scored "
        "block sits at. Same warm-up, same step count, same everything else. Compared against "
        "`tft_cascade` at 0.13429 late, re-run as a control in the SAME batch — S2 Stage 2's "
        "cross-environment drift was the size of its own best effect, and the predicted effect "
        "here is zero.",
    )
)


# S3.3 — Chronos parity. `chronos_ft` (the LoRA fine-tune) has been on the cube since 3.5; its
# ZERO-SHOT twin never was, so 0.190-vs-0.169 has always been two numbers from two harnesses rather
# than two members on one set of rows. This closes that, and it costs nothing: the gap-honest
# 672-step Chronos frames already exist per window, and the member is the last 336 steps of the
# block that produced `tft_cascade`'s covariate. A filter over frames we hold, not a new run.
#
# It is exactly the same shape as the four S2 foundation members, so it reuses their runner rather
# than getting its own -- which also means it inherits the provenance fence: `run_member` verifies
# the frame is gap-honest at this cutoff before scoring, because a forecast whose context ran past
# the cutoff scores BETTER as a member too.
register(
    MemberSpec(
        name="chronos2_zeroshot",
        kind="foundation",
        run=_zeroshot_member_runner("chronos2_zeroshot", "chronos2_forecast"),
        seedable=False,  # one deterministic forward pass, already made
        needs_gpu=False,  # the GPU was spent generating the frame; this only reads it
        honours_gap_cov=False,  # its forecast is fixed in the frame; the gap fill cannot move it
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_cut{cut}.csv",
        status="untested",
        tags=("s3", "zeroshot", "member", "parity"),
        note="Plan S3.3. The zero-shot twin of `chronos_ft`, on the SAME cube and the same rows, "
        "so the exposé's 'zero-shot ~0.169, fine-tuned 0.160' can be replaced with a measured "
        "pair. 3.8 established 0.169 IS the fine-tuned score and zero-shot is 0.190, so the "
        "exposé compares two fine-tuned numbers and understates its own headroom.",
    )
)


for _fm in FOUNDATION_CANDIDATES:
    _col = f"{_fm}_forecast"
    _single = f"data/derived/train_{_fm}_cut{{cut}}.csv"
    _both = f"data/derived/train_chronos_{_fm}_cut{{cut}}.csv"

    register(
        MemberSpec(
            name=f"{_fm}_zeroshot",
            kind="foundation",
            run=_zeroshot_member_runner(f"{_fm}_zeroshot", _col),
            seedable=False,  # one deterministic forward pass, already made
            needs_gpu=False,  # the GPU was spent generating the frame; this only reads it
            honours_gap_cov=False,  # target-only: it never sees a planning covariate, gap or not
            # Declared even though this member does not *condition* on the channel — it IS the
            # channel. The declaration is what routes it to the derived frame AND what makes
            # `run_member` verify the provenance before scoring: a forecast whose context ran past
            # the cutoff scores BETTER as a member too, so the member axis needs the same fence.
            cascade=(_col,),
            train_csv=_single,
            status="untested",
            tags=("s2", "zeroshot", "member"),
            note=f"Plan S2 member arm. {_FOUNDATION_WHY[_fm]}. Free by-product of the covariate "
            "screen: the last 336 steps of the same 672-step block. Reported as evidence about the "
            "two-cluster hypothesis, not as a decision S2 spends anything on — S1 proved solo WAPE "
            "is the wrong axis.",
        )
    )

    register(
        MemberSpec(
            name=f"tft_cascade_{_fm}",
            kind="neural",
            # Same TFT hyperparameters as the incumbent cascade, deliberately: the comparison is
            # about the PRIOR, so anything else changing would confound it. windows_batch_size 32 is
            # the memory ceiling at h=672, and one extra scalar covariate does not move it.
            run=_nf_runner(f"tft_cascade_{_fm}", config="configs/tft_chronos.yaml"),
            seedable=True,
            needs_gpu=True,
            cascade=(_col,),
            train_csv=_single,
            status="untested",
            tags=("s2", "cascade", "instead-of"),
            note=f"Plan S2 Stage 2, INSTEAD-OF mode: {_fm} replaces Chronos-2 as the TFT's prior. "
            "Asks whether it is a BETTER prior; preferred when corr(X, chronos2_forecast) is HIGH. "
            "GATE: beat tft_cascade 0.13409 late, paired over 3 windows, CI excluding zero, then "
            "reprice the +lgbm pair against 0.1325.",
        )
    )

    register(
        MemberSpec(
            name=f"tft_cascade_chronos_{_fm}",
            kind="neural",
            run=_nf_runner(f"tft_cascade_chronos_{_fm}", config="configs/tft_chronos.yaml"),
            seedable=True,
            needs_gpu=True,
            # Order matters only for readability; `cascade_channels` re-sorts against
            # CASCADE_FORECASTS_ALL so the futr list is stable whatever order a spec declares.
            cascade=("chronos2_forecast", _col),
            train_csv=_both,
            status="untested",
            tags=("s2", "cascade", "alongside"),
            note=f"Plan S2 Stage 2, ALONGSIDE mode: BOTH priors, so the VSN gates each separately. "
            f"Asks whether two priors beat one; preferred when corr({_fm}, chronos2_forecast) is "
            "LOW. Run FIRST — it is the mode that can raise the ceiling, and if the new prior is "
            "simply better the VSN can down-gate chronos2_forecast on its own. Needs the merged "
            "two-channel frame (src.models.foundation.merge_frames) and a two-column sidecar; "
            "run_member checks each channel's grid separately.",
        )
    )

del _fm, _col, _single, _both


register(
    MemberSpec(
        name="chronos_ft",
        kind="foundation",
        run=_chronos_ft_runner(),
        seedable=False,  # inference-only against a fixed LoRA adapter
        needs_gpu=True,
        needs_adapter=True,
        note="Chronos-2 + per-window LoRA adapter. Zero-shot is 0.190; this fine-tune is 0.169.",
    )
)


# ----------------------------------------------------------------------------- the 2026-08-17 sweep
# Two members that did not exist when the sprint closed, both added to answer a specific question
# rather than to re-open modelling:
#
#   bitcn_wide      — "the TCN seems to be underperforming": it was never tuned, it ran at
#                     neuralforecast's DEFAULTS (hidden_size 16, dropout 0.5). This is the same
#                     architecture at the shipped TFT's capacity, so 0.18050 can be read as a
#                     ceiling rather than as a default.
#   chronos_full_ft — "i suspect we only did LoRA and no full finetune". Confirmed: no full
#                     fine-tune exists anywhere in this project. The flag has been there since
#                     S3.4 and was never run.
#
# Both are registrations over existing runners; neither needs new modelling code.

register(
    MemberSpec(
        name="bitcn_wide",
        kind="neural",
        run=_nf_runner("bitcn_wide"),
        seedable=True,
        needs_gpu=True,
        status="untested",
        tags=("sweep-2026-08-17", "capacity"),
        note="BiTCN at the shipped TFT's capacity (hidden_size 64, dropout 0.1) instead of "
        "neuralforecast's library defaults (16 / 0.5), which is what every recorded BiTCN number "
        "was fitted at. Compare against `bitcn` (0.18050 late): the A/B isolates model size, since "
        "nothing else differs. Its err-corr vs the tuned tree is 0.938, so a better BiTCN competes "
        "for LightGBM's blend slot rather than opening a third cluster — read the fitted WEIGHT, "
        "not the solo WAPE (plan 4.7).",
    )
)

register(
    MemberSpec(
        name="chronos_full_ft",
        kind="foundation",
        run=_chronos_ft_runner("chronos_full_ft"),
        seedable=False,  # inference-only against a fixed checkpoint
        needs_gpu=True,
        needs_adapter=True,
        status="untested",
        tags=("sweep-2026-08-17", "s3.4-resumed"),
        note="Chronos-2 with EVERY parameter fine-tuned (configs/chronos2_fullft.yaml), against "
        "`chronos_ft`'s LoRA r=8. The one S3.4 technique postponed rather than retired, because it "
        "is the only one that buys a measured point instead of a predicted one. Prior is still "
        "cluster-B: `chronos_ft` sits at err-corr 0.941 against its own ZERO-SHOT twin, so full FT "
        "moving *where the errors live* would be the surprise, not moving its solo WAPE.",
    )
)


# --- The 2026-08-17 capacity probe, rung 2: is the width curve still climbing? ---
register(
    MemberSpec(
        name="bitcn_wider",
        kind="neural",
        run=_nf_runner("bitcn_wider"),
        seedable=True,
        needs_gpu=True,
        status="untested",
        tags=("sweep-2026-08-17", "capacity"),
        note="BiTCN at hidden_size 128. 16 -> 64 bought +9.4% relative; this asks whether "
        "64 -> 128 "
        "still climbs. Note bitcn_wide took ZERO blend weight at every combination size while its "
        "err-corr vs the tree ROSE 0.924 -> 0.951, so a better BiTCN is expected to be a write-up "
        "result, not a member.",
    )
)

register(
    MemberSpec(
        name="tft_cascade_w128",
        kind="neural",
        run=_nf_runner("tft_cascade_w128", config="configs/tft_chronos_w128.yaml"),
        seedable=True,
        needs_gpu=True,
        cascade=("chronos2_forecast",),
        train_csv="data/derived/train_chronos_cut{cut}.csv",
        status="untested",
        tags=("sweep-2026-08-17", "capacity"),
        note="The SHIPPED cascade backbone at hidden_size 128 instead of 64. S5's sweep gave width "
        "128 exactly ONE trial (0.16219 vs the incumbent's 0.16249 on the search window, single "
        "seed, against cross-seed sigma 0.0055), so wider was never actually tested. Compare "
        "against `tft_cascade` = 0.13409 late — and read it against that sigma, not against the "
        "third decimal.",
    )
)
