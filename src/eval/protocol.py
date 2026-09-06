"""The single evaluation protocol. Every experiment imports from here, so none invents its own.

Why this module exists
----------------------
Re-verifying the recorded numbers in a clean clone found **three different row-sets** all called
"the score", and **two admission gates that disagree**. Both problems come from every script
carrying its own copy of the rules.

The three row-sets, all real and all in use (numbers are pooled over the three CV windows):

    regime="full"  full 336h scored block, 96,768 rows   TFT 0.1489   LGBM 0.1658
    regime="late"  blk >= 224, the final 112h,  32,256    TFT 0.1463   LGBM 0.1628

Everything quoted as a headline is ``late``. That row-set is *correct for blends* — blend weights
are fitted on ``blk < 224``, so honest scoring must hold out ``blk >= 224`` — but it means the
**standalone** figures quoted next to those blends rest on one third of the data, and
``sigma_tft = 0.0076`` was estimated on that same noisy subset. The word "FAR" makes it worse: in
one early scoring script it meant steps 337-672 of the 672h horizon (the whole scored
block), while in ``scripts/member_admission.py`` it means ``blk >= 224`` *within* that block.

So ``regime`` is a **required argument** here. There is deliberately no default: picking one
silently is exactly how the record ended up ambiguous.

The pooling itself reuses the accumulate-then-divide pattern already proven in
``scripts.ensemble_metalearner._wstats`` and ``scripts.ensemble_architecture_test._pool``: sum the
numerators and denominators across windows *before* dividing. Pooled WAPE is not the mean of
per-window WAPEs (``tests/test_metrics.py`` pins the difference).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import SCORE_LEN
from src.metrics import all_metrics

# The held-out boundary inside the scored block. Rows with blk < CUT_BLK are where blend weights
# and stacker parameters are fitted; blk >= CUT_BLK is the honest scoring region. Mirrors
# scripts.ensemble_architecture_test.CUT_BLK and scripts.ensemble_metalearner.router_panel.
CUT_BLK = 224
BLK = "blk"
CUTOFF = "cutoff"

Regime = Literal["full", "late"]
REGIMES: tuple[str, ...] = ("full", "late")

# Soft-check defaults. err-corr above this means the candidate is redundant with what we have;
# today's member pool sits at 0.77-0.96.
ERR_CORR_THRESHOLD = 0.95

# A candidate must improve in at least this fraction of the CV windows — 2 of 3. Guards the one
# specific pathology a pooled number cannot see: a huge win in one window carrying a candidate that
# is *worse* in the other two. It is not a significance test (P(>=2 of 3) = 0.5 under a coin-flip
# null); it removes a failure mode. This restores the "<= 1 window lost" clause from the original
# scripts.ensemble_architecture_test.decide(), which was correct and got lost.
MIN_WINDOW_FRACTION = 2 / 3

# Bootstrap resamples for the paired SE. 2000 is ample for a standard error; the CI tails would
# want more, which is why n_boot is a parameter.
N_BOOTSTRAP = 2000

# --- horizon halves -----------------------------------------------------------------------
# A THIRD axis, orthogonal to `regime`, and the two must not be confused. `regime` slices the
# *scored block* by position within it (full 336h vs the final 112h). `horizon` slices the whole
# 672h forecast into its two halves:
#
#     near  steps 1..336    forecast starts +1h after the last observed target (the VALIDATION
#                           scenario) — this is the leaderboard's near window
#     far   steps 337..672  a 336h covariate-absent gap sits first (the PRIVATE TEST scenario)
#
# Only `far` corresponds to what we are graded on. Issue #55 requires both to be reported, because
# a model can win one and lose the other — and the cascade does exactly that (docs/protocol.md
# and results/member_metrics_near_far.json). Requires a `step` column, which
# the full-horizon prediction cubes carry and the scored-block-only frames do not.
STEP = "step"
NEAR_FAR_SPLIT = SCORE_LEN  # 336

Horizon = Literal["near", "far", "all"]
HORIZONS: tuple[str, ...] = ("near", "far", "all")


# --------------------------------------------------------------------------- seed policy
#
# The policy (README.md, docs/protocol.md) is deliberately *asymmetric* and this module
# encodes that asymmetry rather than flattening it:
#
#   * **Tuning / ablation / every A/B: single seed 42.** Multi-seeding each comparison would cost
#     5x for no decision value — a paired A/B on a shared seed already controls seed variance,
#     because both arms carry the same draw.
#   * **Finalists only: the frozen five, mean +- std.** This is the number that goes in the report
#     and it is the only place the 5x is worth paying.
#
# What the policy does *not* give you is the size of seed noise, and that gap matters: the
# admission bar in ``evaluate_admission`` subtracts a sigma estimated across *windows*
# (``window_sigma``). If cross-seed dispersion is larger, the bar is too loose and single-seed A/B
# deltas below it are unreadable. Measure it once per model family, not per experiment.

TUNING_SEED = 42

# Drawn once, reproducibly. The derivation IS the definition, so this tuple is in **draw order**.
# README.md lists the same five sorted ascending, which is harmless for "run all five,
# report mean +- std" but not for the documented compute-tight fallback: "the first 3 of the set"
# is a prefix only in draw order (drawn -> 892, 7739, 6545; sorted -> 892, 4330, 4388 — a different
# subset). ``frozen_seeds(n)`` is therefore the only sanctioned way to take fewer than five.
FROZEN_SEEDS: tuple[int, ...] = (892, 7739, 6545, 4388, 4330)
FROZEN_SEED_DERIVATION = "np.random.default_rng(42).integers(0, 10000, size=5)"


def verify_frozen_seeds() -> tuple[int, ...]:
    """Re-derive ``FROZEN_SEEDS`` from its documented RNG call and assert it still matches.

    Cheap tripwire against a numpy change or an edit to the tuple silently redefining "the frozen
    set" after numbers have been reported against it.
    """
    derived = tuple(int(s) for s in np.random.default_rng(TUNING_SEED).integers(0, 10000, size=5))
    assert derived == FROZEN_SEEDS, (
        f"frozen seed set no longer reproduces from `{FROZEN_SEED_DERIVATION}`: "
        f"derived {derived}, declared {FROZEN_SEEDS}"
    )
    return derived


def frozen_seeds(n: int | None = None) -> tuple[int, ...]:
    """The first ``n`` frozen seeds in draw order (all five by default).

    Taking a prefix is the sanctioned way to run fewer than five under a compute budget. Any other
    subset is seed selection, which is what the frozen set exists to prevent.
    """
    if n is None:
        return FROZEN_SEEDS
    if not 1 <= n <= len(FROZEN_SEEDS):
        raise ValueError(f"n must be in 1..{len(FROZEN_SEEDS)}, got {n}")
    return FROZEN_SEEDS[:n]


def summarize_seed_runs(scores: Mapping[int, float], *, metric: str = "wape") -> dict:
    """Aggregate one config's per-seed scores into the reportable ``mean +- std``.

    ``scores`` maps seed -> score. The seeds must be exactly ``frozen_seeds(len(scores))`` — a
    prefix of the draw order. Anything else raises: a re-drawn seed, or a hand-picked subset, is
    precisely the failure this guards, and a silent aggregate would launder it into the report.

    ``std`` is the sample std (ddof=1) — the estimator you quote as "+- std" for five runs.
    ``std_pop`` is the population std, reported alongside only so it can be compared like-for-like
    against ``window_sigma``, which uses ddof=0. They are different estimators of different things;
    never quote one as the other.
    """
    if not scores:
        raise ValueError("no seed scores given")
    expected = frozen_seeds(len(scores))
    if set(scores) != set(expected):
        raise ValueError(
            f"seeds {sorted(scores)} are not the frozen prefix {list(expected)}. Tune on "
            f"{TUNING_SEED}; confirm finalists on frozen_seeds(n) — never re-draw, never pick."
        )

    seeds = expected  # report in draw order, not sorted, so the prefix rule stays visible
    vals = np.array([float(scores[s]) for s in seeds], dtype=float)
    n = len(vals)
    std = float(vals.std(ddof=1)) if n > 1 else float("nan")
    complete = n == len(FROZEN_SEEDS)
    return {
        "metric": metric,
        "n_seeds": n,
        "seeds": list(seeds),
        "per_seed": {int(s): float(scores[s]) for s in seeds},
        "mean": float(vals.mean()),
        "std": std,
        "std_pop": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "spread": float(vals.max() - vals.min()),
        "complete": complete,
        "headline": f"{vals.mean():.4f} +- {std:.4f} ({metric}, n={n} seeds)",
        "caveat": (
            ""
            if complete
            else f"PARTIAL: {n}/{len(FROZEN_SEEDS)} frozen seeds. Report with an explicit caveat."
        ),
    }


# --------------------------------------------------------------------------- row selection


def add_block_index(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``blk`` — the 0-based hour index within each series' scored block, per cutoff.

    Grouping by ``cutoff`` as well as ``unique_id`` matters: a frame holding several windows would
    otherwise get one running counter per series spanning all of them, so ``blk >= 224`` would
    select the tail of the *concatenation* rather than the tail of each window's own block.
    """
    df = df.sort_values([c for c in (CUTOFF, NF_ID, NF_TIME) if c in df.columns]).reset_index(
        drop=True
    )
    keys = [CUTOFF, NF_ID] if CUTOFF in df.columns else [NF_ID]
    df[BLK] = df.groupby(keys).cumcount()
    return df


def select_regime(df: pd.DataFrame, regime: Regime) -> pd.DataFrame:
    """Restrict to the rows a regime scores. ``regime`` is required — see the module docstring."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    if regime == "full":
        return df
    if BLK not in df.columns:
        df = add_block_index(df)
    return df[df[BLK] >= CUT_BLK]


# --------------------------------------------------------------------------- pooled WAPE


def wape_stats(y, yhat) -> dict[str, float]:
    """WAPE plus the numerator and denominator, so callers can pool exactly across windows."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ae = float(np.abs(y - yhat).sum())
    ya = float(np.abs(y).sum())
    return {"wape": (ae / ya) if ya > 0 else float("nan"), "ae": ae, "ya": ya}


def compute_pooled_wape(
    df_all_windows: pd.DataFrame,
    member: str,
    regime: Regime,
    *,
    target: str = NF_TARGET,
) -> dict:
    """Pooled WAPE for one member across every window in ``df_all_windows``.

    Accumulates ``sum|y-yhat|`` and ``sum|y|`` over all windows and divides **once**. Returns the
    pooled figure, the per-window breakdown, and the raw numerator/denominator so callers can pool
    further without re-reading the frame.

    ``df_all_windows`` must carry a ``cutoff`` column identifying each window; without it the whole
    frame is treated as a single window and ``per_window`` is keyed ``-1``.
    """
    scored = select_regime(df_all_windows, regime)
    if member not in scored.columns:
        raise KeyError(f"member column {member!r} not in frame; have {sorted(scored.columns)}")

    groups = scored.groupby(CUTOFF) if CUTOFF in scored.columns else [(-1, scored)]
    per_window, ae_total, ya_total = {}, 0.0, 0.0
    for cut, g in groups:
        st = wape_stats(g[target], g[member])
        per_window[int(cut)] = st["wape"]
        ae_total += st["ae"]
        ya_total += st["ya"]

    return {
        "member": member,
        "regime": regime,
        "pooled_wape": (ae_total / ya_total) if ya_total > 0 else float("nan"),
        "per_window_wape": dict(sorted(per_window.items())),
        "ae": ae_total,
        "ya": ya_total,
        "n_rows": int(len(scored)),
    }


def select_horizon(df: pd.DataFrame, horizon: Horizon) -> pd.DataFrame:
    """Restrict to one half of the 672h forecast. Requires a ``step`` column."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon!r}")
    if horizon == "all":
        return df
    if STEP not in df.columns:
        raise KeyError(
            f"horizon={horizon!r} needs a {STEP!r} column (1..672). Frames from results/impute/ "
            "hold the scored block only, which is entirely 'far' — use horizon='all' there."
        )
    steps = df[STEP].to_numpy()
    return df[steps <= NEAR_FAR_SPLIT] if horizon == "near" else df[steps > NEAR_FAR_SPLIT]


def metric_report(
    df: pd.DataFrame,
    member: str,
    *,
    horizons: tuple[str, ...] = HORIZONS,
    target: str = NF_TARGET,
) -> dict:
    """All six leaderboard metrics for one member, split NEAR / FAR / all. Issue #55.

    Reporting WAPE alone hides two things this project has been bitten by. First, WAPE is
    volume-weighted, so a member can improve it while getting *most* series worse — MAPE and sMAPE
    weight every row equally and expose that. Second, a model can win the near half and lose the
    far half; only the far half resembles the private test, and the cascade is exactly this case.

    Each horizon carries the pooled figure plus the per-window breakdown, so nothing here can be
    quoted without also being able to see its spread.
    """
    if member not in df.columns:
        raise KeyError(f"member column {member!r} not in frame; have {sorted(df.columns)}")

    out: dict = {"member": member, "horizons": {}}
    for h in horizons:
        sl = select_horizon(df, h)
        if sl.empty:
            out["horizons"][h] = None
            continue
        pooled = all_metrics(sl[target], sl[member])
        per_window = {}
        if CUTOFF in sl.columns:
            for cut, g in sl.groupby(CUTOFF):
                per_window[int(cut)] = all_metrics(g[target], g[member])
        out["horizons"][h] = {
            "pooled": pooled,
            "per_window": dict(sorted(per_window.items())),
            "n_rows": int(len(sl)),
        }
    return out


def window_sigma(pooled_report: dict) -> float:
    """Population std of a member's per-window WAPEs — the sigma the admission rule subtracts.

    Note this is cross-*window* dispersion, not cross-seed. It says how much the score moves as the
    evaluation block moves, and says nothing about seed variance.
    """
    vals = [v for v in pooled_report["per_window_wape"].values() if not np.isnan(v)]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


# --------------------------------------------------------------------------- schema validation


def validate_prediction_df(df: pd.DataFrame, member_name: str) -> pd.DataFrame:
    """Assert a member's prediction frame is well-formed; return it unchanged.

    Enforces ``['unique_id', 'ds', 'cutoff', 'y', <member_name>]``, sorted by
    ``['unique_id', 'ds']`` within each cutoff, and no NaN in the target or prediction column.
    Raises ``AssertionError`` on any violation — this is a tripwire, not a repair function.
    """
    required = [NF_ID, NF_TIME, CUTOFF, NF_TARGET, member_name]
    missing = [c for c in required if c not in df.columns]
    assert not missing, (
        f"{member_name}: missing required column(s) {missing}; have {sorted(df.columns)}"
    )

    assert len(df) > 0, f"{member_name}: empty prediction frame"

    for col in (NF_TARGET, member_name):
        n_nan = int(df[col].isna().sum())
        assert n_nan == 0, f"{member_name}: {n_nan} NaN in column {col!r}"

    sort_keys = [CUTOFF, NF_ID, NF_TIME]
    expected = df.sort_values(sort_keys).reset_index(drop=True)
    actual = df.reset_index(drop=True)
    assert actual[sort_keys].equals(expected[sort_keys]), (
        f"{member_name}: rows are not sorted by {sort_keys}. Pooling and blk indexing both assume "
        "this order, so an unsorted frame silently mis-assigns the held-out block."
    )

    dupes = int(df.duplicated(subset=[CUTOFF, NF_ID, NF_TIME]).sum())
    assert dupes == 0, f"{member_name}: {dupes} duplicate (cutoff, unique_id, ds) rows"

    return df


# --------------------------------------------------------------------------- admission


def error_correlation(
    df: pd.DataFrame, member_a: str, member_b: str, *, target: str = NF_TARGET
) -> float:
    """Pearson correlation of two members' residuals. Low = orthogonal = worth blending (#37)."""
    ra = df[member_a].to_numpy(dtype=float) - df[target].to_numpy(dtype=float)
    rb = df[member_b].to_numpy(dtype=float) - df[target].to_numpy(dtype=float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def min_windows_required(n_windows: int) -> int:
    """How many of ``n_windows`` a candidate must win. 3 -> 2, 2 -> 2, 1 -> 1."""
    return max(1, math.ceil(MIN_WINDOW_FRACTION * n_windows))


def paired_bootstrap_delta(
    df: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    regime: Regime,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
    target: str = NF_TARGET,
) -> dict:
    """Cluster-bootstrap the pooled-WAPE **difference**, ``baseline - candidate``.

    This is the honest uncertainty of an A/B, and it is *not* ``window_sigma``. ``window_sigma``
    measures how much one model's score moves as the evaluation block moves; an A/B runs on
    identical rows, so window difficulty is common to both arms and largely cancels out of the
    difference. Charging a candidate for it charges it for noise the comparison design removed.
    Measured on the cached late cube: ``window_sigma(tft) = 0.0076`` vs a paired SE of ``0.0028``.

    Resamples whole ``(cutoff, unique_id)`` blocks, not rows: hourly residuals within a series are
    strongly autocorrelated, so a row-level resample would badly understate the SE. Both arms see
    the same draw, which is what makes it paired.

    Returns ``delta`` (positive = candidate better), ``se``, ``ci95``, and ``delta_in_se``.
    """
    scored = select_regime(df, regime).reset_index(drop=True)
    for m in (candidate, baseline):
        if m not in scored.columns:
            raise KeyError(f"member column {m!r} not in frame; have {sorted(scored.columns)}")

    keys = [c for c in (CUTOFF, NF_ID) if c in scored.columns]
    blocks = list(scored.groupby(keys, sort=True).indices.values()) if keys else [scored.index]
    y = scored[target].to_numpy(dtype=float)
    pc = scored[candidate].to_numpy(dtype=float)
    pb = scored[baseline].to_numpy(dtype=float)

    def _delta(idx) -> float:
        return wape_stats(y[idx], pb[idx])["wape"] - wape_stats(y[idx], pc[idx])["wape"]

    observed = _delta(np.arange(len(scored)))
    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = _delta(np.concatenate([blocks[j] for j in rng.integers(0, n_blocks, n_blocks)]))

    se = float(draws.std(ddof=1))
    return {
        "delta": float(observed),
        "se": se,
        "ci95": (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))),
        "delta_in_se": float(observed / se) if se > 0 else float("nan"),
        "n_blocks": n_blocks,
        "n_boot": n_boot,
    }


def evaluate_admission(
    candidate_preds: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    regime: Regime,
    err_corr_threshold: float = ERR_CORR_THRESHOLD,
    incumbents: list[str] | None = None,
    n_boot: int | None = N_BOOTSTRAP,
    bootstrap_seed: int = 0,
    target: str = NF_TARGET,
) -> dict:
    """Run the three admission checks and **report**. Never drops a member automatically.

    The rule, fixed 2026-07-30 (see ``docs/protocol.md``):

      1. **Pooled improvement** — the candidate's pooled WAPE beats the baseline's. *Any*
         improvement counts; there is no sigma threshold to clear. For a convex blend a member with
         positive expected gain is worth having, and the weight search decides how much to trust it.
      2. **Majority of windows** — the candidate improves in at least ``min_windows_required(n)``
         windows, i.e. 2 of 3. Guards the pathology a pooled number cannot see: one huge win
         carrying a candidate that is worse everywhere else.
      3. **Orthogonality** — error correlation against every incumbent below ``err_corr_threshold``.
         A candidate that merely reproduces an existing member adds fitting risk without adding
         information, however good it looks alone.

    The same three checks serve both stages; only the data changes. **Screening** (every Phase-4
    lever) feeds single-seed-42 predictions — cheap, and a false negative costs more than a false
    positive. **Admission** (final members) feeds seed-averaged predictions over
    ``frozen_seeds()``. Note that seed-averaging removes training stochasticity (LGBM: 0.0005) but
    *not* the dominant unit-sampling noise (paired SE: 0.0028) — more seeds is not more data, so a
    small multi-seed win is an expected-value argument, not a statistical one.

    ``evidence`` carries the paired bootstrap CI and per-window deltas. These are **reported, never
    gated on** — the two rules this replaces (the 19% relative bar, and 1x cross-window sigma) both
    failed by turning a dispersion measured on one quantity into a threshold on another.
    """
    cand = compute_pooled_wape(candidate_preds, candidate, regime, target=target)
    base = compute_pooled_wape(baseline_preds, baseline, regime, target=target)

    margin = base["pooled_wape"] - cand["pooled_wape"]  # positive = candidate better
    pooled_improves = bool(margin > 0)

    shared = sorted(set(cand["per_window_wape"]) & set(base["per_window_wape"]))
    per_window_delta = {w: base["per_window_wape"][w] - cand["per_window_wape"][w] for w in shared}
    windows_won = [w for w, d in per_window_delta.items() if d > 0]
    required = min_windows_required(len(shared))
    majority_windows = bool(shared) and len(windows_won) >= required

    corrs: dict[str, float] = {}
    if incumbents:
        joined = select_regime(candidate_preds, regime)
        for other in incumbents:
            if other in joined.columns and candidate in joined.columns:
                corrs[other] = error_correlation(joined, candidate, other, target=target)
    orthogonal = (
        all(c < err_corr_threshold for c in corrs.values() if not np.isnan(c)) if corrs else None
    )

    checks = {
        "pooled_improves": {
            "pass": pooled_improves,
            "candidate_pooled_wape": cand["pooled_wape"],
            "baseline_pooled_wape": base["pooled_wape"],
            "margin": margin,
            "margin_relative": (margin / base["pooled_wape"])
            if base["pooled_wape"]
            else float("nan"),
        },
        "majority_windows_improve": {
            "pass": majority_windows,
            "windows_won": windows_won,
            "windows_required": required,
            "windows_total": len(shared),
            "per_window_delta": per_window_delta,
        },
        "orthogonal": {
            "pass": orthogonal,
            "threshold": err_corr_threshold,
            "err_corr": corrs,
        },
    }
    passed = [k for k, v in checks.items() if v["pass"] is True]
    missed = [k for k, v in checks.items() if v["pass"] is False]

    evidence: dict = {
        "per_window_delta": per_window_delta,
        # Spread of the per-window deltas. A HETEROGENEITY diagnostic, deliberately not a noise bar:
        # with n=3 it cannot separate "the effect varies by window" from "we measured it noisily",
        # and on the tft-vs-lgbm case it reads 0.0155 — larger than the cross-window sigma it would
        # be replacing, because TFT's margin genuinely ranges 0.0014-0.0318 across windows.
        "per_window_delta_spread": (
            float(np.std(list(per_window_delta.values()), ddof=1)) if len(shared) > 1 else 0.0
        ),
        "baseline_window_sigma": window_sigma(base),
    }
    if n_boot:
        try:
            evidence["paired_bootstrap"] = paired_bootstrap_delta(
                candidate_preds,
                candidate=candidate,
                baseline=baseline,
                regime=regime,
                n_boot=n_boot,
                seed=bootstrap_seed,
                target=target,
            )
        except KeyError:
            # candidate and baseline live in different frames — a paired resample is undefined
            evidence["paired_bootstrap"] = None

    return {
        "candidate": candidate,
        "baseline": baseline,
        "regime": regime,
        "checks": checks,
        "checks_passed": passed,
        "checks_missed": missed,
        "evidence": evidence,
        "summary": (
            f"{candidate} vs {baseline} [{regime}]: "
            f"{cand['pooled_wape']:.4f} vs {base['pooled_wape']:.4f} "
            f"(margin {margin:+.4f}, {margin / base['pooled_wape']:+.1%}); "
            f"windows won {len(windows_won)}/{len(shared)} (need {required}); "
            f"{len(passed)}/{len(checks)} checks pass"
        ),
        "note": "Advisory only — no member is dropped automatically. Read the checks and decide.",
    }
