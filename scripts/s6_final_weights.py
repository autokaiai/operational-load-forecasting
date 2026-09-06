"""S6 — the final two-member weight fit, on seed-averaged cubes.

Why this script exists rather than another ``--extra`` on ``blend_vs_cascade.py``
--------------------------------------------------------------------------------
Three things S6 has to do differently, and each of them breaks that script's assumptions:

1. **The members live in different directories.** The cascade's nine cubes are S5b's ``elu_plain``
   arm (3 windows x 3 seeds) under ``results/s5ab__elu_plain_s*/``; the tree is the ``interp`` arm
   under ``results/gapcov_nan-interp/``. S3 adopted ``interp`` for the TREE ONLY and left the
   cascade on ``median``, so the ship blend is a *cross-directory* join by construction.
   ``blend_vs_cascade.py`` reads one root.

2. **The weights must be fitted on a SEED-AVERAGED cascade, not on one draw.** S5b measured
   ``tft_cascade`` at 0.13945 +- 0.00380 across three seeds, and the 0.13429 on record is a single
   draw ~1.4 sigma on the favourable side. Fitting a convex weight against one draw tunes it to
   that draw's error structure — and the error structure is exactly what a blend weight reads.

3. **Averaging those nine cubes IS Tier D's seed-bagging lever**, delivered at zero extra GPU
   because S5b already paid for the fits. So the bagged member's own WAPE is a result in its own
   right, not merely an input to the weight fit, and this script reports it as one.

What is measured, and on which rows
-----------------------------------
Weights are fitted on ``blk < 224`` and scored on ``blk >= 224`` — the split ``src.eval.protocol``
reserves for exactly this. ``blk`` is the 0-based hour inside each window's own 336h scored block,
so the fit sees hours 0-223 and the score sees 224-335, per window, never mixed across cutoffs.

The **oracle** weight (#37) is the same grid minimised directly on the scored rows. It cannot be
used to ship anything — it has seen the answer — but the gap between it and the honestly-fitted
weight is the price of choosing the weight, and #37 asks whether an oracle two-member blend clears
its bar at all. If the oracle barely beats the fitted weight, the weight search is not where the
remaining headroom is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.protocol import (
    CUT_BLK,
    add_block_index,
    error_correlation,
    frozen_seeds,
    paired_bootstrap_delta,
    summarize_seed_runs,
    wape_stats,
)

CUTOFFS = {0: 3648, 1: 3312, 2: 2976}
SEEDS = frozen_seeds(5)  # complete frozen set, draw order — S5b's 3 + S6's top-up
CASCADE = "tft_cascade"
TREE = "lgbm_s24_unitcat"

# THREE cascade cubes exist locally and they are NOT interchangeable, which is worth pinning in
# code because two of them are within 0.0002 of each other and the third is a different decision:
#
#   results/gapcov/                     late 0.13409  the S2-era harness (S2 Stage 2's baseline)
#   results/gapfill__median/            late 0.13429  THIS ONE — S3's harness at median/median,
#                                                     bit-identical to results/s4__median/
#   results/gapfill_nan-interp__median/ late 0.14342  the cascade under `interp`, which S3 REJECTED
#
# S5b's nine seed cubes ran at nan_fill="median", gap_fill="median" (``_trial_body`` defaults), so
# ``gapfill__median`` is the only control on the same footing as the seeds. Picking ``gapcov``
# instead silently compares across harness generations — the arms differ by max|dpred| 5.94.
CONTROL_ROOT = "results/gapfill__median"
TREE_ROOT = "results/gapcov_nan-interp"  # S3 adopted `interp` for the TREE ONLY

# The plan's recorded ship candidate, reproduced as a guard rather than trusted:
# 0.33*lgbm_s24_unitcat(interp) + 0.67*tft_cascade(median) = 0.13244 late.
RECORDED_SHIP_WAPE = 0.13244
RECORDED_SHIP_TOL = 5e-5

# The bar #37 was filed against. Stale — the incumbent is far below it — but the issue asks the
# question against this number, so it is answered against this number and against the incumbent.
ORACLE_BAR_37 = 0.143


def _seed_dir(seed: int, window: int) -> Path:
    return Path(f"results/s5ab__elu_plain_s{seed}/window{window}")


def build_frame(*, tree_root: str, control_root: str) -> pd.DataFrame:
    """One wide frame: every member on identical ``(cutoff, unique_id, ds)`` rows.

    Every merge is checked for row loss rather than trusted. A silent inner-join drop would not
    raise anywhere downstream — it would just quietly score a different set of hours for one
    member than for another, which is the failure mode ``validate_prediction_df`` exists to catch
    one level up.
    """
    frames = []
    for w in sorted(CUTOFFS):
        # The cascade seeds. Column is ``tft_cascade`` in every cube, so rename per seed.
        base = None
        for seed in SEEDS:
            path = _seed_dir(seed, w) / f"{CASCADE}_preds.csv"
            df = pd.read_csv(path)
            if CASCADE not in df.columns:
                raise SystemExit(f"{path} has no column {CASCADE!r}; found {sorted(df.columns)}")
            col = f"casc_s{seed}"
            df = df[["unique_id", "ds", "cutoff", "y", CASCADE]].rename(columns={CASCADE: col})
            if base is None:
                base = df
            else:
                n = len(base)
                base = base.merge(df[["unique_id", "ds", col]], on=["unique_id", "ds"], how="inner")
                if len(base) != n:
                    raise SystemExit(f"W{w} seed {seed}: {n} -> {len(base)} rows after merge")

        # The historical single-draw control (effective seed 1) — continuity with 0.13429.
        ctl = pd.read_csv(f"{control_root}/window{w}/{CASCADE}_preds.csv")
        ctl = ctl[["unique_id", "ds", CASCADE]].rename(columns={CASCADE: "casc_seed1"})
        n = len(base)
        base = base.merge(ctl, on=["unique_id", "ds"], how="inner")
        if len(base) != n:
            raise SystemExit(f"W{w} control: {n} -> {len(base)} rows after merge")

        # The tree, under the fill S3 adopted for it.
        tre = pd.read_csv(f"{tree_root}/window{w}/{TREE}_preds.csv")
        if TREE not in tre.columns:
            raise SystemExit(f"{tree_root}/window{w}: no column {TREE!r}")
        tre = tre[["unique_id", "ds", TREE]].rename(columns={TREE: "tree"})
        n = len(base)
        base = base.merge(tre, on=["unique_id", "ds"], how="inner")
        if len(base) != n:
            raise SystemExit(f"W{w} tree: {n} -> {len(base)} rows after merge")

        base["window"] = w
        frames.append(base)

    out = add_block_index(pd.concat(frames, ignore_index=True))
    # Seed-bagging: average the FORECASTS, then score once. Averaging the three per-seed WAPEs
    # instead would be a mean-of-ratios and a different (worse, and non-shippable) quantity.
    out["casc_bag"] = out[[f"casc_s{s}" for s in SEEDS]].mean(axis=1)
    return out


def pooled(df: pd.DataFrame, col: str) -> float:
    """Pooled WAPE over whatever rows are passed — numerators summed, divided ONCE."""
    return wape_stats(df["y"], df[col])["wape"]


def per_window(df: pd.DataFrame, col: str) -> dict[int, float]:
    return {int(w): pooled(g, col) for w, g in df.groupby("window")}


def fit_weight(fit_df: pd.DataFrame, tree_col: str, casc_col: str, grid: int = 101) -> float:
    """Weight on the TREE minimising pooled WAPE over ``fit_df``.

    Returns w in ``w*tree + (1-w)*cascade``.
    """
    ws = np.linspace(0.0, 1.0, grid)
    y = fit_df["y"].to_numpy(float)
    a = fit_df[tree_col].to_numpy(float)
    b = fit_df[casc_col].to_numpy(float)
    scores = [wape_stats(y, w * a + (1 - w) * b)["wape"] for w in ws]
    return float(ws[int(np.argmin(scores))])


def blend_report(df: pd.DataFrame, casc_col: str) -> dict:
    """Fit the pair weight honestly, score on the held-out tail, price the oracle beside it."""
    fit = df[df["blk"] < CUT_BLK]
    late = df[df["blk"] >= CUT_BLK]

    w_fit = fit_weight(fit, "tree", casc_col)
    w_oracle = fit_weight(late, "tree", casc_col)

    scored = df.copy()
    scored["blend_fit"] = w_fit * scored["tree"] + (1 - w_fit) * scored[casc_col]
    scored["blend_oracle"] = w_oracle * scored["tree"] + (1 - w_oracle) * scored[casc_col]
    late_scored = scored[scored["blk"] >= CUT_BLK]

    return {
        "cascade_member": casc_col,
        "tree_weight_fitted": w_fit,
        "tree_weight_oracle": w_oracle,
        "late_blend_fitted": pooled(late_scored, "blend_fit"),
        "late_blend_oracle": pooled(late_scored, "blend_oracle"),
        "late_cascade_alone": pooled(late, casc_col),
        "late_tree_alone": pooled(late, "tree"),
        "err_corr_tree_vs_cascade": error_correlation(late, "tree", casc_col),
        "per_window_blend_fitted": per_window(late_scored, "blend_fit"),
        "selection_cost": pooled(late_scored, "blend_fit") - pooled(late_scored, "blend_oracle"),
        "_scored": late_scored,  # stripped before serialising
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree-root", default=TREE_ROOT)
    ap.add_argument("--control-root", default=CONTROL_ROOT)
    ap.add_argument("--out", default="results/s6_final_weights.json")
    args = ap.parse_args()

    df = build_frame(tree_root=args.tree_root, control_root=args.control_root)
    late = df[df["blk"] >= CUT_BLK]
    print(
        f"rows: {len(df)} total, {len(late)} late (blk >= {CUT_BLK}), {df.window.nunique()} windows"
    )

    # ------------------------------------------------------------- 0. reproduce what is on record
    # If the recorded ship candidate does not come back out of this frame, every number below is
    # being computed on a different instrument than the one the plan quotes, and the phase should
    # stop rather than report. S4's `max|dpred| = 0` is the standard this is held to.
    fixed = 0.33 * late["tree"] + 0.67 * late["casc_seed1"]
    repro = wape_stats(late["y"], fixed)["wape"]
    ok = abs(repro - RECORDED_SHIP_WAPE) <= RECORDED_SHIP_TOL
    print("\n=== 0. drift guard ===")
    print(
        f"  recorded recipe 0.33*tree(interp)+0.67*cascade(median): {repro:.5f} "
        f"vs recorded {RECORDED_SHIP_WAPE} -> {'OK' if ok else 'MISMATCH'}"
    )
    print(
        f"  control cascade {pooled(late, 'casc_seed1'):.5f} (expect 0.13429)   "
        f"tree {pooled(late, 'tree'):.5f} (expect 0.15200)"
    )
    if not ok:
        raise SystemExit(
            f"drift guard failed: {repro:.5f} != {RECORDED_SHIP_WAPE}. Wrong cube roots?"
        )

    # ---------------------------------------------------------------- 1. the members, solo
    seed_cols = [f"casc_s{s}" for s in SEEDS]
    seed_scores = {s: pooled(late, f"casc_s{s}") for s in SEEDS}
    seed_mean = float(np.mean(list(seed_scores.values())))
    seed_std = float(np.std(list(seed_scores.values()), ddof=1))
    bag = pooled(late, "casc_bag")

    solo = {
        "seed_scores": seed_scores,
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "seed_best": min(seed_scores.values()),
        "bagged": bag,
        "bag_vs_mean_seed": seed_mean - bag,
        "bag_vs_best_seed": min(seed_scores.values()) - bag,
        "control_seed1": pooled(late, "casc_seed1"),
        "tree_interp": pooled(late, "tree"),
        "per_window_bag": per_window(late, "casc_bag"),
        "per_window_control": per_window(late, "casc_seed1"),
    }

    print("\n=== 1. members, solo (late, pooled over 3 windows) ===")
    for s in SEEDS:
        print(f"  tft_cascade  seed {s:>5}      {seed_scores[s]:.5f}")
    print(f"  mean +- std                 {seed_mean:.5f} +- {seed_std:.5f}")
    print(
        f"  SEED-BAGGED ({len(SEEDS)} forecasts)   {bag:.5f}   "
        f"(vs mean seed {seed_mean - bag:+.5f}, "
        f"vs best seed {min(seed_scores.values()) - bag:+.5f})"
    )
    print(f"  control, effective seed 1   {solo['control_seed1']:.5f}")
    print(f"  lgbm_s24_unitcat (interp)   {solo['tree_interp']:.5f}")

    # Is bagging a real lever, or is it inside the noise of the thing it averages?
    boot_bag = paired_bootstrap_delta(
        late.assign(**{"cand": late["casc_bag"], "base": late["casc_s892"]}),
        candidate="cand",
        baseline="base",
        regime="full",
    )
    solo["bag_vs_seed892_bootstrap"] = boot_bag
    print(
        f"  bagged vs seed 892: delta {boot_bag['delta']:+.5f}, "
        f"CI [{boot_bag['ci95'][0]:+.5f}, {boot_bag['ci95'][1]:+.5f}], "
        f"{boot_bag['delta_in_se']:.1f} SE"
    )

    # ---------------------------------------------------------------- 2. the weight fit
    print("\n=== 2. the two-member weight fit (fit blk<224, score blk>=224) ===")
    print(
        f"{'cascade arm':<22} {'w(tree)':>8} {'blend late':>12} "
        f"{'oracle':>10} {'sel.cost':>10} {'err-corr':>9}"
    )
    blends = {}
    for col in ["casc_bag", "casc_seed1", *seed_cols]:
        rep = blend_report(df, col)
        blends[col] = rep
        print(
            f"{col:<22} {rep['tree_weight_fitted']:>8.2f} {rep['late_blend_fitted']:>12.5f} "
            f"{rep['late_blend_oracle']:>10.5f} {rep['selection_cost']:>10.5f} "
            f"{rep['err_corr_tree_vs_cascade']:>9.3f}"
        )

    # ------------------------------------------------- 2b. what bagging buys AT THE BLEND LEVEL
    # The member-level bagging gain is not the shipping question. What ships is a *blend*, and a
    # single-seed blend is a draw from a distribution whose mean is what an unseen block would get.
    # So compare the bagged blend against the mean of the three single-seed blends, not against the
    # luckiest of them — picking the best-scoring seed is selection on three windows, which is
    # precisely the move S5 was punished for.
    seed_blends = {s: blends[f"casc_s{s}"]["late_blend_fitted"] for s in SEEDS}
    sb_mean = float(np.mean(list(seed_blends.values())))
    sb_std = float(np.std(list(seed_blends.values()), ddof=1))
    bag_blend = blends["casc_bag"]["late_blend_fitted"]
    ctl_blend = blends["casc_seed1"]["late_blend_fitted"]

    bagging = {
        "single_seed_blends": seed_blends,
        "single_seed_blend_mean": sb_mean,
        "single_seed_blend_std": sb_std,
        "bagged_blend": bag_blend,
        "bagging_gain_vs_expected_single_seed": sb_mean - bag_blend,
        "control_seed1_blend": ctl_blend,
        "bagged_vs_control": ctl_blend - bag_blend,
    }
    print("\n=== 2b. what seed-bagging buys AT THE BLEND LEVEL ===")
    for s in SEEDS:
        print(f"  blend on seed {s:>5}          {seed_blends[s]:.5f}")
    print(f"  expected single-seed blend  {sb_mean:.5f} +- {sb_std:.5f}")
    print(
        f"  BAGGED blend                {bag_blend:.5f}   "
        f"({sb_mean - bag_blend:+.5f} vs the expected single seed)"
    )
    print(
        f"  control (effective seed 1)  {ctl_blend:.5f}   "
        f"({ctl_blend - bag_blend:+.5f} vs bagged — one lucky draw, not an expectation)"
    )

    # Is the recorded 0.13244 actually distinguishable from the bagged blend, or is the whole
    # difference inside the noise the bag exists to remove?
    bs = blends["casc_bag"]["_scored"]
    cs = blends["casc_seed1"]["_scored"]
    cmp_frame = bs[["cutoff", "unique_id", "y"]].copy()
    cmp_frame["bagged"] = bs["blend_fit"].to_numpy()
    cmp_frame["seed1"] = cs["blend_fit"].to_numpy()
    boot_blend = paired_bootstrap_delta(
        cmp_frame, candidate="bagged", baseline="seed1", regime="full"
    )
    bagging["bagged_vs_control_bootstrap"] = boot_blend
    print(
        f"  bagged vs control, paired:  delta {boot_blend['delta']:+.5f}, "
        f"CI [{boot_blend['ci95'][0]:+.5f}, {boot_blend['ci95'][1]:+.5f}], "
        f"{boot_blend['delta_in_se']:.1f} SE"
    )

    # ------------------------------------------------------- 3. does the blend beat its parts
    print("\n=== 3. the shipped blend vs its own members (paired, 3 windows) ===")
    ship = blends["casc_bag"]
    sc = ship["_scored"]
    ab = {}
    for base in ("casc_bag", "tree"):
        boot = paired_bootstrap_delta(
            sc.assign(cand=sc["blend_fit"], base=sc[base]),
            candidate="cand",
            baseline="base",
            regime="full",
        )
        wins = sum(
            1
            for _, g in sc.groupby("window")
            if wape_stats(g["y"], g["blend_fit"])["wape"] < wape_stats(g["y"], g[base])["wape"]
        )
        ab[base] = {**boot, "windows_won": wins}
        print(
            f"  vs {base:<12} delta {boot['delta']:+.5f}  "
            f"CI [{boot['ci95'][0]:+.5f}, {boot['ci95'][1]:+.5f}]  "
            f"{boot['delta_in_se']:>5.1f} SE  {wins}/3 windows"
        )

    # ---------------------------------------------------------------- 4. #37's oracle gate
    print("\n=== 4. #37's oracle gate ===")
    oracle_late = ship["late_blend_oracle"]
    gate37 = {
        "bar": ORACLE_BAR_37,
        "oracle_two_member_blend": oracle_late,
        "beats_bar": bool(oracle_late < ORACLE_BAR_37),
        "fitted_two_member_blend": ship["late_blend_fitted"],
        "headroom_left_by_weight_choice": ship["selection_cost"],
    }
    print(
        f"  oracle 2-member blend  {oracle_late:.5f}  vs #37's bar {ORACLE_BAR_37}  "
        f"-> {'CLEARS' if gate37['beats_bar'] else 'MISSES'}"
    )
    print(
        f"  honestly-fitted blend  {ship['late_blend_fitted']:.5f}  "
        f"(the weight choice costs {ship['selection_cost']:+.5f})"
    )

    # Stage 2's formal record. `summarize_seed_runs` refuses a subset that is not a DRAW-ORDER
    # prefix of FROZEN_SEEDS, so a hand-picked three cannot be laundered into a `mean +- std`.
    stage2 = {
        "member": summarize_seed_runs(seed_scores),
        "blend": summarize_seed_runs(seed_blends),
    }
    print(f"\n=== 4b. Stage 2 record over frozen_seeds({len(SEEDS)}) ===")
    for k, v in stage2.items():
        print(f"  {k:<7} {v['headline']}  seeds {v['seeds']}")
        if v["caveat"]:
            print(f"          {v['caveat']}")

    print("\n=== 5. per-window, the two ship candidates (late) ===")
    for name, rep in (("bagged blend", blends["casc_bag"]), ("seed-1 blend", blends["casc_seed1"])):
        pw = rep["per_window_blend_fitted"]
        print(
            f"  {name:<14} W0 {pw[0]:.5f}  W1 {pw[1]:.5f}  W2 {pw[2]:.5f}   "
            f"pooled {rep['late_blend_fitted']:.5f}"
        )

    payload = {
        "bagging_at_blend_level": bagging,
        "stage2_frozen_seeds": stage2,
        "solo": solo,
        "blends": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in blends.items()
        },
        "ship_blend_vs_members": ab,
        "gate_37_oracle": gate37,
        "seeds": list(SEEDS),
        "n_rows_late": int(len(late)),
        "cut_blk": CUT_BLK,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
