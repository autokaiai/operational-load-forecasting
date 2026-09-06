"""Sweep the member pool for the best combination — one frame, one code path, one set of rows.

The 2026-08-17 question: which combination yields the best result, with every individual number
reported rather than only the winner.

Why no retraining was needed for comparability
----------------------------------------------
Every member here is ALREADY at the gapped protocol. ``members.py:575`` computes
``h = gapped_horizon(df, cut_idx)`` = 672 and OVERRIDES whatever the config says, so
``configs/bitcn.yaml``'s inherited ``h: 336`` never reached a model. Verified against the cubes:
window0 spans hours 3984..4319 for every member, i.e. exactly ``[cut+336, cut+672)`` — the FAR
block — on byte-identical ``(unique_id, ds)`` rows. A blend search needs predictions, not weights.

The one rule this script enforces
---------------------------------
S6 caught me quoting a pooled number from one cube generation against another. Here every member
and every candidate blend is scored on ONE joined frame, and the incumbent is RE-DERIVED inside
that frame rather than quoted from the plan. A number here is comparable only to another number
here.

Weights are fitted on ``blk < 224`` (the first 224h of the far block) and scored on ``blk >= 224``
(the last 112h) — the split ``src.eval.protocol`` reserves for exactly this.

Why the search is exact and fast
--------------------------------
Pooled WAPE accumulates numerator and denominator across windows and divides ONCE, so on a fixed
row set it is ``sum|y - Pw| / sum|y|`` — a convex function of ``w``. On the simplex that makes
Frank-Wolfe exact, and its line search is a *weighted median* (closed form, one sort) rather than a
grid. Everything runs out of two dense numpy matrices built once, so a combination costs
milliseconds instead of a DataFrame copy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from src.eval.protocol import CUT_BLK, add_block_index, error_correlation, paired_bootstrap_delta

CUTOFFS = {0: 3648, 1: 3312, 2: 2976}
SEEDS = (892, 7739, 6545, 4388, 4330)  # DRAW order — frozen_seeds()

# The pool. name -> (generation root, file stem, column inside the file)
POOL: dict[str, tuple[str, str, str]] = {
    "tft": ("results/gapcov", "tft", "tft"),
    "tft_cascade": ("results/gapcov", "tft_cascade", "tft_cascade"),
    "chronos2_zeroshot": ("results/gapcov", "chronos2_zeroshot", "chronos2_zeroshot"),
    "chronos_full_ft": ("results/gapcov", "chronos_full_ft", "chronos_full_ft"),
    "bitcn": ("results/gapcov", "bitcn", "bitcn"),
    "bitcn_wide": ("results/gapcov", "bitcn_wide", "bitcn_wide"),
    "catboost": ("results/gapcov", "catboost", "catboost"),
    "lgbm_s24_unitcat": ("results/gapcov", "lgbm_s24_unitcat", "lgbm_s24_unitcat"),
    "lgbm_s24_norm": ("results/gapcov", "lgbm_s24_norm", "lgbm_s24_norm"),
    "lgbm_s24_recency": ("results/gapcov", "lgbm_s24_recency", "lgbm_s24_recency"),
    "lgbm_s24_wlag": ("results/gapcov", "lgbm_s24_wlag", "lgbm_s24_wlag"),
}

# S12 — the four pretrained families as BLEND MEMBERS, which they have never been.
#
# S2 screened them (2026-08-04) against `tft_cascade` alone and closed the phase because none beat
# the incumbent PAIR. The pool has since changed underneath that verdict: `chronos_full_ft` joined
# at 0.13250 and the ship model is three members at 0.12722, so "does this candidate earn weight"
# is being asked against a different incumbent. S10.2's law is what makes it worth re-asking —
# what earns weight is accuracy NOT ALREADY CORRELATED with an incumbent, and every one of these
# sits at err-corr 0.59-0.95 against a pool that no longer looks the way it did.
#
# Free: the frames were generated in S2 and the member-cube builder slices the scored block out of
# them. No model, no GPU, no re-run.
FOUNDATION: dict[str, tuple[str, str, str]] = {
    m: ("results/gapcov", m, m)
    for m in (
        "toto_cov",
        "timesfm_cov",
        "tabpfn_ts_cov",
        "toto",
        "timesfm",
        "tabpfn_ts",
        "tirex",
    )
}
# The SHIPPED members, carried so 0.13162 is a like-for-like reference, not a quotation.
REFERENCE: dict[str, tuple[str, str, str]] = {
    "tree_interp": ("results/gapcov_nan-interp", "lgbm_s24_unitcat", "lgbm_s24_unitcat"),
}
SEED_SRC = {
    f"casc_s{s}": (f"results/s5ab__elu_plain_s{s}", "tft_cascade", "tft_cascade") for s in SEEDS
}
BAG = "casc_bag5"


def build(optional: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Join every member on (unique_id, ds) per window. Members in ``optional`` may be absent."""
    sources, missing = {**POOL, **REFERENCE, **SEED_SRC, **FOUNDATION}, []
    for name, (root, stem, _) in sources.items():
        paths = [f"{root}/window{w}/{stem}_preds.csv" for w in CUTOFFS]
        if not all(os.path.exists(q) for q in paths):
            if name not in optional:
                raise SystemExit(f"missing cube for required member {name!r}")
            missing.append(name)
    sources = {k: v for k, v in sources.items() if k not in missing}

    frames = []
    for w, cut in CUTOFFS.items():
        base = None
        for name, (root, stem, col) in sources.items():
            d = pd.read_csv(f"{root}/window{w}/{stem}_preds.csv")
            if col not in d.columns:
                raise SystemExit(f"{root}/window{w}/{stem}: no column {col!r}")
            d = d[["unique_id", "ds", "y", col]].rename(columns={col: name})
            if base is None:
                base = d
                continue
            n0 = len(base)
            base = base.merge(
                d.drop(columns=["y"]), on=["unique_id", "ds"], how="inner", validate="one_to_one"
            )
            if len(base) != n0:
                raise SystemExit(f"W{w}: {name} covers {len(base)} of {n0} rows")
        base["cutoff"] = cut
        frames.append(base)
    df = add_block_index(pd.concat(frames, ignore_index=True))
    df[BAG] = df[[f"casc_s{s}" for s in SEEDS]].mean(axis=1)
    return df, missing


def _wmedian_step(r: np.ndarray, d: np.ndarray) -> float:
    """Exact minimiser of ``sum|r - g*d|`` over g in [0,1]: a weighted median, one sort."""
    nz = d != 0
    if not nz.any():
        return 0.0
    t, wt = r[nz] / d[nz], np.abs(d[nz])
    o = np.argsort(t, kind="stable")
    c = np.cumsum(wt[o])
    g = float(t[o][int(np.searchsorted(c, 0.5 * c[-1]))])
    return float(min(max(g, 0.0), 1.0))


def fit_simplex(P: np.ndarray, y: np.ndarray, iters: int = 300) -> np.ndarray:
    """Frank-Wolfe on the simplex — exact for this convex L1 objective."""
    n = P.shape[1]
    if n == 1:
        return np.ones(1)
    if n == 2:  # closed form: one weighted median, no iteration
        g = _wmedian_step(y - P[:, 1], P[:, 0] - P[:, 1])
        return np.array([g, 1.0 - g])
    w = np.full(n, 1.0 / n)
    pred = P @ w
    stall = 0
    for _ in range(iters):
        j = int(np.argmin(-(P.T @ np.sign(y - pred))))
        d = P[:, j] - pred
        g = _wmedian_step(y - pred, d)
        if g <= 1e-9:
            stall += 1
            if stall >= 5:
                break
            continue
        stall = 0
        w *= 1 - g
        w[j] += g
        pred += g * d
    return w


class Search:
    """Two dense matrices built once; every combination is then pure arithmetic."""

    def __init__(self, df: pd.DataFrame, cols: list[str]):
        self.cols = cols
        self.idx = {c: i for i, c in enumerate(cols)}
        fit, sc = df[df["blk"] < CUT_BLK], df[df["blk"] >= CUT_BLK]
        self.Pf = np.column_stack([fit[c].to_numpy(float) for c in cols])
        self.yf = fit["y"].to_numpy(float)
        self.Ps = np.column_stack([sc[c].to_numpy(float) for c in cols])
        self.ys = sc["y"].to_numpy(float)
        self.ya = np.abs(self.ys).sum()
        self.cut = sc["cutoff"].to_numpy()
        self.wins = sorted(set(self.cut.tolist()))

    def _pooled(self, pred: np.ndarray) -> float:
        return float(np.abs(self.ys - pred).sum() / self.ya)

    def _per_window(self, pred: np.ndarray) -> dict[str, float]:
        out = {}
        for c in self.wins:
            m = self.cut == c
            out[str(int(c))] = float(np.abs(self.ys[m] - pred[m]).sum() / np.abs(self.ys[m]).sum())
        return out

    def solo(self, c: str) -> dict:
        p = self.Ps[:, self.idx[c]]
        return {"pooled": self._pooled(p), "per_window": self._per_window(p)}

    def run(self, names: list[str]) -> dict:
        j = [self.idx[c] for c in names]
        w = fit_simplex(self.Pf[:, j], self.yf)
        pred = self.Ps[:, j] @ w
        return {
            "members": names,
            "weights": {c: round(float(x), 4) for c, x in zip(names, w, strict=True) if x > 0.005},
            "pooled": self._pooled(pred),
            "per_window": self._per_window(pred),
        }

    def blend(self, weights: dict[str, float]) -> np.ndarray:
        return sum(v * self.Ps[:, self.idx[k]] for k, v in weights.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/ensemble_full_sweep.json")
    ap.add_argument("--max-combo", type=int, default=4)
    args = ap.parse_args()

    df, missing = build(optional={"chronos_full_ft", "bitcn_wide", *FOUNDATION})
    pool = [m for m in {**POOL, **FOUNDATION} if m not in missing]
    wide = [*pool, BAG, "tree_interp"]
    S = Search(df, wide)

    print(
        f"[sweep] {len(df):,} rows | {len(S.yf):,} fit (blk<{CUT_BLK}) | {len(S.ys):,} scored "
        f"(blk>={CUT_BLK}) | {len(S.wins)} windows"
    )
    if missing:
        print(f"[sweep] NOT YET AVAILABLE (excluded): {', '.join(missing)}")
    print(f"[sweep] pool ({len(pool)}): {', '.join(pool)}")

    rep: dict = {"rows": int(len(df)), "regime": "late", "pool": pool, "missing": missing}

    # ---------------- 1. every individual number ----------------
    rep["solo"] = {m: S.solo(m) for m in wide}
    order = sorted(wide, key=lambda m: rep["solo"][m]["pooled"])
    print(f"\n  {'member':<22}{'pooled':>9}{'W0':>9}{'W1':>9}{'W2':>9}")
    for m in order:
        r = rep["solo"][m]
        pw = r["per_window"]
        tag = "  <- SHIPPED" if m in (BAG, "tree_interp") else ""
        print(
            f"  {m:<22}{r['pooled']:>9.5f}{pw['3648']:>9.5f}{pw['3312']:>9.5f}"
            f"{pw['2976']:>9.5f}{tag}"
        )

    # ---------------- 2. the incumbent, re-derived HERE ----------------
    rep["incumbent"] = S.run(["tree_interp", BAG])
    print(
        f"\n  INCUMBENT re-derived: {rep['incumbent']['weights']} -> "
        f"{rep['incumbent']['pooled']:.5f}"
    )

    # ---------------- 3. whole-pool simplexes ----------------
    for label, cols in (("KAI_POOL", pool), ("KAI_POOL_plus_shipped", wide)):
        rep[label] = S.run(cols)
        print(f"\n  simplex over {label} ({len(cols)}) -> {rep[label]['pooled']:.5f}")
        for k, v in sorted(rep[label]["weights"].items(), key=lambda kv: -kv[1]):
            print(f"      {k:<22}{v:>7.3f}")

    # ---------------- 4. exhaustive combinations ----------------
    for n in range(2, args.max_combo + 1):
        res = sorted(
            (S.run(list(c)) for c in itertools.combinations(wide, n)), key=lambda r: r["pooled"]
        )
        rep[f"best_{n}"] = res[:10]
        print(f"\n  -- top {n}-member combinations ({len(res)} searched) --")
        for r in res[:6]:
            print(f"  {r['pooled']:.5f}  {r['weights']}")

    # ---------------- 5. diversity ----------------
    scored = df[df["blk"] >= CUT_BLK]
    rep["err_corr"] = {
        m: {
            "vs_casc_bag5": float(error_correlation(scored, m, BAG)),
            "vs_tree_interp": float(error_correlation(scored, m, "tree_interp")),
        }
        for m in pool
    }
    print(f"\n  {'member':<22}{'ec vs bag':>11}{'ec vs tree':>12}")
    for m in pool:
        e = rep["err_corr"][m]
        print(f"  {m:<22}{e['vs_casc_bag5']:>11.3f}{e['vs_tree_interp']:>12.3f}")

    # ---------------- 6. is anything actually better? ----------------
    cands = {k: rep[k] for k in ("KAI_POOL", "KAI_POOL_plus_shipped")}
    cands.update({f"best_{n}": rep[f"best_{n}"][0] for n in range(2, args.max_combo + 1)})
    best_key = min(cands, key=lambda k: cands[k]["pooled"])
    best = cands[best_key]
    tmp = df.copy()
    tmp["_c"] = sum(v * tmp[k] for k, v in best["weights"].items())
    tmp["_i"] = sum(v * tmp[k] for k, v in rep["incumbent"]["weights"].items())
    bs = paired_bootstrap_delta(tmp, candidate="_c", baseline="_i", regime="late")
    sc = tmp[tmp["blk"] >= CUT_BLK]
    won = [
        int(c)
        for c, g in sc.groupby("cutoff")
        if np.abs(g["y"] - g["_c"]).sum() < np.abs(g["y"] - g["_i"]).sum()
    ]
    rep["best_vs_incumbent"] = {
        "which": best_key,
        "weights": best["weights"],
        "pooled": best["pooled"],
        "incumbent_pooled": rep["incumbent"]["pooled"],
        "delta": float(bs["delta"]),
        "se": float(bs["se"]),
        "ci95": [float(x) for x in bs["ci95"]],
        "windows_won": won,
        "note": "selected from ~1000 combinations — the CI is conditional on that selection.",
    }
    print(f"\n  BEST = {best_key} {best['weights']} -> {best['pooled']:.5f}")
    print(
        f"  vs incumbent {rep['incumbent']['pooled']:.5f}: delta {bs['delta']:+.5f} "
        f"SE {bs['se']:.5f} CI [{bs['ci95'][0]:+.5f}, {bs['ci95'][1]:+.5f}] "
        f"windows won {len(won)}/3"
    )

    with open(args.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\n[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
