"""Admission probe — does adding a candidate member to the incumbent blend earn its place?

Runs the protocol in ``docs/protocol.md`` end to end for any candidate:

  1. **Error correlation** on the gapped OOF residuals (pred - y) against every incumbent.
     Orthogonality screen = < 0.95.
  2. **Blend gain.** Fit convex weights on the EARLY block (``blk < CUT``) only, score on the
     held-out block (``blk >= CUT``), POOL across the rolling gapped windows. Incumbent blend vs
     incumbent+candidate.
  3. **The three soft checks** in ``src.eval.protocol.evaluate_admission`` — pooled improvement,
     2-of-3 windows, orthogonality — plus the paired bootstrap CI, per-window deltas and their
     spread as *reported evidence*. **Nothing is admitted or dropped automatically.** The call is
     human and informed; turning a dispersion into a threshold is how both retired bars failed.

Generic over the member registry: it began with ``tft``, ``chronos_ft`` and ``lgbm`` written into
``main()``, and became ``--incumbents`` / ``--candidate`` once there were four more members to put
through it.

Reads the cached gapped member OOF cube (``results/impute/window{0,1,2}_preds.csv``; columns:
unique_id, ds, y, then one column per member). Pure CPU; no GPU.

    PYTHONPATH=. python scripts/member_admission.py --candidate lgbm --incumbents tft,chronos_ft

RECORDED RESULT for the LGBM 3-way case (2026-06-10, board #44): err-corr(LGBM,FT-Chronos)=0.93,
err-corr(LGBM,TFT)=0.77; held-out pooled WAPE TFT-alone 0.1463 -> pair 0.1427 -> 3-way 0.1411
(+1.1%). LGBM weight 0.25/0.00/0.10 across W0/W1/W2 — the gain is concentrated in W0, which is also
the only window where LGBM is close to TFT standalone. Under the retired 19% bar this was recorded
"below bar"; under the adopted rule it is a real, small, window-heterogeneous gain.

CAVEAT — the cached cube still draws the 336h gap covariates from train.csv (the leaky path). Plan
3.5 regenerates it covariate-absent (#48), which will raise the absolute WAPEs and may shift both
the err-corrs and the blend weights. LightGBM's own numbers do not move (it never traverses the
gap), but its *relative* standing does. Re-run after 3.5 before treating any verdict as final.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scripts.ensemble_metalearner import _wstats, best_convex_pair, best_simplex

from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.protocol import CUT_BLK, evaluate_admission

CUTOFFS = {0: 3648, 1: 3312, 2: 2976}  # train-end _hidx per window (src.eval.splits)


def load(preds_dir: str, w: int) -> pd.DataFrame:
    d = pd.read_csv(f"{preds_dir}/window{w}_preds.csv").sort_values([NF_ID, NF_TIME])
    d["blk"] = d.groupby(NF_ID).cumcount()
    return d


def fit_score(
    frames: dict[int, pd.DataFrame], members: list[str], cut_blk: int
) -> tuple[float, list]:
    """Fit weights on the early block, score on the held-out block, pooled across windows."""
    te_y, te_p, perwin = [], [], []
    for w, f in frames.items():
        tr = f["blk"].to_numpy() < cut_blk
        te = f["blk"].to_numpy() >= cut_blk
        y = f[NF_TARGET].to_numpy()
        if len(members) == 1:
            pred = f[members[0]].to_numpy()[te]
            wts = {members[0]: 1.0}
        elif len(members) == 2:
            a, b = (f[m].to_numpy() for m in members)
            wt, _ = best_convex_pair(y[tr], a[tr], b[tr])
            pred = wt * a[te] + (1 - wt) * b[te]
            wts = {members[0]: round(wt, 3), members[1]: round(1 - wt, 3)}
        else:
            wv, _ = best_simplex(y[tr], members, f.iloc[np.where(tr)[0]])
            pred = np.column_stack([f[m].to_numpy()[te] for m in members]) @ np.array(wv)
            wts = {m: round(float(x), 3) for m, x in zip(members, wv, strict=True)}
        perwin.append((w, _wstats(y[te], pred)["wape"], wts))
        te_y.append(y[te])
        te_p.append(pred)
    pooled = _wstats(np.concatenate(te_y), np.concatenate(te_p))["wape"]
    return pooled, perwin


def main() -> None:
    ap = argparse.ArgumentParser(description="Admission probe for one candidate member.")
    ap.add_argument("--candidate", default="lgbm", help="the member under test")
    ap.add_argument(
        "--incumbents", default="tft,chronos_ft", help="comma-separated current blend members"
    )
    ap.add_argument("--baseline", default=None, help="standalone baseline (default: 1st incumbent)")
    ap.add_argument("--preds-dir", default="results/impute", help="dir with window{W}_preds.csv")
    ap.add_argument("--windows", default="0,1,2")
    ap.add_argument("--cut-blk", type=int, default=CUT_BLK, help="early/held-out block boundary")
    ap.add_argument("--regime", default="late", choices=["full", "late"])
    args = ap.parse_args()

    incumbents = [m.strip() for m in args.incumbents.split(",") if m.strip()]
    candidate = args.candidate
    baseline = args.baseline or incumbents[0]
    windows = [int(w) for w in args.windows.split(",")]

    frames = {w: load(args.preds_dir, w) for w in windows}
    allf = pd.concat(frames.values(), ignore_index=True)
    have = set(allf.columns)
    missing = [m for m in (*incumbents, candidate, baseline) if m not in have]
    if missing:
        raise SystemExit(
            f"{args.preds_dir} has no column(s) for {missing}. Present members: "
            f"{sorted(have - {NF_ID, NF_TIME, NF_TARGET, 'blk'})}. Generate them with "
            "scripts/member_preds_window.py."
        )

    print(f"== Error correlation on gapped OOF residuals (screen < 0.95), candidate={candidate} ==")
    late_mask = allf["blk"] >= args.cut_blk
    for label, mask in (("ALL rows", None), (f"blk >= {args.cut_blk}", late_mask)):
        s = allf if mask is None else allf[mask]
        res = {m: s[m].to_numpy() - s[NF_TARGET].to_numpy() for m in {*incumbents, candidate}}
        pairs = "   ".join(
            f"err-corr({candidate},{m})={np.corrcoef(res[candidate], res[m])[0, 1]:.3f}"
            for m in incumbents
        )
        print(f"  [{label}] {pairs}")

    print(f"\n== Held-out pooled WAPE (early-fit / late-scored, pooled over W{windows}) ==")
    base_pooled, _ = fit_score(frames, [baseline], args.cut_blk)
    inc_pooled, _ = fit_score(frames, incumbents, args.cut_blk)
    new_pooled, pw = fit_score(frames, [*incumbents, candidate], args.cut_blk)
    print(f"  {baseline} alone:{'':<12}{base_pooled:.4f}")
    print(f"  incumbent blend:{'':<12}{inc_pooled:.4f}   <- admission baseline")
    print(f"  + {candidate}:{'':<{max(1, 18 - len(candidate))}}{new_pooled:.4f}")
    for w, wape, wts in pw:
        print(f"      W{w}: held-out WAPE={wape:.4f}  weights={wts}")

    gain = (inc_pooled - new_pooled) / inc_pooled
    cand_w = [wts.get(candidate, 0) for _, _, wts in pw]
    print(
        f"\n== Blend gain: {inc_pooled:.4f} -> {new_pooled:.4f} "
        f"({gain:+.1%} relative, {inc_pooled - new_pooled:+.4f} absolute)"
    )
    print(f"   {candidate} weight per window: {cand_w}")

    # Soft checks on the standalone member, via the shared protocol. Reported, never enforced.
    cube = pd.concat(
        [frames[w].assign(cutoff=CUTOFFS[w]) for w in sorted(frames)], ignore_index=True
    ).sort_values(["cutoff", NF_ID, NF_TIME])
    report = evaluate_admission(
        cube,
        cube,
        candidate=candidate,
        baseline=baseline,
        regime=args.regime,
        incumbents=incumbents,
    )
    print(f"\n== Standalone soft checks (src.eval.protocol, regime={args.regime}) ==")
    print(f"   {report['summary']}")
    for name, chk in report["checks"].items():
        mark = "PASS" if chk["pass"] else "miss" if chk["pass"] is False else " n/a"
        print(f"   [{mark}] {name}")
    boot = report["evidence"].get("paired_bootstrap") or {}
    if boot:
        lo, hi = boot["ci95"]
        print(
            f"   evidence: paired delta {boot['delta']:+.4f} +- {boot['se']:.4f} "
            f"(95% CI [{lo:+.4f}, {hi:+.4f}], {boot['n_blocks']} blocks)"
        )
    print(f"   per-window delta: {report['evidence']['per_window_delta']}")
    print(f"   {report['note']}")


if __name__ == "__main__":
    main()
