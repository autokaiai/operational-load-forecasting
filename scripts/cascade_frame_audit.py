#!/usr/bin/env python3
"""Pre-flight for S2 Stage 2: is each cascade member's frame worth spending a TFT train on?

Why this exists
---------------
``cascade_provenance.check_gap_honest`` asks one question — does the covariate's context run past
the cutoff? That is a **leakage** check, and it is deliberately one-sided because a leaky covariate
scores BETTER. It says nothing about **coverage**, and Stage 2's expensive failure is the opposite
shape: a frame that is perfectly gap-honest and simply *empty* where the TFT trains.

Our Stage 1 ``_cov`` frames are horizon-only — one block, ``[cut, cut+672)``. That is all the screen
needed, because the screen never trained anything. Hand the same frame to a TFT and the channel is
NaN across the whole of ``[0, cut)``; the imputation cannot rescue it either, because
``members._load_window_long`` fits the fill on ``raw[hidx < cut]``, where the column has no value to
take a median of. Measured on ``train_tabpfn_ts_cov_cut3648.csv``: 350,208 of 350,208 train rows NaN
after the real load path, and the ``*_missing`` flag reads 1.000 over the train region against 0.000
over the horizon — a perfect train/inference separator for a channel the VSN never saw carry
information. Nothing in the stack raises. That is 4-8 GPU-hours spent to learn a gate on a constant.

So this checks three things per (member, window), all arithmetic, all free:

1. **Honesty** — ``check_gap_honest`` for every declared channel, unchanged and still fail-closed.
2. **Coverage** — the channel's first non-NaN hour must equal the earliest block start the sidecar
   declares, and the horizon block must be fully populated. Tying the artifact to its own
   provenance is stronger than a coverage threshold: it catches a frame that is merely *shorter*
   than its sidecar claims, which a fraction-based rule would wave through.
3. **Routing** — every declared channel appears in ``futr_exog_list()`` under the member's own
   ``cascade_channels``. This is the 3.10 trap in test form: ``configs/tft_chronos.yaml`` claimed a
   covariate that was never registered, so running it trained a PLAIN TFT that ignored the column
   while looking exactly like a cascade. A wasted train that reports a plausible number is worse
   than one that crashes.

    PYTHONPATH=. .venv/bin/python scripts/cascade_frame_audit.py \
        --members tft_cascade_tabpfn_ts_cov,tft_cascade_chronos_tabpfn_ts_cov --windows 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data.features import ID, cascade_channels, futr_exog_list
from src.eval.splits import SCORE_LEN
from src.models import cascade_provenance as cp
from src.models.members import get_member, member_train_csv

GAPPED_HORIZON = 2 * SCORE_LEN
CUTOFFS = [3648, 3312, 2976]


def audit_one(member: str, cut: int, *, derived_root: Path, horizon: int = GAPPED_HORIZON) -> dict:
    """Every check for one (member, cutoff). ``ok`` is the gate; ``notes`` is why."""
    spec = get_member(member)
    out: dict = {"member": member, "cut": cut, "ok": False, "channels": {}, "notes": []}
    if not spec.cascade:
        out["notes"].append("declares no cascade channel — nothing to audit")
        out["ok"] = True
        return out

    # ---------------------------------------------------------------- routing (free, no file read)
    with cascade_channels(*spec.cascade):
        futr = set(futr_exog_list())
    unrouted = [c for c in spec.cascade if c not in futr]
    if unrouted:
        out["notes"].append(
            f"NOT ROUTED: {unrouted} absent from futr_exog_list() under this member's "
            "cascade_channels — the TFT would train as a PLAIN TFT and report a plausible number"
        )
        return out

    frame = Path(member_train_csv(member, "", cut))
    if not frame.is_absolute():
        frame = derived_root / frame
    out["frame"] = str(frame)
    if not frame.exists():
        out["notes"].append(f"MISSING frame {frame}")
        return out

    usecols = [ID, *spec.cascade]
    df = pd.read_csv(frame, usecols=usecols)
    df["_h"] = df.groupby(ID).cumcount()
    n_hours = int(df.groupby(ID)["_h"].max().min()) + 1
    out["n_hours"] = n_hours

    ok = True
    for col in spec.cascade:
        info: dict = {}
        # ------------------------------------------------------------------------------- honesty
        try:
            meta = cp.check_gap_honest(frame, col, cut, horizon)
        except Exception as e:  # noqa: BLE001 — the message is the diagnostic
            info["honest"] = False
            info["error"] = f"{type(e).__name__}: {e}"
            out["channels"][col] = info
            ok = False
            continue
        info["honest"] = True
        starts = sorted(int(s) for s, _ in meta.get("blocks", []))
        info["blocks"] = len(starts)
        info["earliest_block_start"] = starts[0] if starts else None

        # ------------------------------------------------------------------------------ coverage
        present = df.loc[df[col].notna(), "_h"]
        first_covered = int(present.min()) if len(present) else None
        info["first_non_nan_hour"] = first_covered
        train = df[df["_h"] < cut]
        hz = df[(df["_h"] >= cut) & (df["_h"] < cut + horizon)]
        info["train_coverage"] = round(float(train[col].notna().mean()), 4) if len(train) else 0.0
        info["horizon_coverage"] = round(float(hz[col].notna().mean()), 4) if len(hz) else 0.0

        if info["horizon_coverage"] < 1.0:
            info["verdict"] = "horizon block INCOMPLETE — the scored rows are what we grade"
            ok = False
        elif first_covered is None:
            info["verdict"] = "channel is entirely NaN"
            ok = False
        elif first_covered >= cut:
            info["verdict"] = (
                f"HORIZON-ONLY: no coverage before hour {cut}. The TFT would train on an all-NaN "
                "channel (the fill is fitted on [0, cut), so it has no median to impute with) and "
                "nothing would raise. Regenerate with `--train-region`."
            )
            ok = False
        elif starts and first_covered != starts[0]:
            info["verdict"] = (
                f"frame and sidecar DISAGREE: first non-NaN hour {first_covered}, earliest "
                f"declared block start {starts[0]}. Not the grid its provenance claims."
            )
            ok = False
        else:
            info["verdict"] = "ok"
        out["channels"][col] = info

    out["ok"] = ok
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--members", required=True, help="comma-separated member names")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--derived-root", default=".", help="root the frame paths resolve against")
    ap.add_argument("--out", default="results/s2_stage2_frame_audit.json")
    args = ap.parse_args()

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    cuts = CUTOFFS[: args.windows]
    rows = [
        audit_one(m, cut, derived_root=Path(args.derived_root)) for m in members for cut in cuts
    ]

    width = max(len(r["member"]) for r in rows)
    for r in rows:
        flag = "ok  " if r["ok"] else "FAIL"
        print(f"[{flag}] {r['member']:<{width}} cut{r['cut']}")
        for col, info in r["channels"].items():
            print(
                f"          {col:<26} train={info.get('train_coverage')} "
                f"horizon={info.get('horizon_coverage')} blocks={info.get('blocks')} "
                f"-> {info.get('verdict', info.get('error'))}"
            )
        for n in r["notes"]:
            print(f"          {n}")

    failed = [f"{r['member']}@{r['cut']}" for r in rows if not r["ok"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows, "failed": failed}, indent=2))
    print(f"\n[audit] wrote {args.out}")
    if failed:
        print(f"[audit] {len(failed)}/{len(rows)} FAILED: {failed}")
        print("[audit] refusing to certify — a TFT train on these frames would burn GPU for noise.")
        sys.exit(1)
    print(f"[audit] all {len(rows)} (member, window) frames certified for training.")


if __name__ == "__main__":
    main()
