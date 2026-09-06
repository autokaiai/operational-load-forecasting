#!/usr/bin/env python3
"""Summarise results/*.json into one comparison table.

    python scripts/summarize_results.py                 # all results/*.json
    python scripts/summarize_results.py --sort train_wape

Reads each per-model results JSON written by ``src.train`` and prints a single table. Works for
both full-diagnostics runs (CV + gapped WAPE) and ``--skip-diagnostics`` runs (in-sample
``train_wape`` only). Files without a top-level "model" key (e.g. results/baselines.json) are
skipped. Nothing is recomputed here — this only tabulates what the JSONs already hold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")


def _row(data: dict) -> dict:
    """Flatten one results JSON to the columns we print (missing fields -> None)."""
    cv = data.get("cross_validation") or {}
    gapped = (data.get("gapped_eval") or {}).get("gapped_metrics") or {}
    return {
        "name": data.get("name", "?"),
        "model": data.get("model", "?"),
        "cv_wape": cv.get("cv_wape_mean"),
        "cv_std": cv.get("cv_wape_std"),
        "gapped_wape": gapped.get("wape"),
        "train_wape": data.get("train_wape"),
        "params": data.get("n_params"),
        "train_s": data.get("train_seconds"),
    }


def _fmt(x, spec: str = ".4f") -> str:
    """Format a number, or '-' when it's missing (None)."""
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description="Tabulate results/*.json.")
    ap.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    ap.add_argument(
        "--sort",
        default="auto",
        choices=["auto", "cv_wape", "gapped_wape", "train_wape"],
        help="auto = CV WAPE when present, else train WAPE.",
    )
    args = ap.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if "model" not in data:  # e.g. baselines.json — different schema
            continue
        rows.append(_row(data))

    if not rows:
        print(f"No model result JSONs found in {args.results_dir}/")
        return

    def sort_key(r: dict) -> float:
        v = (
            (r["cv_wape"] if r["cv_wape"] is not None else r["train_wape"])
            if args.sort == "auto"
            else r[args.sort]
        )
        return float("inf") if v is None else float(v)

    rows.sort(key=sort_key)

    header = (
        f"{'name':<14}{'model':<14}{'CV WAPE':>16}"
        f"{'gapped':>9}{'train':>9}{'params':>13}{'train_s':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        cv = "-"
        if r["cv_wape"] is not None:
            std = f"+/-{r['cv_std']:.3f}" if r["cv_std"] is not None else ""
            cv = f"{r['cv_wape']:.4f}{std}"
        params = f"{r['params']:,}" if isinstance(r["params"], int) else "-"
        gp, tr = _fmt(r["gapped_wape"]), _fmt(r["train_wape"])
        ts = _fmt(r["train_s"], ".1f")
        print(f"{r['name']:<14}{r['model']:<14}{cv:>16}{gp:>9}{tr:>9}{params:>13}{ts:>9}")

    if any(r["cv_wape"] is None for r in rows):
        print(
            "\nNote: rows without CV WAPE were run with --skip-diagnostics; their 'train' "
            "column is IN-SAMPLE (optimistic, not a generalization estimate)."
        )
        print("Promote finalists to full CV before pinning an architecture.")


if __name__ == "__main__":
    main()
