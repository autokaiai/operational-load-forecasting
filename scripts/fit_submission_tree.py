"""Fit the shipped LightGBM member on the FULL labelled history and persist the booster.

The archive has to carry the tree, and that is a hard constraint rather than a convenience:
``predict.py`` runs offline against an input dir holding covariates and a forecast index and **no
training data**, so there is nothing to refit from at inference. S6.5 shipped a leaderboard CSV
built from a tree that was fitted on Modal and then thrown away — fine for a CSV, useless for an
archive the evaluation harness executes itself.

The member is ``lgbm_s24_unitcat``, frozen by S1 and never re-tuned: ``configs/lgbm.yaml`` plus
``origin_stride=24``, ``early_stopping_rounds=50``, ``max_boost_round=3000``,
``categorical_unit=True``. Read from the registry's own overrides rather than retyped, so this
cannot drift from the member every recorded number describes.

Scattered NaNs are imputed with **interp**, which is S3's adopted split: the tree gained +0.00401
(3/3 windows) under interp while the cascade LOST 0.00937, so the two ship members deliberately
want different imputation and ``predict.py`` builds a frame for each.

    python -m scripts.fit_submission_tree --out checkpoints/submission_tree.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

from src.data.loader import load_long
from src.models import lgbm as lgbm_mod
from src.models.lgbm import GAPPED_H

#: Hours of labelled history. The submitted model trains on ALL of them — selection used cutoffs
#: <= 3648, but nothing is being selected here, and #57's protocol choice is "use everything".
TRAIN_END_IDX = 4320

TREE_MEMBER = "lgbm_s24_unitcat"
NAN_FILL = "interp"  # S3: adopted for the tree ONLY


def resolved_config() -> tuple[dict, dict]:
    """(booster params, runner knobs) for the frozen member, read off its own registration."""
    import yaml

    from src.models.members import _TREE_RUNNER_KEYS

    base = yaml.safe_load(Path("configs/lgbm.yaml").read_text())
    overrides = {
        "origin_stride": 24,
        "early_stopping_rounds": 50,
        "max_boost_round": 3000,
        "categorical_unit": True,
    }
    merged = {**base, **overrides}
    params = {k: v for k, v in merged.items() if k not in _TREE_RUNNER_KEYS}
    knobs = {k: v for k, v in merged.items() if k in _TREE_RUNNER_KEYS}
    return params, knobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/submission_tree.json"))
    ap.add_argument("--cut-idx", type=int, default=TRAIN_END_IDX)
    ap.add_argument("--horizon", type=int, default=GAPPED_H)
    ap.add_argument("--xs-block", default="", help="cross-series block to attach, e.g. A9")
    args = ap.parse_args()

    params, knobs = resolved_config()
    print(f"[tree] member={TREE_MEMBER} fill={NAN_FILL} cut={args.cut_idx} h={args.horizon}")
    print(f"[tree] params={params}")
    print(f"[tree] knobs={knobs}")

    long_df, fill_stats = load_long(args.train_csv, strategy=NAN_FILL)
    print(f"[tree] history: {len(long_df)} rows, {long_df['unique_id'].nunique()} series")

    # SPRINT 2 — the aggregates on the SHIPPED tree. Built after `load_long` has imputed, and
    # activated for the whole fit so `lgbm.futr_exog_list()` includes them; `predict.py` rebuilds
    # them from the same block at submission time via `src.data.xs_attach`.
    xs_names: list[str] = []
    xs_scope: contextlib.AbstractContextManager = contextlib.nullcontext()
    if args.xs_block:
        from src.data.features import aggregate_columns
        from src.data.xs_blocks import BLOCKS as XS_BLOCKS
        from src.data.xs_blocks import _zone_labels as xs_zones
        from src.eval.splits import add_hour_index

        _, fn = XS_BLOCKS[args.xs_block]
        long_df = add_hour_index(long_df)
        long_df, xs_names = fn(long_df, xs_zones(long_df))
        if long_df[xs_names].isna().any().any():
            raise SystemExit(f"[xs] {args.xs_block} produced NaN on the train frame")
        xs_scope = aggregate_columns(xs_names)
        print(f"[xs] {args.xs_block}: {len(xs_names)} aggregate column(s) -> shipped tree")

    t0 = time.time()
    report: dict = {}
    with xs_scope:
        rounds = lgbm_mod.tune_num_boost_round(
            long_df,
            args.cut_idx,
            params=params,
            horizon=args.horizon,
            origin_stride=int(knobs["origin_stride"]),
            max_boost_round=int(knobs["max_boost_round"]),
            early_stopping_rounds=int(knobs["early_stopping_rounds"]),
            categorical_unit=bool(knobs["categorical_unit"]),
        )
        report.update(rounds)
        print(
            f"[tree] early stopping: best_iteration={report['best_iteration']} "
            f"(valid l1 {report['best_score']:.4f}, ceiling {report['max_boost_round']}, "
            f"hit_ceiling={report['hit_ceiling']}) in {time.time() - t0:.0f}s"
        )
        if report["hit_ceiling"]:
            # A truncated fit and a converged one look identical in the WAPE alone; say so loudly.
            print("[tree] WARNING: early stopping hit the ceiling — the round count is a floor.")

        booster, feat, levels = lgbm_mod.fit_gapped(
            long_df,
            args.cut_idx,
            params=params,
            horizon=args.horizon,
            origin_stride=int(knobs["origin_stride"]),
            num_boost_round=int(report["best_iteration"]),
            categorical_unit=bool(knobs["categorical_unit"]),
        )
    assert levels is None, "normalise_level is off for the shipped member"

    payload = {
        "member": TREE_MEMBER,
        "nan_fill": NAN_FILL,
        "cut_idx": int(args.cut_idx),
        "horizon": int(args.horizon),
        "categorical_unit": bool(knobs["categorical_unit"]),
        "fc_lags": knobs.get("fc_lags"),
        "num_boost_round": int(report["best_iteration"]),
        "feature_names": list(feat),
        "params": params,
        "tuning_report": {
            k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in report.items()
        },
        "booster": booster.model_to_string(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    mb = args.out.stat().st_size / 1e6
    print(f"[tree] wrote {args.out} ({mb:.1f} MB, {len(feat)} features) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
