"""Produce one member's scored-block predictions for one rolling gapped window.

This is the per-(member, window) unit of work the ensemble architecture test fans out (see
`tools/modal_ensemble.py` and the plan). It writes a tidy CSV the combiner can merge:

    results/window{W}/<member>_preds.csv   columns: unique_id, ds, cutoff, y, <member>

**How members are dispatched.** They are not, here. This script is a thin CLI over
``src.models.members``: the registry owns what each member is, how to run it, whether it can honour
a seed, and whether it needs a GPU or a LoRA adapter. Adding a member means registering it there,
not adding a branch here — which is the point, since Phase 5 Tier A adds four.

Leakage discipline (plan pitfall #5): imputation stats are fitted on the window's OWN train slice
(``_hidx < cut_idx``) by ``load_window_long`` and applied to the whole frame — never fitted across
the scored block.

    python -m scripts.member_preds_window --member tft --window 0 --cut-idx 3648
    python -m scripts.member_preds_window --member chronos_ft --window 1 --cut-idx 3312 \
        --adapter checkpoints/chronos2_ft_w1
    python -m scripts.member_preds_window --members            # list the registry and exit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import NF_ID, NF_TARGET
from src.models.members import (
    RunContext,
    available_members,
    get_member,
    load_window_long,
    member_train_csv,
    run_member,
)

RESULTS_DIR = Path("results")


def print_registry() -> None:
    print(f"{'member':<18}{'kind':<12}{'status':<11}{'seed':<7}{'gpu':<6}{'gap-cov':<9}note")
    for name in available_members():
        s = get_member(name)
        print(
            f"{s.name:<18}{s.kind:<12}{s.status:<11}"
            f"{'yes' if s.seedable else 'no':<7}{'yes' if s.needs_gpu else 'no':<6}"
            f"{'yes' if s.honours_gap_cov else 'n/a':<9}{s.note}"
        )


def _aggregate_scope(names: list[str]):
    """Declare the aggregate columns for the duration of the run, or a no-op when there are none."""
    import contextlib

    if not names:
        return contextlib.nullcontext()
    from src.data.features import aggregate_columns

    return aggregate_columns(names)


def _run(args, long_df):
    return run_member(
        args.member,
        RunContext(
            long_df=long_df,
            cut_idx=args.cut_idx,
            seed=args.seed,
            gap_cov=args.gap_cov,
            adapter=args.adapter,
            device=args.device,
            batch_series=args.batch_series,
            max_steps=args.max_steps or None,
            gap_fill=args.gap_fill,
            regime=args.regime,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="One member's scored-block preds for one window.")
    ap.add_argument("--member", help=f"one of: {', '.join(available_members())}")
    ap.add_argument("--members", action="store_true", help="list the member registry and exit")
    ap.add_argument("--window", type=int, help="window index W (output subdir)")
    ap.add_argument("--cut-idx", type=int, help="train-end _hidx for this window")
    ap.add_argument("--train_csv", default="data/raw/train.csv")
    ap.add_argument("--adapter", default=None, help="LoRA dir (chronos_ft only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-series", type=int, default=0)
    ap.add_argument("--limit-series", type=int, default=0, help="first N series only (smoke)")
    ap.add_argument("--max-steps", type=int, default=0, help="override nf max_steps (smoke)")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="explicit member seed; omit to use the member's own default (42). Seedable members "
        "only — a deterministic member raises rather than ignore it.",
    )
    ap.add_argument(
        "--gap-cov",
        choices=["real", "impute"],
        default="impute",
        help="gap covariate condition: impute (median #32 floor) | real (train.csv)",
    )
    ap.add_argument(
        "--gap-fill",
        default="median",
        help="how the GAP block's absent covariates are reconstructed (src.data.gap_fill): "
        "median (incumbent) | how168 | hod24 | chronos2 | ... , optionally +exact. Inference-only, "
        "so arms share identical weights.",
    )
    ap.add_argument(
        "--nan-fill",
        default="median",
        help="how the SCATTERED ~4.5%% NaNs are reconstructed: median (incumbent) | interp | "
        "ffill | how168. This surface reaches EVERY member, including the tree.",
    )
    ap.add_argument(
        "--regime",
        choices=["far", "near"],
        default="far",
        help="which half of the 672h forecast becomes the cube: far (steps 337-672, the graded "
        "private-test block — the DEFAULT and what every recorded number is) or near (steps "
        "1-336, the public leaderboard's scenario). near is an OPT-IN DIAGNOSTIC: no weight, no "
        "member and no submission decision may be made on a near-regime number.",
    )
    ap.add_argument("--out-dir", default=None, help="default results/window{W}")
    # SPRINT 2. Materialise one cross-series aggregate block onto the frame AND declare its columns
    # in `futr_exog_list()`, so the block reaches whichever member is being run. Both halves are
    # required: declaring without materialising raises KeyError downstream (loud, fine), and
    # materialising without declaring is a silent no-op — plan law 5, and the reason
    # `src.data.features.aggregate_columns` mutates STATE rather than rebinding the function.
    ap.add_argument(
        "--xs-block",
        default=None,
        help="cross-series block, e.g. A1 or A9 (scripts.xs_aggregate_screen)",
    )
    args = ap.parse_args()

    if args.members:
        print_registry()
        return
    for required in ("member", "window", "cut_idx"):
        if getattr(args, required) is None:
            ap.error(f"--{required.replace('_', '-')} is required (or pass --members to list)")

    spec = get_member(args.member)  # fails fast on an unknown name, before loading 326MB of CSV
    if not spec.honours_gap_cov and args.gap_cov == "real":
        print(
            f"[member_preds_window] note: {spec.name} ignores --gap-cov (no gap traversal), so "
            "'real' and 'impute' produce identical output."
        )

    train_csv = member_train_csv(args.member, args.train_csv, args.cut_idx)
    if train_csv != args.train_csv:
        print(f"[member_preds_window] {spec.name} needs its derived frame: {train_csv}")
    long_df = load_window_long(train_csv, args.cut_idx, member=args.member, nan_fill=args.nan_fill)
    if args.limit_series > 0:
        keep = sorted(long_df[NF_ID].unique())[: args.limit_series]
        long_df = long_df[long_df[NF_ID].isin(keep)]

    xs_names: list[str] = []
    if args.xs_block:
        from src.data.xs_blocks import BLOCKS, NEEDS_CUT, _zone_labels
        from src.eval.splits import add_hour_index

        desc, fn = BLOCKS[args.xs_block]
        long_df = add_hour_index(long_df)
        zones = _zone_labels(long_df)
        long_df, xs_names = (
            fn(long_df, zones, args.cut_idx) if args.xs_block in NEEDS_CUT else fn(long_df, zones)
        )
        missing = [c for c in xs_names if c not in long_df.columns]
        if missing:
            raise SystemExit(f"[xs] block {args.xs_block} did not materialise {missing}")
        if long_df[xs_names].isna().any().any():
            # The aggregates are built from an ALREADY-IMPUTED frame, so a NaN here means the
            # aggregation itself introduced one (a shift, a divide) and it would reach the model.
            bad = long_df[xs_names].isna().sum()
            raise SystemExit(f"[xs] block {args.xs_block} produced NaN:\n{bad[bad > 0]}")
        print(f"[xs] {args.xs_block} ({desc}): {len(xs_names)} columns -> {args.member}")

    with _aggregate_scope(xs_names):
        if xs_names:
            # LAW 5, checked INSIDE the scope and on the real conditioning set rather than on the
            # state variable we just set. `registry` binds `futr_exog_list` by value at import, so
            # only state can reach it; if that ever regresses to a rebinding this fails here rather
            # than returning a confident null after a full GPU fit.
            from src.data.features import futr_exog_list

            gone = [c for c in xs_names if c not in futr_exog_list()]
            if gone:
                raise SystemExit(
                    f"[xs] {gone} never reached futr_exog_list(); the arm would be a no-op"
                )
        out = _run(args, long_df)

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / f"window{args.window}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.member}_preds.csv"
    out.to_csv(out_csv, index=False)
    from src.metrics import wape

    w = wape(out[NF_TARGET], out[args.member])
    seed_note = f" seed={args.seed}" if args.seed is not None else ""
    if args.regime != "far":
        seed_note += f" regime={args.regime}"
    print(
        f"[member_preds_window] {args.member} W{args.window} cut={args.cut_idx}{seed_note}: "
        f"{len(out)} rows, WAPE={w:.4f} -> {out_csv}"
    )


if __name__ == "__main__":
    main()
