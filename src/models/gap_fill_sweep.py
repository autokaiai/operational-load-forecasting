"""Run every gap-fill arm for one (member, window) off a SINGLE fit.

The plan's Finding 3, made executable. ``_nf_runner`` fits on the window's train slice — where the
covariates are real — and applies the withholding only to ``futr_df`` at ``nf.predict()``. So the
gap-fill strategy is an **inference-only** change and the arms are predicts, not trains::

    naive    7 arms x 3 windows x 2 members = 42 TFT trains  (~8-16 GPU-h)
    actual   1 fit per (member, window), 7 predicts          -> 6 containers

The efficiency is the smaller half. **The correctness is the point:** every arm comes out of the
same fitted weights, so the cross-run early-stopping variance that caveated S2 Stage 2 (~0.001,
about the size of that phase's best deficit) is not merely bounded here, it is exactly **zero**. A
paired bootstrap across these arms therefore measures the covariate fill and nothing else, and the
CI means what it says without a train-noise caveat attached.

One arm is ``real`` — the genuine covariates, i.e. no withholding at all. It is the ceiling on
anything an imputer could ever buy, and it is simultaneously the true-vs-imputed delta behind
W1 and W2, open since the imputation lane started and arriving here for zero extra compute.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from src.data.features import cascade_channels
from src.eval.protocol import CUTOFF
from src.models.cascade_provenance import check_gap_honest
from src.models.members import (
    GAPPED_HORIZON,
    RunContext,
    get_member,
    load_window_long,
    member_train_csv,
    nf_fit_and_sweep,
    validate_member_frame,
)


def run_sweep(
    *,
    member: str,
    cut_idx: int,
    arms: list[str],
    train_csv: str,
    out_dirs: dict[str, Path],
    repo: Path | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    device: str = "cuda",
    nan_fill: str = "median",
    vsn_out: Path | None = None,
    overrides: dict | None = None,
    regime: str = "far",
) -> int:
    """Fit once, predict per arm, validate and write each arm's cube. Returns the number written.

    Every arm is validated through ``validate_member_frame`` exactly as ``run_member`` would, so a
    swept cube carries the same guarantees as a singly-produced one — the shared fit changes how
    the predictions are obtained, never what a cube is allowed to look like.
    """
    with _in_repo(repo):
        return _run_sweep(
            member=member,
            cut_idx=cut_idx,
            arms=arms,
            train_csv=train_csv,
            out_dirs=out_dirs,
            seed=seed,
            max_steps=max_steps,
            device=device,
            nan_fill=nan_fill,
            vsn_out=vsn_out,
            overrides=overrides,
            regime=regime,
        )


@contextmanager
def _in_repo(repo: Path | None):
    """Run with cwd = the repo root.

    Every relative path in the codebase -- ``configs/base.yaml``, ``data/derived/...``, a member's
    declared ``train_csv``, a gap-fill strategy's frame template -- is repo-relative, and the
    single assumption that makes them all work is cwd. ``run_member`` gets that for free because
    Modal reaches it through a subprocess launched with ``cwd=repo``; this path runs IN-PROCESS,
    where the container's cwd is not the repo. Fixing it once here beats discovering each relative
    path in turn, which is exactly how the first two smoke runs failed.
    """
    if repo is None:
        yield
        return
    prev = os.getcwd()
    os.chdir(str(repo))
    try:
        yield
    finally:
        os.chdir(prev)


def _run_sweep(
    *,
    member: str,
    cut_idx: int,
    arms: list[str],
    train_csv: str,
    out_dirs: dict[str, Path],
    seed: int | None,
    max_steps: int | None,
    device: str,
    nan_fill: str,
    vsn_out: Path | None = None,
    overrides: dict | None = None,
    regime: str = "far",
) -> int:
    spec = get_member(member)
    if spec.kind != "neural":
        raise ValueError(
            f"member {member!r} is {spec.kind!r}, not neural — only an nf member has the "
            "fit-once/predict-many structure this sweep exists for."
        )
    if not spec.honours_gap_cov:
        raise ValueError(
            f"member {member!r} does not read the gap block, so every arm would be identical "
            "and the sweep would report a difference of exactly zero as if it were a result."
        )

    resolved = member_train_csv(member, train_csv, cut_idx)
    long_df = load_window_long(resolved, cut_idx, member=member, nan_fill=nan_fill)
    ctx = RunContext(
        long_df=long_df,
        cut_idx=cut_idx,
        seed=seed,
        device=device,
        max_steps=max_steps,
        regime=regime,
    )
    # The member's OWN config, read off its runner, not guessed from its name: `tft_cascade` is
    # driven by configs/tft_chronos.yaml, and guessing configs/tft_cascade.yaml would either fail
    # loudly or -- worse -- silently fit a different model than the registry declares.
    cfg_path = getattr(spec.run, "member_config", None) or f"configs/{member}.yaml"

    print(
        f"[sweep] {member} cut={cut_idx} nan_fill={nan_fill}: one fit, {len(arms)} arms {arms}",
        flush=True,
    )
    # The same scoping and the same fail-closed honesty check `run_member` applies. This path
    # bypasses run_member (it returns N cubes, not one), so the guards are replicated rather than
    # inherited — and a cascade channel that is not ACTIVE here would silently drop out of
    # futr_exog_list(), fitting a plain TFT that ignores the covariate entirely.
    with cascade_channels(*spec.cascade):
        if spec.cascade:
            absent = [c for c in spec.cascade if c not in set(long_df.columns)]
            if absent:
                raise AssertionError(
                    f"{member}: cascade column(s) {absent} are not in the frame — load from "
                    f"{spec.train_csv!r}, not the raw train.csv."
                )
            for channel in spec.cascade:
                check_gap_honest(resolved, channel, cut_idx, GAPPED_HORIZON)
        cubes = nf_fit_and_sweep(member, cfg_path, ctx, [(a, a) for a in arms], vsn_out, overrides)

    written = 0
    for arm, cube in cubes.items():
        cube = cube.copy()
        cube[CUTOFF] = cut_idx
        cube = validate_member_frame(cube, member, cut_idx)
        out_dir = Path(out_dirs[arm])
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{member}_preds.csv"
        cube.to_csv(path, index=False)
        print(f"[sweep]   {arm:<24} -> {path}", flush=True)
        written += 1
    return written
