"""Assemble ``final_submission.zip`` — the graded artifact, in the prescribed layout.

The submission contract fixes both the contents and the command:

    final_submission.zip = predict.py  requirements.txt  checkpoint.pt  src/
    python predict.py --input_dir /data/input --output_file /output/predictions.csv \
        --checkpoint /submission/checkpoint.pt

``README.md`` rides along beside those four. It is not in the template's list, but the contract
requires it separately -- *"Document training and inference steps in your README"* -- and the
archive is read on its own, so that is where it belongs. Source of truth is ``docs/method.md``,
copied in rather than generated, so it is reviewable in git.

Two stages, because they fail differently:

1. **bundle** — fold the persisted LightGBM booster and S6's blend weight into the neural bundle,
   producing a single ``checkpoint.pt``. This is a load/re-save ROUND TRIP of existing weights, not
   a refit: S6 §1b measured that the same seed on a different card produces a *different model*
   (max|dpred| 13.40, as far apart as a fresh seed), so the shipped weights must be the ones that
   were fitted, moved rather than remade.
2. **package** — zip exactly the prescribed entries plus the README, assert nothing hidden or
   OS-generated crept into ``src/``, and assert the result is under the Space's 200 MB per-file cap
   before anyone tries to upload it.

``configs/`` is deliberately NOT shipped: the template does not list it, and nothing on the
inference path reads a YAML (checked — the only ``yaml.safe_load`` in the reachable modules sits
inside a CLI ``main()``).

    python -m scripts.make_submission --smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

#: S6's fitted weight: 0.24*lgbm_s24_unitcat(interp) + 0.76*tft_cascade_bag5(median) = 0.13162 late.
#: Fitted on `blk < 224` and scored on `blk >= 224`, so the number it buys was never fitted on.
TREE_WEIGHT = 0.24

#: The submission host's per-file upload limit. A published clarification (2026-06-07) sanctions
#: hosting oversized weights elsewhere, which is why Chronos-2's 456 MB is fetched rather than
#: shipped — but everything we DO ship has to clear this.
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024

INFERENCE_REQUIREMENTS = """\
# Inference-only dependencies for the graded run. Deliberately narrower than the repo's
# requirements.txt: catboost / scikit-learn / optuna are training-time only.
torch>=2.2
pandas>=2.2
numpy>=1.26
pyyaml>=6
# The neural bag. Pinned to the versions the shipped weights were fitted and validated against —
# src.bundle asserts the version on load and warns on a mismatch.
neuralforecast==3.1.9
pytorch-lightning==2.5.6
utilsforecast==0.2.16
# The blended tree member (0.186 of the shipped prediction), restored from a serialised booster.
lightgbm>=4.0
# TWO sets of Chronos-2 weights are fetched at inference, both PUBLIC, both at pinned revisions,
# and NEITHER in this archive (455.8 MB each, against a 200 MB cap):
#   * `amazon/chronos-2`                     — zero-shot, supplies the cascade covariate
#   * `autokai/chronos2-fullft-dlam-g100`    — our full fine-tune, a 0.472-weight blend member
# If either fetch is impossible, predict.py steps DOWN a defined ladder rather than failing: losing
# the fine-tune gives the previously-submitted two-member model at its own measured weights, and
# losing both gives the offline arm. See src/models/{fullft,cascade}_inference.py.
chronos-forecasting>=2.0
huggingface_hub>=0.24
"""


#: S10's three-member fit: weights on `blk < 224`, scored on `blk >= 224`, pooled over the three
#: CV windows -> 0.12722 against rung 2's 0.13163 (+0.00441, SE 0.00049, 9.1 SE, 3/3 windows).
#: G1 seed-replicated the full-FT member (sigma 0.00233, n=3) and found this draw CONSERVATIVE:
#: the seed mean scores 0.12680. Asserted against the cubes by tests/test_s10_blend.py.
FULLFT_WEIGHTS = {
    # SPRINT 2. Every member carries the A9 cross-series aggregates; weights refit on `blk < 224`
    # and scored on `blk >= 224` over the three gapped CV windows.
    #   0.12722 -> 0.12029, delta +0.00693, SE 0.00047, CI95 (+0.00601, +0.00793), 14.7 SE, 3/3.
    # Confirmed by a 182-combination search over 8 candidate members: the best alternative gains
    # +0.00007 (one seventh of an SE) by adding a fourth member and is rejected.
    #
    # *** THE EWMA TREE WAS TESTED HERE AND REJECTED (2026-09-04). ***
    # A handoff proposed replacing `tree_A9` with `tree_EWA9` (A9 + backward EWMA, 35 columns) at
    # weights 0.4289/0.1577/0.4134 for a claimed +0.00189. Rebuilt on OUR cube, the member gain
    # REPLICATES (`tree_A9` 0.14284 -> `tree_EWA9` 0.12915, +0.01369 at 6.9 SE) but the BLEND gain
    # does not: with the amplitude recalibration applied to both, shipped 0.11807 vs EWMA 0.11808,
    # i.e. **-0.00001 at 0.0 SE, 1/3 windows** — a null on every admission criterion.
    #
    # THE DISAGREEMENT IS THEIR BASELINE, NOT THEIR ARITHMETIC. Their shipped blend is 0.12216
    # against our 0.12029, because their third member is `casc_bag5` WITHOUT A9. A weaker neural
    # side leaves room a sharper tree can fill, which is why their refit gives the tree 0.4134 and
    # ours gives it 0.2904. On our blend the other two members already carry what EWMA supplies.
    #
    # THE GENERAL LESSON, and it is the third instance today: a member can improve substantially and
    # add NOTHING to a blend when its improvement is correlated with what the blend already holds.
    "chronos_full_ft": 0.4723535251605685,
    "cascade_bag": 0.3413176080096898,
    # Set by subtraction so the three sum to EXACTLY 1.0: `tests/test_s10_blend.py` asserts a
    # convex combination at 1e-9, and it caught a 3-decimal rounding that summed to 0.999.
    "tree": 0.1863288668297417,
    "source": "Sprint 2 three-member simplex fit, all members carrying A9",
    "cv_pooled_wape": 0.12029,
}


def bundle_checkpoint(nf_ckpt: Path, tree_json: Path | None, out: Path, weight: float) -> Path:
    """Round-trip the neural bundle, folding in the booster and the blend weight."""
    from src import bundle

    nf, sidecar = bundle.load(nf_ckpt)
    cfg = dict(sidecar.get("config") or {})
    cfg.setdefault("model", sidecar["model"])
    cfg.setdefault("seed", sidecar.get("seed", 42))

    tree = json.loads(tree_json.read_text()) if tree_json and tree_json.exists() else None
    blend = {"tree_weight": weight if tree else 0.0, "source": "S6 final weight fit"}
    if tree:
        # RUNG 1's weights travel WITH the checkpoint, and they are a SEPARATE measurement from
        # `tree_weight` rather than a refinement of it. Rung 2 keeps its own 0.2438/0.7562 so that
        # losing the full-FT fetch degrades to a model we have actually scored (0.13163) instead of
        # to rung 1's weights re-spread over two members, which nothing measures.
        blend["fullft"] = dict(FULLFT_WEIGHTS)
    if tree is None:
        print("[submission] NO TREE — shipping the neural bag alone (weight 0.00)")

    bundle.save(nf, sidecar["fill_stats"], cfg, out, tree=tree, blend=blend)
    print(f"[submission] checkpoint.pt: {out.stat().st_size / 1e6:.1f} MB")
    return out


#: Editor/OS droppings that a recursive walk will happily ship. None of these exist in the repo
#: today; the filter is here because the cost of one appearing is an archive that advertises a
#: `.DS_Store` in the archive, and the cost of the filter is a `startswith`.
JUNK_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini", ".gitkeep", ".gitignore"})


def _shippable(path: Path) -> bool:
    """A file belongs in `src/` only if it is source. Caches, bytecode and dotfiles are not."""
    if not path.is_file():
        return False
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        return False
    # Dotfiles anywhere in the path, not just the leaf: a stray `.ipynb_checkpoints/` directory
    # would otherwise contribute several perfectly ordinary-looking `.py` files.
    return not any(part.startswith(".") or part in JUNK_NAMES for part in path.parts)


def package(checkpoint: Path, out_zip: Path, repo: Path) -> Path:
    """Zip the five prescribed entries. `src/` goes in whole, minus caches and OS droppings."""
    reqs = out_zip.parent / "requirements.txt"
    reqs.write_text(INFERENCE_REQUIREMENTS)

    # The archive ships its own README, because documenting the training and inference steps
    # inside it is part of the submission contract rather than a courtesy. A missing one is a
    # failed submission, not a cosmetic gap, so this refuses to build rather than shipping without.
    readme = repo / "docs" / "method.md"
    if not readme.exists():
        raise SystemExit(f"missing {readme} — the archive's README is a required deliverable.")

    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(repo / "predict.py", "predict.py")
        zf.write(reqs, "requirements.txt")
        zf.write(readme, "README.md")
        zf.write(checkpoint, "checkpoint.pt")
        for path in sorted((repo / "src").rglob("*")):
            if _shippable(path):
                zf.write(path, str(Path("src") / path.relative_to(repo / "src")))

    # Assert on the ARCHIVE, not on the walk that produced it. The filter above is the fix; this is
    # the check, and it also covers the three entries written by hand.
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    junk = [n for n in names if any(p.startswith(".") or p in JUNK_NAMES for p in n.split("/"))]
    if junk:
        raise SystemExit(f"archive contains hidden/OS files: {junk}")
    print(f"[submission] {len(names)} entries, no hidden or OS files")

    size = out_zip.stat().st_size
    pct = size / MAX_ARCHIVE_BYTES
    print(f"[submission] {out_zip}: {size / 1e6:.1f} MB ({pct:.0%} of the cap)")
    if size > MAX_ARCHIVE_BYTES:
        raise SystemExit(
            f"archive is {size / 1e6:.1f} MB, over the {MAX_ARCHIVE_BYTES / 1e6:.0f} MB cap."
        )
    return out_zip


def smoke(out_zip: Path, input_dir: Path, workdir: Path) -> None:
    """Unpack the archive somewhere else entirely and run the GRADED command against it.

    Deliberately not an in-process call: the thing being tested is the artifact, and the failure
    modes that matter (a module that only imports because the repo happens to be the cwd, a path
    that only resolves next to `configs/`) are invisible unless the zip is the only thing present.
    """
    workdir = workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    (workdir / "submission").mkdir(parents=True)
    (workdir / "output").mkdir(parents=True)
    with zipfile.ZipFile(out_zip) as zf:
        zf.extractall(workdir / "submission")

    cmd = [
        sys.executable,
        "predict.py",
        "--input_dir",
        str(input_dir.resolve()),
        "--output_file",
        str(workdir / "output" / "predictions.csv"),
        "--checkpoint",
        "checkpoint.pt",
    ]
    print(f"[submission] SMOKE in {workdir / 'submission'}\n  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=workdir / "submission", text=True)
    if res.returncode != 0:
        raise SystemExit(f"smoke FAILED (exit {res.returncode})")

    import pandas as pd

    out = pd.read_csv(workdir / "output" / "predictions.csv")
    idx_name = next(p.name for p in input_dir.glob("forecast_index*.csv"))
    idx = pd.read_csv(input_dir / idx_name)
    assert list(out.columns) == ["series_id", "timestamp", "prediction"], list(out.columns)
    assert len(out) == len(idx), f"{len(out)} rows vs {len(idx)} index rows"
    assert out["prediction"].notna().all(), "NaN predictions"
    print(
        f"[submission] SMOKE OK — {len(out)} rows, "
        f"mean {out['prediction'].mean():.4f}, min {out['prediction'].min():.4f}, "
        f"max {out['prediction'].max():.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nf-checkpoint", type=Path, default=Path("checkpoints/cascade_bag5.pt"))
    ap.add_argument("--tree", type=Path, default=Path("checkpoints/submission_tree.json"))
    ap.add_argument("--weight", type=float, default=TREE_WEIGHT)
    ap.add_argument("--out-dir", type=Path, default=Path("submission"))
    ap.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--smoke", action="store_true", help="unzip elsewhere and run the real CLI")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = bundle_checkpoint(
        args.nf_checkpoint, args.tree, args.out_dir / "checkpoint.pt", args.weight
    )
    zip_path = package(ckpt, args.out_dir / "final_submission.zip", repo)
    if args.smoke:
        smoke(zip_path, args.input_dir, Path("/tmp/dlam_submission_smoke"))


if __name__ == "__main__":
    main()
