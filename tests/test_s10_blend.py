"""S10: the four-rung ladder, the shipped weights, and the member/covariate line.

Why these exist at all
----------------------
This project has shipped TWO schema-perfect wrong answers. The #32 horizon defect wrote a forecast
of the wrong HOURS into a valid CSV; the bag defect wrote a forecast from the wrong MODEL into one
(seed 0 of five, 2.3 WAPE points). Both passed every check their artifact carried, because none of
those checks compared against a NUMBER. `test_the_shipped_weights_rescore_to_the_recorded_cv_number`
is the one that would have caught both, and it is not optional.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import predict
from scripts.make_submission import FULLFT_WEIGHTS, TREE_WEIGHT
from src.models.fullft_inference import FULLFT_COL

SHIP_CV_POOLED = 0.12722  # S10.1, three windows, scored block
RUNG2_CV_POOLED = 0.13163  # S6's shipped pair, re-derived in the same frame


def _vecs(n: int = 8):
    """Three DISTINCT member vectors, so any rung that ignores one is detectable."""
    rng = np.random.default_rng(0)
    return rng.normal(10, 1, n), rng.normal(10, 1, n), rng.normal(10, 1, n)


# --------------------------------------------------------------------------- T1: the arithmetic
def test_rung1_is_exactly_the_three_measured_weights():
    """T1. FAILS before the fix: `choose_rung` did not exist and `main` had no third member."""
    full, bag, tree = _vecs()
    rung, values, _ = predict.choose_rung(
        neural_values=bag,
        tree_values=tree,
        fullft_values=full,
        tree_weight=0.2438,
        fullft_weights=FULLFT_WEIGHTS,
    )
    assert rung == 1
    # Sprint 2 weights, WRITTEN OUT rather than read from FULLFT_WEIGHTS. The duplication is the
    # test: it is what makes a silent edit to the shipped constant fail here. All three members now
    # carry the A9 cross-series aggregates; late 0.12029 against the previous 0.12722.
    expected = 0.4723535251605685 * full + 0.3413176080096898 * bag + 0.18632886682974184 * tree
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)


def test_the_rung1_weights_are_a_convex_combination():
    """A blend whose weights do not sum to 1 is a scaled forecast, and WAPE is scale-sensitive."""
    w = FULLFT_WEIGHTS
    assert w["chronos_full_ft"] + w["cascade_bag"] + w["tree"] == pytest.approx(1.0, abs=1e-9)
    assert min(w["chronos_full_ft"], w["cascade_bag"], w["tree"]) > 0


def test_non_convex_rung1_weights_raise_rather_than_silently_rescaling():
    full, bag, tree = _vecs()
    with pytest.raises(ValueError, match="sum to"):
        predict.choose_rung(
            neural_values=bag,
            tree_values=tree,
            fullft_values=full,
            tree_weight=0.2438,
            fullft_weights={"chronos_full_ft": 0.5, "cascade_bag": 0.5, "tree": 0.5},
        )


# ------------------------------------------------------- T2: losing the member must land on rung 2
def test_without_the_member_rung2_is_EXACTLY_todays_shipped_model():
    """T2. The degradation must reproduce the model we already scored at 13.622, bit for bit.

    Asserted against ``TREE_WEIGHT`` — the constant the archive actually ships (0.24, S6's fitted
    weight) — NOT against S10's re-derived 0.2438. Those are two different frames' answers to the
    same question and only the first one has a leaderboard number attached to it.

    It must also NOT renormalise rung 1's weights onto two members (0.4482/0.1273 ->
    0.7788/0.2212). That is a different model and nothing has measured it.
    """
    full, bag, tree = _vecs()
    rung, values, _ = predict.choose_rung(
        neural_values=bag,
        tree_values=tree,
        fullft_values=None,
        tree_weight=TREE_WEIGHT,
        fullft_weights=FULLFT_WEIGHTS,
    )
    assert rung == 2
    expected = TREE_WEIGHT * tree + (1.0 - TREE_WEIGHT) * bag
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)

    renormalised = (0.4482 * bag + 0.1273 * tree) / (0.4482 + 0.1273)
    assert not np.allclose(values, renormalised), "rung 2 renormalised rung 1 instead of using S6's"


def test_the_shipped_tree_weight_is_the_one_that_scored_on_the_leaderboard():
    """Guards the pair above from drifting apart: `v-final-2` = 13.622 was produced at this
    weight, so a change here silently invalidates that reference point."""
    assert TREE_WEIGHT == pytest.approx(0.24, abs=1e-9)


def test_a_checkpoint_without_rung1_weights_still_ships_rung2():
    """An older checkpoint carries no `blend["fullft"]`. It must degrade, not crash."""
    full, bag, tree = _vecs()
    rung, values, _ = predict.choose_rung(
        neural_values=bag,
        tree_values=tree,
        fullft_values=full,
        tree_weight=0.2438,
        fullft_weights=None,
    )
    assert rung == 2
    np.testing.assert_allclose(values, 0.2438 * tree + 0.7562 * bag, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- T3: rungs announce
def _rung(bag, tree, full):
    return predict.choose_rung(
        neural_values=bag,
        tree_values=tree,
        fullft_values=full,
        tree_weight=0.2438,
        fullft_weights=FULLFT_WEIGHTS,
    )


def test_every_rung_announces_itself_and_names_its_members():
    full, bag, tree = _vecs()
    r1, _, m1 = _rung(bag, tree, full)
    r2, _, m2 = _rung(bag, tree, None)
    r4, _, m4 = _rung(None, tree, None)
    assert (r1, r2, r4) == (1, 2, 4)
    assert FULLFT_COL in m1 and FULLFT_COL not in m2
    assert "DEGRADED" in m4


def test_the_ladder_is_monotone_in_members_present():
    """Losing a member may only move DOWN the ladder, never up."""
    full, bag, tree = _vecs()
    rungs = [_rung(bag, tree, full)[0], _rung(bag, tree, None)[0], _rung(None, tree, None)[0]]
    assert rungs == sorted(rungs), f"the ladder is not monotone: {rungs}"


# ------------------------------------------------- T4: one source of truth for the weights
def test_the_weights_are_defined_once_and_travel_in_the_checkpoint():
    """T4. S9 found a sidecar disagreeing with its own model; a second hardcoded copy of these
    numbers is the same defect waiting to happen. `predict.py` must read them, never define them."""
    src = Path("predict.py").read_text()
    for literal in ("0.4245", "0.4482", "0.1273"):
        assert literal not in src, f"{literal} is hardcoded in predict.py; read it from the sidecar"


# ------------------------------- T5: the member is NEVER a covariate (the standing rule)
def test_the_fullft_member_is_never_a_future_covariate():
    """T5. The standing architectural rule: a model FITTED ON OUR DATA enters only as a blend
    member. An OOF forecast for hour t from a model trained on our targets can encode target[t]."""
    from src.data.features import CASCADE_FORECASTS_ALL, futr_exog_list

    assert FULLFT_COL not in futr_exog_list()
    assert FULLFT_COL not in set(CASCADE_FORECASTS_ALL)


def test_disabling_the_member_is_possible_for_the_cleanroom_rung2_check():
    from src.models import fullft_inference as ff

    prev = os.environ.get(ff._DISABLE_ENV)
    try:
        os.environ[ff._DISABLE_ENV] = "1"
        assert not ff.fullft_enabled()
        os.environ[ff._DISABLE_ENV] = ""
        assert ff.fullft_enabled()
    finally:
        os.environ.pop(ff._DISABLE_ENV, None)
        if prev is not None:
            os.environ[ff._DISABLE_ENV] = prev


def test_an_unset_repo_resolves_to_none_rather_than_guessing():
    """A wrong repo id would fetch SOMEONE ELSE'S weights and produce a confident wrong forecast."""
    from src.models import fullft_inference as ff

    assert ff.FULLFT_REPO == "" or isinstance(ff.FULLFT_REPO, str)
    if not ff.FULLFT_REPO:
        assert ff.resolve_source(allow_download=True) is None


# ------------------------------------------------------------------ T8: compare against a NUMBER
def _all_cubes_present() -> bool:
    """`build(optional=set())` needs EVERY registered member, not just one.

    The old guard checked a single cube. That was fine while the local tree was all-or-nothing, and
    it broke the moment a PARTIAL set arrived (sprint 2 pulled the shipped members plus the seed
    bags from the volume, but the seven FOUNDATION members and `chronos2_zeroshot` are not stored
    under `results/gapcov` at all). The test then ran and died in `build` with SystemExit instead of
    skipping — a red suite that says nothing about the artifact. Guard on what the call needs.
    """
    from scripts.ensemble_full_sweep import CUTOFFS, FOUNDATION, POOL, REFERENCE, SEED_SRC

    sources = {**POOL, **REFERENCE, **SEED_SRC, **FOUNDATION}
    return all(
        Path(f"{root}/window{w}/{stem}_preds.csv").exists()
        for _, (root, stem, _) in sources.items()
        for w in CUTOFFS
    )


@pytest.mark.skipif(
    not _all_cubes_present(),
    reason="CV cubes are gitignored; this asserts locally where the FULL member set lives",
)
def test_the_shipped_weights_rescore_to_the_recorded_cv_number():
    """T8 — THE ONLY TEST HERE THAT COMPARES AGAINST A NUMBER, and the one that would have caught
    both historical defects. If the shipped weights stop reproducing 0.12722 on the cube, the
    artifact and its claim have come apart."""
    from scripts.ensemble_full_sweep import BAG, Search, build

    df, missing = build(optional=set())
    assert not missing, missing
    S = Search(df, [FULLFT_COL, BAG, "tree_interp"])

    ship = S.blend(
        {
            FULLFT_COL: FULLFT_WEIGHTS["chronos_full_ft"],
            BAG: FULLFT_WEIGHTS["cascade_bag"],
            "tree_interp": FULLFT_WEIGHTS["tree"],
        }
    )
    rung2 = S.blend({"tree_interp": 0.2438, BAG: 0.7562})

    assert S._pooled(ship) == pytest.approx(SHIP_CV_POOLED, abs=1e-4)
    assert S._pooled(rung2) == pytest.approx(RUNG2_CV_POOLED, abs=1e-4)
    assert S._pooled(ship) < S._pooled(rung2), "rung 1 must beat rung 2 on the cube"

    # ...and it must win EVERY window, not merely pool better. A pooled win carried by one window
    # is the pathology `majority_windows_improve` exists to catch.
    per_ship, per_r2 = S._per_window(ship), S._per_window(rung2)
    assert all(per_ship[k] < per_r2[k] for k in per_ship), f"{per_ship} vs {per_r2}"


# ------------------------------------------------------------------- G3: the hosted member's pin
def test_the_hosted_member_is_pinned_to_a_specific_revision():
    """An unpinned repo resolves whatever `main` holds on the day.

    That would let the graded run fetch DIFFERENT weights from the ones every number in S10
    describes, with no symptom whatsoever — the same hazard `cascade_inference.PINNED_REVISION`
    was added for. Unlike that pin (added retroactively), this one predates the repo's only commit.
    """
    from src.models import fullft_inference as ff

    assert ff.FULLFT_REPO == "autokai/chronos2-fullft-dlam-g100"
    assert len(ff.FULLFT_REVISION) == 40, "pin a full 40-char commit sha, not a branch name"
    assert all(c in "0123456789abcdef" for c in ff.FULLFT_REVISION)
