"""Tests for the seed policy in src.eval.protocol.

The policy was prose in the README and nothing enforced it: all 23 ``results/*.json``
carry ``seed: 42`` and no code path has ever touched the frozen set. These tests pin the two things
that make it mechanical rather than aspirational — the set still re-derives from its documented RNG
call, and an aggregate cannot be built from a hand-picked subset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.protocol import (
    FROZEN_SEEDS,
    TUNING_SEED,
    frozen_seeds,
    summarize_seed_runs,
    verify_frozen_seeds,
)


def test_frozen_set_reproduces_from_its_documented_derivation():
    """`np.random.default_rng(42).integers(0, 10000, size=5)` must still give the declared tuple."""
    assert verify_frozen_seeds() == FROZEN_SEEDS
    assert len(FROZEN_SEEDS) == 5
    assert len(set(FROZEN_SEEDS)) == 5, "a repeated seed would silently shrink the sample"


def test_frozen_set_is_stored_in_draw_order_not_sorted():
    """Docs list them sorted; the module stores the draw. Only the draw makes a prefix meaningful.

    If this ever flips to sorted order, `frozen_seeds(3)` silently returns a different subset than
    the one the compute-tight fallback means, and partial runs stop being comparable.
    """
    drawn = tuple(int(s) for s in np.random.default_rng(TUNING_SEED).integers(0, 10000, size=5))
    assert FROZEN_SEEDS == drawn
    assert FROZEN_SEEDS != tuple(sorted(FROZEN_SEEDS)), "stored sorted — prefix rule is now wrong"
    assert frozen_seeds(3) == drawn[:3]


def test_tuning_seed_is_not_in_the_frozen_report_set():
    """42 seeds the draw itself, so reusing it to report would reuse the seed we tuned on."""
    assert TUNING_SEED not in FROZEN_SEEDS


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_frozen_seeds_prefix_lengths(n):
    assert frozen_seeds(n) == FROZEN_SEEDS[:n]


@pytest.mark.parametrize("n", [0, 6, -1])
def test_frozen_seeds_rejects_out_of_range(n):
    with pytest.raises(ValueError, match="n must be in"):
        frozen_seeds(n)


# --------------------------------------------------------------------------- aggregation


def test_summarize_reports_mean_and_sample_std():
    scores = dict(zip(FROZEN_SEEDS, [0.160, 0.162, 0.161, 0.163, 0.159], strict=True))
    out = summarize_seed_runs(scores)
    vals = np.array(list(scores.values()))
    assert out["mean"] == pytest.approx(vals.mean())
    assert out["std"] == pytest.approx(vals.std(ddof=1))
    assert out["std_pop"] == pytest.approx(vals.std(ddof=0))
    assert out["std"] > out["std_pop"], "ddof=1 must exceed ddof=0; the two are not interchangeable"
    assert out["n_seeds"] == 5
    assert out["complete"] is True
    assert out["caveat"] == ""


def test_summarize_reports_in_draw_order():
    scores = {s: 0.16 + i * 1e-4 for i, s in enumerate(FROZEN_SEEDS)}
    out = summarize_seed_runs(scores)
    assert out["seeds"] == list(FROZEN_SEEDS)
    assert list(out["per_seed"]) == list(FROZEN_SEEDS)


def test_summarize_rejects_a_hand_picked_subset():
    """The exact failure the frozen set exists to prevent: three seeds that are not the prefix."""
    picked = {FROZEN_SEEDS[0]: 0.160, FROZEN_SEEDS[3]: 0.159, FROZEN_SEEDS[4]: 0.158}
    with pytest.raises(ValueError, match="not the frozen prefix"):
        summarize_seed_runs(picked)


def test_summarize_rejects_a_redrawn_seed():
    scores = dict(zip(FROZEN_SEEDS, [0.16] * 5, strict=True))
    scores[1234] = 0.15  # a seed nobody drew, quietly improving the mean
    del scores[FROZEN_SEEDS[-1]]
    with pytest.raises(ValueError, match="not the frozen prefix"):
        summarize_seed_runs(scores)


def test_summarize_rejects_the_tuning_seed_as_a_report_seed():
    with pytest.raises(ValueError, match="not the frozen prefix"):
        summarize_seed_runs({TUNING_SEED: 0.1463})


def test_summarize_accepts_a_prefix_but_marks_it_partial():
    """The sanctioned compute-tight path: first 3 in draw order, flagged so it reads as partial."""
    out = summarize_seed_runs(dict.fromkeys(frozen_seeds(3), 0.16))
    assert out["n_seeds"] == 3
    assert out["complete"] is False
    assert "PARTIAL" in out["caveat"]


def test_summarize_rejects_empty():
    with pytest.raises(ValueError, match="no seed scores"):
        summarize_seed_runs({})


def test_single_seed_std_is_nan_not_zero():
    """One run has no dispersion estimate. Zero would read as 'perfectly reproducible'."""
    out = summarize_seed_runs({FROZEN_SEEDS[0]: 0.1463})
    assert np.isnan(out["std"])
    assert out["complete"] is False


def test_headline_carries_the_metric_name():
    out = summarize_seed_runs(dict.fromkeys(frozen_seeds(2), 0.16), metric="pooled_wape[late]")
    assert "pooled_wape[late]" in out["headline"]
