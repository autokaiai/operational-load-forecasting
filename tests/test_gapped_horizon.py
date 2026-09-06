"""Regression test for the gapped-horizon slice (the W1/W2 mis-scoring bug).

The bug: three call sites computed the gapped horizon as ``df[_hidx >= cut_idx]`` (every row after
the cutoff). That equals 672 only at the FINAL cutoff (W0); at earlier cutoffs it over-runs (W1 ->
1008, W2 -> 1344), so a downstream ``tail(SCORE_LEN)`` silently scored the LAST 336h block
([3984,4320)) for every window instead of the window's own block. ``splits.gapped_horizon`` pins the
length; this test asserts every window scores its own disjoint block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import (
    HOUR_IDX,
    SCORE_LEN,
    add_hour_index,
    cv_window_cutoffs,
    gapped_horizon,
)

N_SERIES = 4
N_HOURS = 4320


def _toy_long() -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=N_HOURS, freq="h")
    frames = []
    for s in range(N_SERIES):
        frames.append(
            pd.DataFrame(
                {NF_ID: f"unit_{s:03d}", NF_TIME: ts, NF_TARGET: np.arange(N_HOURS, dtype=float)}
            )
        )
    return add_hour_index(pd.concat(frames, ignore_index=True))


def test_gapped_horizon_is_fixed_672_at_every_cutoff():
    df = _toy_long()
    for cut in cv_window_cutoffs():  # [3648, 3312, 2976]
        horizon = gapped_horizon(df, cut)
        h = int(horizon.groupby(NF_ID).size().min())
        assert h == 2 * SCORE_LEN, f"cut={cut}: h={h}, expected 672"


def test_scored_block_is_the_windows_own_disjoint_block():
    """tail(SCORE_LEN) of the gapped horizon must be [cut+336, cut+672) — NOT the global tail."""
    df = _toy_long()
    seen_blocks = []
    for cut in cv_window_cutoffs():
        scored = gapped_horizon(df, cut).groupby(NF_ID).tail(SCORE_LEN)
        lo = int(scored[HOUR_IDX].min())
        hi = int(scored[HOUR_IDX].max())
        assert (lo, hi) == (cut + SCORE_LEN, cut + 2 * SCORE_LEN - 1), (
            f"cut={cut}: scored [{lo},{hi}], expected [{cut + SCORE_LEN},{cut + 2 * SCORE_LEN - 1}]"
        )
        assert scored.groupby(NF_ID).size().min() == SCORE_LEN
        seen_blocks.append((lo, hi))
    # disjoint, tiling blocks (the whole point of multi-window CV)
    assert len(set(seen_blocks)) == len(seen_blocks), "scored blocks overlap/collapse"


def test_buggy_uncapped_slice_would_collapse_to_global_tail():
    """Documents the original bug: the un-capped slice mis-scores every non-final window."""
    df = _toy_long()
    blocks = []
    for cut in cv_window_cutoffs():
        buggy = df[df[HOUR_IDX] >= cut]  # the old code
        scored = buggy.groupby(NF_ID).tail(SCORE_LEN)
        blocks.append((int(scored[HOUR_IDX].min()), int(scored[HOUR_IDX].max())))
    # all three collapse to the SAME global-tail block -> exactly the silent corruption
    assert len(set(blocks)) == 1 and blocks[0] == (N_HOURS - SCORE_LEN, N_HOURS - 1)
