"""Time-based splits derived from the 4320 labelled hours.

Two diagnostics live downstream of these helpers (see ``src.eval.cv``):

- **Contiguous rolling-origin CV** (handled by ``neuralforecast.cross_validation``) — the
  generalization estimate, mirroring the *validation* scenario where the forecast starts +1h
  after the last observed target.
- **Gapped eval**, mirroring the *private test*. The three official windows tile exactly
  (train h0-4319 / val h4320-4655 / test h4656-4991), so validation labels are never released
  and at test time the freshest label sits +337h before the scored block. We reproduce that
  offset locally: train on hours 1-3648, leave 3649-3984 unobserved (the 336h gap), and score
  hours 3985-4320 (the final 336h of a 672h horizon).
"""

from __future__ import annotations

import pandas as pd

from src.data.loader import NF_ID, NF_TIME

HOUR_IDX = "_hidx"
GAP_TRAIN_END_IDX = 3648  # 0-based: train = idx 0..3647 (hours 1..3648)
SCORE_LEN = 336  # scored block = the final 336h of the 672h gapped horizon

# Defaults for the contiguous rolling-origin CV (non-overlapping 14-day blocks).
CV_N_WINDOWS = 3
CV_STEP_SIZE = 336


def cv_window_cutoffs(n: int = 3, step: int = SCORE_LEN) -> list[int]:
    """Train-end cutoffs for the multi-window gapped architecture test (single source of truth).

    Each window W trains on hours ``_hidx < cut``, leaves the next ``step`` (336h) as the
    unobserved gap, and scores the following 336h — reproducing the private-test +337h offset.
    Anchored on ``GAP_TRAIN_END_IDX`` (W0) and rolled back ``step`` per window so the scored
    blocks are disjoint and tile the labelled tail:
        n=3 -> [3648, 3312, 2976]
        W0 cut 3648: gap [3648, 3984) score [3984, 4320)
        W1 cut 3312: gap [3312, 3648) score [3648, 3984)
        W2 cut 2976: gap [2976, 3312) score [3312, 3648)
    """
    return [GAP_TRAIN_END_IDX - i * step for i in range(n)]


def add_hour_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 0-based per-series hour index (rows sorted chronologically within each series)."""
    df = df.sort_values([NF_ID, NF_TIME]).reset_index(drop=True)
    df[HOUR_IDX] = df.groupby(NF_ID).cumcount()
    return df


def gapped_horizon(df: pd.DataFrame, cut_idx: int, horizon: int | None = None) -> pd.DataFrame:
    """Rows of the FIXED gapped window ``[cut_idx, cut_idx + horizon)`` (default 672 = gap+scored).

    Single source of truth for every gapped-eval slice (``cv.run_gapped_eval_at``,
    ``scripts.member_preds_window.nf_member_preds``, ``models.chronos2_eval.run_mode``). Pinning the
    length matters: ``df[_hidx >= cut_idx]`` (every row after the cutoff) equals ``horizon`` ONLY at
    the final cutoff (W0, where ``cut+672 == end-of-data``). For an earlier cutoff it over-runs
    (W1 -> 1008, W2 -> 1344), and a downstream ``tail(SCORE_LEN)`` then silently scores the LAST
    block ([3984,4320)) for every window instead of the window's own block — the bug this prevents.

    ``df`` must already carry ``HOUR_IDX`` (call ``add_hour_index`` first).
    """
    h = horizon if horizon is not None else 2 * SCORE_LEN
    return df[(df[HOUR_IDX] >= cut_idx) & (df[HOUR_IDX] < cut_idx + h)]


# --------------------------------------------------------------------------- block regime
#
# A FOURTH axis, and it is the one that must never move by accident. `regime` in
# ``src.eval.protocol`` slices the scored block by position within it (full vs blk>=224);
# `horizon` there slices a 672h forecast into halves for reporting. THIS one decides which half of
# the 672h forecast a member's cube is written from in the first place, i.e. what "the scored
# block" means for every downstream number:
#
#     far   steps 337..672 — a 336h covariate-absent gap sits first. The PRIVATE TEST scenario,
#           the only regime we are graded on, and the regime every weight this project holds was
#           fitted on.
#     near  steps 1..336   — the forecast starts +1h after the last observed target. The public
#           leaderboard's scenario.
#
# ``far`` is the default everywhere and it stays that way (final-push lane 1E): a near cube is an
# opt-in diagnostic, never a default, because a weight fitted on near optimises for the regime the
# grade does not score. Callers pass it explicitly or they get ``far``.
BLOCK_REGIMES: tuple[str, ...] = ("far", "near")


def take_block(df: pd.DataFrame, regime: str = "far", score_len: int = SCORE_LEN) -> pd.DataFrame:
    """The ``score_len`` rows per series that ``regime`` scores, out of a gapped-horizon frame.

    ``far`` is ``groupby(NF_ID).tail(score_len)`` verbatim — the call every member path already
    made — so a far-regime cube stays bit-identical to the ones on record. ``near`` is the
    corresponding ``head``.

    ``df`` must hold one series' full 672h horizon per group, in chronological order.
    """
    if regime not in BLOCK_REGIMES:
        raise ValueError(f"regime must be one of {BLOCK_REGIMES}, got {regime!r}")
    g = df.groupby(NF_ID)
    return g.tail(score_len) if regime == "far" else g.head(score_len)
