"""EWMA must be built on history+horizon TOGETHER, never per frame. This fails silently otherwise.

WHY THIS FILE EXISTS (handoff 2a, 2026-09-04). `predict.run_tree` used to attach the cross-series
block to `hist` and `fut` SEPARATELY. That is correct for the A-blocks, which are WITHIN-HOUR
statistics — both frames give the identical answer. **The EW blocks walk TIME.** Attached per frame
they restart at the history/horizon boundary, every horizon hour loses its accumulation, and the
strict variant ships with NO exception raised and a perfectly well-formed CSV.

There is no output signature to check: both constructions produce finite, plausible, correctly
shaped predictions. The only way to catch it is to compare the horizon's EWMA values against what
the joined construction gives for the same hours — which is what this file does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.loader import NF_ID
from src.data.xs_blocks import BLOCKS, _zone_labels
from src.eval.splits import HOUR_IDX, add_hour_index

EW_COL = "ew_service_irregularity_risk_forecast_6"


@pytest.fixture(scope="module")
def frame():
    """Skipped rather than errored when the corpus is absent — `data/` is not in git.

    Without the guard a module-scoped fixture raises at collection time, which reports as an
    ERROR rather than a skip and turns a data-less clone's suite red.
    """
    from src.data.loader import load_long

    if not Path("data/raw/train.csv").exists():
        pytest.skip("data/raw/train.csv not present in this checkout")
    df, _ = load_long("data/raw/train.csv", strategy="interp")
    return add_hour_index(df)


def _build(df):
    return BLOCKS["EWA9"][1](df, _zone_labels(df))[0]


def test_per_frame_attachment_loses_the_accumulation(frame):
    """THE REGRESSION ITSELF: splitting the frame changes the horizon's values.

    If this ever stops failing to differ, the block has silently become within-hour and the whole
    concern is moot — but then EWMA is not doing what it claims either."""
    cut = 3648
    fut = frame[frame[HOUR_IDX] >= cut]
    joined = _build(frame)
    split_fut = _build(fut)

    j = joined[joined[HOUR_IDX] >= cut].sort_values([NF_ID, HOUR_IDX])[EW_COL].to_numpy()
    s = split_fut.sort_values([NF_ID, HOUR_IDX])[EW_COL].to_numpy()
    assert not np.allclose(j, s), (
        "per-frame and joined construction agree — either the block stopped walking time, "
        "or this test is no longer exercising the horizon"
    )
    # and the damage is concentrated at the boundary, which is the mechanism
    first = joined[joined[HOUR_IDX] == cut].sort_values(NF_ID)[EW_COL].to_numpy()
    first_split = split_fut[split_fut[HOUR_IDX] == cut].sort_values(NF_ID)[EW_COL].to_numpy()
    assert not np.allclose(first, first_split)


def test_the_horizon_ewma_is_not_constant(frame):
    """The cheap field check the handoff names: a restarted EWMA collapses toward a flat value.

    Non-constancy alone does not PROVE the joined construction, but a constant column is a
    guaranteed symptom, and it is checkable on any shipped artifact without a reference."""
    built = _build(frame)
    hz = built[built[HOUR_IDX] >= 3648]
    spread = hz.groupby(NF_ID)[EW_COL].std()
    assert (spread > 0).all(), "EWMA is constant across the horizon for some series"


def test_hour_t_is_excluded_from_its_own_feature(frame):
    """`shift(1)` is what makes the block legal at gap 336; without it a row reads its own hour."""
    built = _build(frame)
    scrambled = frame.copy()
    m = scrambled[HOUR_IDX] == 4000
    scrambled.loc[m, "service_irregularity_risk_forecast"] = 999.0
    b2 = _build(scrambled)
    a = built[built[HOUR_IDX] == 4000].sort_values(NF_ID)[EW_COL].to_numpy()
    b = b2[b2[HOUR_IDX] == 4000].sort_values(NF_ID)[EW_COL].to_numpy()
    assert np.allclose(a, b), "hour t's own value leaked into its own feature"
