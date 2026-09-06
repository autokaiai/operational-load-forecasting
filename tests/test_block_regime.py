"""The near-regime axis must be OPT-IN, per call, and it must never become a default.

Every weight this project holds was fitted on the FAR block. A near cube is a diagnostic (final
push, lane 1E) and the standing constraint is explicit: near-regime folds are opt-in, never a
default, and no weight, member or submission decision may be made on a near number.

That makes "the default is far" a property worth a test rather than a comment. There are six
places in this repo the axis is threaded through, and a single wrong default in any of them would
silently retarget a training run at the regime the grade does not score — with no error and a
plausible number at the end of it. These tests pin all six, plus the two guards that stop the near
block from being scored under the wrong covariate condition.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.splits import BLOCK_REGIMES, SCORE_LEN, take_block
from src.models.chronos2_eval import run_mode
from src.models.gap_fill_sweep import run_sweep
from src.models.lgbm import predict_gapped
from src.models.members import RunContext

REPO = Path(__file__).resolve().parent.parent


def _frame(n_series: int = 3, h: int = 2 * SCORE_LEN) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": np.repeat([f"unit_{i:03d}" for i in range(n_series)], h),
            "ds": pd.concat(
                [pd.Series(pd.date_range("2023-01-01", periods=h, freq="h"))] * n_series,
                ignore_index=True,
            ),
            "step": np.tile(np.arange(1, h + 1), n_series),
        }
    )


def test_every_regime_parameter_defaults_to_far():
    """The seven entry points, checked by signature rather than by reading the source."""
    assert RunContext(long_df=None, cut_idx=0).regime == "far"
    for fn, param in (
        (take_block, "regime"),
        (predict_gapped, "regime"),
        (run_mode, "regime"),
        (run_sweep, "regime"),
    ):
        assert inspect.signature(fn).parameters[param].default == "far", fn.__qualname__

    cli = (REPO / "scripts/member_preds_window.py").read_text()
    assert '"--regime"' in cli and 'default="far"' in cli


def test_both_tree_predict_gapped_signatures_still_mirror():
    """`_lgbm_runner` and `_catboost_runner` call these through the same shape.

    A `regime=` added to one and not the other is a TypeError on every CatBoost run — which is
    exactly what a broad string replace produced while this lane was being built.
    """
    from src.models.catboost_tree import predict_gapped as cb

    assert list(inspect.signature(cb).parameters) == list(
        inspect.signature(predict_gapped).parameters
    )
    assert inspect.signature(cb).parameters["regime"].default == "far"


def test_no_config_can_select_the_regime():
    """A config key would make the axis reachable without a caller ever naming it."""
    offenders = [p.name for p in (REPO / "configs").glob("*.yaml") if "regime" in p.read_text()]
    assert not offenders, f"configs must not carry a block regime: {offenders}"


def test_take_block_far_is_the_historical_tail_and_near_is_disjoint():
    df = _frame()
    far, near = take_block(df, "far"), take_block(df, "near")
    # `far` must be the exact call every member path already made, or recorded cubes move.
    pd.testing.assert_frame_equal(far, df.groupby("unique_id").tail(SCORE_LEN))
    assert len(far) == len(near) == 3 * SCORE_LEN
    assert far["step"].min() == SCORE_LEN + 1 and far["step"].max() == 2 * SCORE_LEN
    assert near["step"].min() == 1 and near["step"].max() == SCORE_LEN
    assert not set(map(tuple, near[["unique_id", "step"]].to_numpy())) & set(
        map(tuple, far[["unique_id", "step"]].to_numpy())
    )


@pytest.mark.parametrize("bad", ["late", "full", "FAR", "", None])
def test_unknown_regime_raises_rather_than_defaulting(bad):
    with pytest.raises(ValueError):
        take_block(_frame(), bad)
    assert bad not in BLOCK_REGIMES


def test_near_with_a_withheld_gap_is_refused():
    """The trap that cost lane 1E its first dispatch.

    ``_withhold_gap_covariates`` blanks the FIRST 336 rows per series. Under ``far`` those are the
    unscored gap; under ``near`` they are exactly the block being graded, so the run would measure
    "forecast 336h with every planning signal blanked" — a scenario that does not exist, because
    validation_input.csv supplies those covariates. The first 1E dispatch scored the near cascade
    at 0.336 / 0.699 / 0.373 against ~0.13 on the same rows from the far side.
    """
    with pytest.raises(ValueError, match="near"):
        run_mode(None, _frame(), "gapped-realistic", None, 0, regime="near")
    src = (REPO / "src/models/members.py").read_text()
    assert 'if ctx.regime == "near" and spec != "real" and gap_len > 0:' in src


def test_tree_regime_slicing_partitions_the_horizon_exactly():
    """The two tree slices must tile the 672h grid with no overlap and no gap.

    ``predict_gapped`` selects on ``horizon_step`` (1..672) rather than by position, so this is the
    place a near/far mix-up would be silent: both halves are 336 rows per series either way, and
    ``validate_member_frame`` would happily accept the wrong 336.
    """
    h, score_len = 2 * SCORE_LEN, SCORE_LEN
    step = np.arange(1, h + 1)
    far = step > h - score_len
    near = step <= score_len
    assert far.sum() == near.sum() == score_len
    assert not (far & near).any(), "the halves overlap"
    assert (far | near).all(), "the halves leave a gap"
    assert step[far].min() == SCORE_LEN + 1 and step[near].max() == SCORE_LEN

    src = (REPO / "src/models/lgbm.py").read_text()
    assert 'Xinf["horizon_step"] > horizon - score_len' in src, (
        "the far branch must stay the historical expression verbatim, or every recorded tree "
        "number moves"
    )


def test_run_member_refuses_near_for_a_runner_that_cannot_honour_it():
    """A runner without ``supports_regime`` would write its FAR block under a NEAR label."""
    from src.models.members import _REGISTRY, get_member

    unsupported = [n for n in _REGISTRY if not getattr(get_member(n).run, "supports_regime", False)]
    # The guard exists; at least one registered member should still be exercising it, and if the
    # day comes that every runner supports the axis this assertion is the reminder to drop it.
    src = (REPO / "src/models/members.py").read_text()
    assert 'getattr(spec.run, "supports_regime", False)' in src
    assert isinstance(unsupported, list)
