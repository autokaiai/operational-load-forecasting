"""Derived cross-series aggregates — do they REACH the model?

Plan law 5: *a lever that does not reach the model returns a confident null*. Sprint 2 promotes
cross-series aggregate covariates from the LightGBM screen to the cascade TFT, and the obvious way
to do that — monkeypatching ``futr_exog_list``, which is exactly how the aggregate screen injects
them into the tree — **cannot work for a neural member**.
``src/models/registry.py`` binds the function by value at module import, so it keeps its own
reference and never sees the patch. The TFT would then train without the aggregates and the arm
would read as "cross-series does not transfer", which is the most expensive possible way to be
wrong about this sprint.

``test_monkeypatching_futr_exog_list_does_not_reach_the_model`` pins that failure mode, and
``test_active_aggregates_reach_a_constructed_model`` pins the fix. Asserting the state variable was
set would not be evidence — a test that cannot fail is not evidence (law 6).
"""

from __future__ import annotations

import pytest

from src.data import features as F


@pytest.fixture(autouse=True)
def _clean():
    """No test may leak an activation into the next one."""
    previous = F.set_aggregate_columns([])
    yield
    F.set_aggregate_columns(previous)


NAMES = ["xs_sys_queue_pressure_forecast", "xs_zone_queue_pressure_forecast"]


def test_inert_when_nothing_is_active() -> None:
    """With no aggregates the conditioning set must be IDENTICAL to the shipped one, or every
    checkpoint on record stops being comparable."""
    assert F.active_aggregate_columns() == []
    base = F.futr_exog_list()
    with F.aggregate_columns(NAMES):
        pass
    assert F.futr_exog_list() == base


def test_active_columns_are_appended_last() -> None:
    """Appended, never interleaved: column ORDER is part of a saved design's identity."""
    base = F.futr_exog_list()
    with F.aggregate_columns(NAMES):
        assert F.futr_exog_list() == [*base, *NAMES]


def test_aggregates_are_not_imputed_channels() -> None:
    """They are DERIVED from already-imputed columns, so they carry no `*_missing` twin."""
    with F.aggregate_columns(NAMES):
        assert not set(NAMES) & set(F.nan_col_list())
        assert not any(f"{n}_missing" in F.missing_indicator_cols() for n in NAMES)


def test_context_manager_restores_a_previous_selection() -> None:
    """Nesting must not leak, or one member's aggregates land in the next member's run."""
    with F.aggregate_columns(NAMES[:1]):
        with F.aggregate_columns(NAMES[1:]):
            assert F.active_aggregate_columns() == NAMES[1:]
        assert F.active_aggregate_columns() == NAMES[:1]
    assert F.active_aggregate_columns() == []


def test_bad_selections_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        F.set_aggregate_columns(["a", "a"])
    with pytest.raises(ValueError, match="non-empty strings"):
        F.set_aggregate_columns(["  "])
    # A name that shadows a real covariate would silently REPLACE it in the design.
    with pytest.raises(ValueError, match="collide"):
        F.set_aggregate_columns([F.KNOWN_FUTURE_SIGNALS[0]])


def test_monkeypatching_futr_exog_list_does_not_reach_the_model() -> None:
    """THE TRAP THIS MECHANISM EXISTS TO AVOID, pinned so nobody re-invents it.

    `registry` imported the function by value; rebinding it here changes nothing downstream.
    """
    pytest.importorskip("neuralforecast")
    from src.models import registry

    original = F.futr_exog_list
    try:
        F.futr_exog_list = lambda: [*original(), "xs_never_arrives"]  # type: ignore[assignment]
        model = registry.build_model(_TINY_TFT)
        assert "xs_never_arrives" not in model.futr_exog_list
    finally:
        F.futr_exog_list = original  # type: ignore[assignment]


def test_active_aggregates_reach_a_constructed_model() -> None:
    """THE SUBSTANTIVE ASSERTION. State, not rebinding, arrives at the model that will be fitted."""
    pytest.importorskip("neuralforecast")
    from src.models import registry

    with F.aggregate_columns(NAMES):
        model = registry.build_model(_TINY_TFT)
    assert all(n in model.futr_exog_list for n in NAMES)


_TINY_TFT = {
    "model": "TFT",
    "name": "t",
    "h": 8,
    "input_size": 16,
    "max_steps": 1,
    "hidden_size": 8,
    "n_head": 1,
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
}
