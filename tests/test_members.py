"""The member runner registry — contract, capabilities, and the dispatch it replaced.

The grading layer (``src.eval.protocol``) was always member-agnostic; the runner layer was not.
These tests pin the properties that make it generic, so a Phase-5 member is a registration rather
than a branch in three scripts.

Nothing here fits a model: the runners are replaced with fakes. What is under test is the registry
and the contract it enforces, not LightGBM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.protocol import CUTOFF, validate_prediction_df
from src.models import members as mem

CUT = 3648
N_SERIES, N_STEPS = 3, 8


def make_output(name: str, *, n_series: int = N_SERIES, n_steps: int = N_STEPS) -> pd.DataFrame:
    """A well-formed single-window member frame, in the shape a runner must return."""
    rows = []
    for s in range(n_series):
        ds = pd.date_range("2023-06-01", periods=n_steps, freq="h")
        y = np.linspace(10.0, 20.0, n_steps) + s
        rows.append(
            pd.DataFrame({NF_ID: f"unit_{s:03d}", NF_TIME: ds, NF_TARGET: y, name: y * 1.05})
        )
    return pd.concat(rows, ignore_index=True)


def fake_spec(name: str, **kw) -> mem.MemberSpec:
    defaults = {
        "kind": "tree",
        "run": lambda ctx: make_output(name),
        "seedable": True,
        "needs_gpu": False,
    }
    return mem.MemberSpec(name=name, **{**defaults, **kw})


@pytest.fixture
def registry_sandbox(monkeypatch):
    """Register into a copy, so a test never mutates the real registry for the ones after it."""
    monkeypatch.setattr(mem, "_REGISTRY", dict(mem._REGISTRY))
    return mem._REGISTRY


def ctx(**kw) -> mem.RunContext:
    return mem.RunContext(long_df=pd.DataFrame(), cut_idx=CUT, **kw)


# --------------------------------------------------------------------------- the real registry


def test_every_shipped_member_is_registered():
    """The members with recorded numbers are all reachable by name."""
    names = mem.available_members()
    for expected in ("lgbm", "tft", "chronos_ft"):
        assert expected in names, f"{expected} missing from the registry: {names}"


def test_registry_is_the_only_dispatch_the_cli_has():
    """`member_preds_window` must not reintroduce a hardcoded per-member branch."""
    src = (mem.Path(__file__).parent.parent / "scripts" / "member_preds_window.py").read_text()
    for hardcoded in ('member == "lgbm"', 'member == "chronos_ft"', "in NF_MEMBERS"):
        assert hardcoded not in src, (
            f"member_preds_window.py reintroduced the hardcoded dispatch {hardcoded!r}. "
            "Register the member in src/models/members.py instead."
        )


def test_capability_flags_match_what_each_member_actually_is():
    assert mem.get_member("lgbm").needs_gpu is False
    assert mem.get_member("lgbm").seedable is True
    # LightGBM never traverses the gap: origin-anchored lags, covariates read at the forecast hour.
    # So --gap-cov cannot move it, and plan 3.5 leaves its number alone.
    assert mem.get_member("lgbm").honours_gap_cov is False

    assert mem.get_member("tft").needs_gpu is True
    assert mem.get_member("tft").honours_gap_cov is True

    # Inference-only against a fixed LoRA adapter -> deterministic, and it cannot run without one.
    assert mem.get_member("chronos_ft").seedable is False
    assert mem.get_member("chronos_ft").needs_adapter is True


def test_tier_a_variants_are_registered_but_flagged_cut():
    """S2 cut them, and the registry says so rather than deleting the record.

    They were never *discarded* — they were never *measured* (``status="untested"`` throughout, no
    Phase-4 composite included them). The cut is on the two-cluster finding: ``lgbm`` and
    ``lgbm_s24_unitcat`` correlate at 0.976, and that is one library at two settings, so a third
    LightGBM variant is a cluster-B duplicate by construction. "We chose not to run this, and here
    is why" is a result the write-up wants, and a deleted registration cannot carry it.
    """
    cut = mem.available_members(status="cut")
    assert "lgbm_dart" in cut and "lgbm_extra_trees" in cut
    # ...and neither a measured-members sweep nor an untested-members one can pick them up now.
    for status in ("measured", "untested"):
        assert "lgbm_dart" not in mem.available_members(status=status)
    assert "lgbm" in mem.available_members(status="measured")
    assert all("CUT" in mem.get_member(m).note for m in ("lgbm_dart", "lgbm_extra_trees"))


def test_unknown_member_names_the_alternatives():
    with pytest.raises(KeyError, match="lgbm"):
        mem.get_member("lightgbm")  # a plausible typo


# --------------------------------------------------------------------------- registration


def test_registering_the_same_name_twice_is_an_error(registry_sandbox):
    mem.register(fake_spec("dup"))
    with pytest.raises(ValueError, match="already registered"):
        mem.register(fake_spec("dup"))


def test_unknown_kind_is_rejected(registry_sandbox):
    with pytest.raises(ValueError, match="unknown kind"):
        mem.register(fake_spec("weird", kind="quantum"))


def test_available_members_filters_by_kind():
    assert "lgbm" in mem.available_members("tree")
    assert "lgbm" not in mem.available_members("neural")
    assert "chronos_ft" in mem.available_members("foundation")


# --------------------------------------------------------------------------- run_member guards


def test_a_deterministic_member_refuses_a_seed_rather_than_ignore_it(registry_sandbox):
    """Silently dropping a seed is the exact defect plan 3.6 found in configs/lgbm.yaml.

    Ignoring it here would be worse than a no-op: it would let a caller report a `mean +- std`
    over five identical runs and claim a precision the member does not have.
    """
    mem.register(fake_spec("fixed", seedable=False))
    with pytest.raises(ValueError, match="deterministic"):
        mem.run_member("fixed", ctx(seed=7))
    # ...and runs fine when no seed is asked for.
    assert len(mem.run_member("fixed", ctx())) == N_SERIES * N_STEPS


def test_a_member_needing_an_adapter_fails_before_spending_the_compute(registry_sandbox):
    ran = []
    mem.register(
        fake_spec("needs_lora", needs_adapter=True, run=lambda c: ran.append(1) or make_output("x"))
    )
    with pytest.raises(ValueError, match="adapter"):
        mem.run_member("needs_lora", ctx())
    assert ran == [], "the guard must fire before the runner is called"


def test_seed_reaches_the_runner(registry_sandbox):
    seen = {}

    def run(c):
        seen["seed"] = c.seed
        return make_output("spy")

    mem.register(fake_spec("spy", run=run))
    mem.run_member("spy", ctx(seed=892))
    assert seen["seed"] == 892


def test_invalid_gap_cov_is_rejected(registry_sandbox):
    mem.register(fake_spec("gc"))
    with pytest.raises(ValueError, match="gap_cov"):
        mem.run_member("gc", ctx(gap_cov="optimistic"))


# --------------------------------------------------------------------------- the output contract


def test_cutoff_is_emitted_so_the_protocol_validator_accepts_the_cube(registry_sandbox):
    """Plan 3.2 flagged that no artifact carried `cutoff`. The emitter now adds it."""
    mem.register(fake_spec("cut"))
    out = mem.run_member("cut", ctx())
    assert CUTOFF in out.columns and set(out[CUTOFF]) == {CUT}
    # concatenating windows now yields a frame validate_prediction_df accepts as-is
    cube = pd.concat(
        [
            mem.run_member("cut", mem.RunContext(long_df=pd.DataFrame(), cut_idx=c))
            for c in (2976, 3312, 3648)
        ],
        ignore_index=True,
    ).sort_values([CUTOFF, NF_ID, NF_TIME])
    validate_prediction_df(cube, "cut")


def test_output_columns_are_canonical_and_ordered(registry_sandbox):
    """A runner may return columns in any order; run_member normalises them."""
    scrambled = make_output("mess")[["mess", NF_TARGET, NF_TIME, NF_ID]]
    mem.register(fake_spec("mess", run=lambda c: scrambled))
    out = mem.run_member("mess", ctx())
    assert list(out.columns) == [NF_ID, NF_TIME, CUTOFF, NF_TARGET, "mess"]


def test_rows_come_back_sorted(registry_sandbox):
    shuffled = make_output("shuf").sample(frac=1.0, random_state=0)
    mem.register(fake_spec("shuf", run=lambda c: shuffled))
    out = mem.run_member("shuf", ctx())
    assert out[[NF_ID, NF_TIME]].equals(
        out.sort_values([NF_ID, NF_TIME])[[NF_ID, NF_TIME]].reset_index(drop=True)
    )


def test_nan_predictions_are_caught_here_not_three_scripts_downstream(registry_sandbox):
    bad = make_output("nan_member")
    bad.loc[5, "nan_member"] = np.nan
    mem.register(fake_spec("nan_member", run=lambda c: bad))
    with pytest.raises(AssertionError, match="NaN"):
        mem.run_member("nan_member", ctx())


def test_ragged_output_is_rejected(registry_sandbox):
    """Unequal per-series row counts silently over-weight the longer series in pooled WAPE."""
    ragged = make_output("ragged").drop(index=[0, 1, 2]).reset_index(drop=True)
    mem.register(fake_spec("ragged", run=lambda c: ragged))
    with pytest.raises(AssertionError, match="ragged"):
        mem.run_member("ragged", ctx())


def test_missing_prediction_column_names_what_it_found(registry_sandbox):
    """The classic bug: a runner forgets to rename its output column to the member's name."""
    wrong_name = make_output("lgbm")  # variant returned the base member's column name
    mem.register(fake_spec("lgbm_variant", run=lambda c: wrong_name))
    with pytest.raises(AssertionError, match="lgbm_variant"):
        mem.run_member("lgbm_variant", ctx())


def test_empty_output_is_rejected(registry_sandbox):
    mem.register(fake_spec("empty", run=lambda c: make_output("empty").iloc[:0]))
    with pytest.raises(AssertionError, match="empty prediction frame"):
        mem.run_member("empty", ctx())


def test_duplicate_rows_are_rejected(registry_sandbox):
    doubled = pd.concat([make_output("dupes")] * 2, ignore_index=True)
    mem.register(fake_spec("dupes", run=lambda c: doubled))
    with pytest.raises(AssertionError, match="duplicate"):
        mem.run_member("dupes", ctx())


# --------------------------------------------------------------------------- config handling


def test_tree_config_is_read_without_the_neural_base_config():
    """`src.train.load_config` merges configs/base.yaml, whose keys are not booster parameters.

    LightGBM ignores unknown parameters with a warning that `verbosity: -1` then hides — invisible
    rather than harmless. CatBoost (Phase 5 Tier A) raises instead, so this had to be fixed before
    the registry gets a second tree family.
    """
    cfg = mem._load_yaml("configs/lgbm.yaml")
    assert "objective" in cfg, "the tree's own config should still be read"
    for neural_only in ("h", "input_size", "max_steps", "accelerator", "windows_batch_size"):
        assert neural_only not in cfg, (
            f"{neural_only!r} leaked from configs/base.yaml into the booster parameter dict"
        )


def test_missing_config_file_is_not_fatal():
    assert mem._load_yaml("configs/does_not_exist.yaml") == {}
