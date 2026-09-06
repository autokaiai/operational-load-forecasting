"""The cube cache exists to save hours, but every test here is about it refusing to.

A cache miss costs one refit. A stale cache **hit** costs a wrong number in a results JSON that
nothing downstream will question — the A/B report does not record whether its arms were computed
or read. So the interesting behaviour is not "does it reuse", it is "does it decline to reuse the
moment anything that could change the predictions has changed".

Hence the shape of this file: one round-trip test, and eight ways to make it miss.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.models import cube_cache as cc
from src.models import members as mem

CUTS = (3648, 3312, 2976)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Never touch the real results/member_cubes, and never inherit a real registry entry."""
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "member_cubes")
    monkeypatch.setattr(mem, "_REGISTRY", dict(mem._REGISTRY))
    monkeypatch.delenv("MEMBER_CUBE_CACHE", raising=False)


def _register(name: str, *, overrides: dict, config: str | None = None, declare: bool = True):
    def run(ctx):  # pragma: no cover - never invoked; the cache is tested, not the runner
        raise AssertionError("the runner must not be called in these tests")

    if declare:
        run.member_config = config
        run.member_overrides = dict(overrides)
        run.member_code = ()
    mem.register(mem.MemberSpec(name=name, kind="tree", run=run, seedable=True, needs_gpu=False))
    return name


def _cube(name: str, value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["unit_000", "unit_000"],
            "ds": [4000, 4001],
            "cutoff": [3648, 3648],
            "y": [2.0, 3.0],
            name: [value, value],
        }
    )


def _key(name: str, *, seed: int = 42, train_csv: str = "data/raw/train.csv"):
    return cc.cache_key(name, seed=seed, cutoffs=CUTS, train_csv=train_csv)


# --------------------------------------------------------------------------- the happy path


def test_a_stored_cube_comes_back_identical():
    n = _register("probe", overrides={"origin_stride": 24})
    k = _key(n)
    assert cc.load(n, k) is None, "nothing stored yet"
    cc.store(n, k, _cube(n))
    pd.testing.assert_frame_equal(cc.load(n, k), _cube(n))


def test_the_key_is_written_beside_the_cube_so_a_miss_is_diagnosable():
    """A cache that misses for invisible reasons is worse than no cache."""
    n = _register("probe", overrides={"origin_stride": 24})
    k = _key(n)
    cc.store(n, k, _cube(n))
    written = json.loads(next(cc.CACHE_DIR.glob("*.key.json")).read_text())
    assert written["overrides"] == {"origin_stride": 24}
    assert written["seed"] == 42 and written["cutoffs"] == list(CUTS)


# --------------------------------------------------------------------------- the ways it misses


def test_a_changed_override_misses():
    """The whole point. `lgbm_s24_norm` and `lgbm_s24_recency` differ only here."""
    a = _register("a", overrides={"origin_stride": 24})
    cc.store(a, _key(a), _cube(a))
    b = _register("b", overrides={"origin_stride": 168})
    assert cc.load(b, _key(b)) is None


def test_two_members_with_identical_parameters_still_do_not_share_a_cube():
    """The member name is part of the key: the prediction *column* is named after it."""
    a = _register("a", overrides={"origin_stride": 24})
    cc.store(a, _key(a), _cube(a))
    b = _register("b", overrides={"origin_stride": 24})
    assert cc.load(b, _key(b)) is None


def test_a_different_seed_misses():
    n = _register("probe", overrides={"origin_stride": 24})
    cc.store(n, _key(n, seed=42), _cube(n))
    assert cc.load(n, _key(n, seed=7739)) is None


def test_a_different_training_frame_misses(tmp_path):
    n = _register("probe", overrides={})
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    one.write_text("a\n1\n")
    two.write_text("a\n1\n2\n")
    cc.store(n, _key(n, train_csv=str(one)), _cube(n))
    assert cc.load(n, _key(n, train_csv=str(two))) is None


def test_a_changed_config_file_misses(tmp_path):
    cfg = tmp_path / "lgbm.yaml"
    cfg.write_text("learning_rate: 0.05\n")
    n = _register("probe", overrides={}, config=str(cfg))
    cc.store(n, _key(n), _cube(n))
    cfg.write_text("learning_rate: 0.01\n")
    assert cc.load(n, _key(n)) is None


def test_changed_fitting_code_misses(tmp_path, monkeypatch):
    """An edit to src/models/lgbm.py changes the predictions without changing any parameter."""
    src = tmp_path / "lgbm.py"
    src.write_text("# v1\n")
    n = _register("probe", overrides={})
    mem.get_member(n).run.member_code = (str(src),)
    cc.store(n, _key(n), _cube(n))
    src.write_text("# v2 — a different tree\n")
    assert cc.load(n, _key(n)) is None


# --------------------------------------------------------------------------- fail closed


def test_a_member_that_does_not_declare_its_parameters_is_never_cached():
    """Silence means refit. We cannot tell a stale cube from a fresh one, so we do not try."""
    n = _register("opaque", overrides={}, declare=False)
    k = _key(n)
    assert k is None
    assert cc.store(n, k, _cube(n)) is None
    assert cc.load(n, k) is None
    assert not cc.CACHE_DIR.exists()


def test_a_cube_carrying_nan_predictions_is_refused():
    n = _register("probe", overrides={})
    bad = _cube(n)
    bad.loc[0, n] = float("nan")
    cc.store(n, _key(n), bad)
    assert cc.load(n, _key(n)) is None


def test_a_cube_missing_its_own_prediction_column_is_refused():
    n = _register("probe", overrides={})
    cc.store(n, _key(n), _cube(n).rename(columns={n: "something_else"}))
    assert cc.load(n, _key(n)) is None


def test_a_key_file_that_disagrees_with_the_request_is_refused():
    """Guards a digest collision, which would otherwise be silent and catastrophic."""
    n = _register("probe", overrides={})
    k = _key(n)
    cc.store(n, k, _cube(n))
    kf = next(cc.CACHE_DIR.glob("*.key.json"))
    kf.write_text(json.dumps({**k, "seed": 999}))
    assert cc.load(n, k) is None


def test_the_env_var_disables_both_halves(monkeypatch):
    n = _register("probe", overrides={})
    k = _key(n)
    cc.store(n, k, _cube(n))
    monkeypatch.setenv("MEMBER_CUBE_CACHE", "0")
    assert cc.load(n, k) is None
    assert cc.store(n, k, _cube(n)) is None


def test_a_half_written_cube_is_not_readable_as_real():
    """store() writes through a temp file, so a Ctrl-C leaves no partial parquet in place."""
    n = _register("probe", overrides={})
    cc.store(n, _key(n), _cube(n))
    assert not list(cc.CACHE_DIR.glob("*.tmp"))


# --------------------------------------------------------------------------- the real registry


def test_every_registered_lgbm_member_declares_its_parameters():
    """If this fails, Phase 4 silently lost its cache and every A/B refits both arms again."""
    lgbm_members = [n for n in mem.available_members() if n.startswith("lgbm")]
    assert lgbm_members, "no lgbm members registered"
    undeclared = [n for n in lgbm_members if not hasattr(mem.get_member(n).run, "member_overrides")]
    assert undeclared == []


def test_the_four_phase4_composites_have_four_distinct_keys():
    """They differ by one override each; a shared key would cross-contaminate all four."""
    names = [f"lgbm_s24_{lv}" for lv in ("unitcat", "norm", "recency", "wlag")]
    keys = {n: _key(n) for n in names}
    digests = {cc._digest(k) for k in keys.values()}
    assert len(digests) == 4
