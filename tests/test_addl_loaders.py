"""Guards for the S8 additional-dataset loaders (plan S8.0).

The bug this exists to prevent: the loader used to wrap its real-data path in a bare
``except Exception`` that fell through to a 6-series synthetic fixture. With
``datasetsforecast`` absent (local) or buggy (in-container), that path *always* fired — so the
transfer phase would have trained the whole panel on a toy fixture and written the result to
``results/addl_dataset_metrics.json`` as a graded deliverable.

``test_missing_corpus_raises`` is the regression test proper: it needs no data and would have
failed on the old code. The rest assert the real corpus loads with the right shape, and are
skipped when it has not been staged.
"""

from __future__ import annotations

import pytest

from src.data.external_paths import resolve
from src.data.m5_loader import (
    M5_STATIC_PARQUET,
    load_m5_long,
    m5_futr_exog_list,
    m5_stat_exog_list,
)


def _staged(relpath: str) -> bool:
    try:
        resolve(relpath)
    except FileNotFoundError:
        return False
    return True


needs_m5 = pytest.mark.skipif(
    not _staged(M5_STATIC_PARQUET),
    reason="M5 corpus not staged",
)


def test_missing_corpus_raises(tmp_path, monkeypatch):
    """A missing corpus must FAIL, never silently downgrade to the synthetic fixture.

    This is the one that would have caught the original defect: on the old code the loader
    swallowed the error and returned ``synthetic=True`` data.
    """
    monkeypatch.setenv("ADDL_DATA_ROOT", str(tmp_path))  # exclusive override -> empty root
    with pytest.raises(FileNotFoundError):
        load_m5_long(n_series=2)


def test_synthetic_is_opt_in_only():
    """The fixture is still reachable for CI/plumbing, but only when explicitly asked for."""
    bundle = load_m5_long(synthetic=True)
    assert bundle.synthetic is True
    assert bundle.meta["source"] == "synthetic"


@needs_m5
def test_m5_loads_real_data():
    b = load_m5_long(n_series=20)
    assert b.synthetic is False, "M5 silently fell back to the synthetic fixture"
    assert b.meta["source"].startswith("parquet:")
    assert b.long_df["unique_id"].nunique() == 20
    assert set(b.long_df.groupby("unique_id").size().unique()) == {1941}  # d_1..d_1941
    assert all(c in b.long_df.columns for c in m5_futr_exog_list())
    assert not b.long_df[m5_futr_exog_list()].isna().any().any()
    assert (b.long_df["y"] >= 0).all()
    # statics must cover exactly the loaded series, label-encoded for neuralforecast
    assert len(b.static_df) == 20
    assert set(b.static_df["unique_id"]) == set(b.long_df["unique_id"])
    for col in m5_stat_exog_list():
        assert str(b.static_df[col].dtype) == "int64"
