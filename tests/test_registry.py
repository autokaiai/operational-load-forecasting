"""Capability-aware exogenous wiring + a NHITS end-to-end smoke.

The CLI contract test (``test_predict.py``) only exercises DLinear, which takes NO exogenous
inputs. NHITS is our futr+stat workhorse — the path that decides whether the covariate models
train at all — so this guards two things the DLinear test cannot:

1. ``build_model`` attaches *exactly* the futr/stat lists each architecture's capability flags
   allow (full / futr-only / none), never lists pulled from the config.
2. NHITS actually fits with a ``static_df`` and forecasts with a ``futr_df`` through ``build_nf``.

(The OOM that stalled NHITS in the family sweep was GPU-memory only — handled separately in
``configs/nhits.yaml``. The wiring exercised here is the part that must stay correct.)
"""

from __future__ import annotations

import pytest

pytest.importorskip("neuralforecast")

from src.data.features import STATIC_COLS, futr_exog_list, stat_exog_list  # noqa: E402
from src.data.loader import NF_ID, NF_TARGET, NF_TIME  # noqa: E402
from src.models.registry import build_model, build_nf, supports_futr, supports_stat  # noqa: E402

# Tiny, CPU-only, single optimisation step — enough to construct/fit without spending real time.
_TINY = {"h": 5, "input_size": 24, "max_steps": 1, "accelerator": "cpu", "devices": 1}


def test_capability_flags_match_known_families() -> None:
    """The flags that drive exog wiring are what we expect per family."""
    assert supports_futr("NHITS") and supports_stat("NHITS")  # full exog
    assert not supports_futr("DLinear") and not supports_stat("DLinear")  # covariate-free
    assert supports_futr("Informer") and not supports_stat("Informer")  # futr-only


def test_build_model_attaches_capability_gated_exog() -> None:
    """build_model hands NHITS both lists and DLinear neither — derived from flags, not config."""
    nhits = build_model({"model": "NHITS", **_TINY})
    assert set(nhits.futr_exog_list) == set(futr_exog_list())
    assert set(nhits.stat_exog_list) == set(stat_exog_list())

    dlinear = build_model({"model": "DLinear", **_TINY})
    assert not dlinear.futr_exog_list
    assert not dlinear.stat_exog_list


def test_nhits_fits_and_predicts_through_registry() -> None:
    """NHITS trains with a static_df and forecasts with a futr_df end-to-end (CPU, tiny)."""
    import numpy as np
    import pandas as pd

    per, h = 60, _TINY["h"]
    series = ["unit_000", "unit_001"]
    futr_cols = futr_exog_list()
    rng = np.random.default_rng(0)

    frames = []
    for sid in series:
        ds = pd.date_range("2023-01-01", periods=per, freq="h")
        y = 10 + 3 * np.sin(np.arange(per) * 2 * np.pi / 24) + rng.normal(0, 0.3, per)
        cols = {NF_ID: sid, NF_TIME: ds, NF_TARGET: y}
        for c in futr_cols:
            cols[c] = rng.normal(0, 1, per)
        frames.append(pd.DataFrame(cols))
    long_df = pd.concat(frames, ignore_index=True)
    static_df = pd.DataFrame({NF_ID: series, **{c: [1.0, 2.0] for c in STATIC_COLS}})

    nf = build_nf({"model": "NHITS", "name": "nhits", "seed": 42, **_TINY})
    nf.fit(long_df, static_df=static_df)

    last = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=per - 1)
    fut = []
    for sid in series:
        fds = pd.date_range(last + pd.Timedelta(hours=1), periods=h, freq="h")
        cols = {NF_ID: sid, NF_TIME: fds}
        for c in futr_cols:
            cols[c] = rng.normal(0, 1, h)
        fut.append(pd.DataFrame(cols))
    futr_df = pd.concat(fut, ignore_index=True)

    preds = nf.predict(futr_df=futr_df)
    assert len(preds) == len(series) * h
    assert "NHITS" in preds.columns
    assert np.isfinite(preds["NHITS"]).all()
