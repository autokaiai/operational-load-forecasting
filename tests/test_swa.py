"""Checkpoint averaging — does the lever REACH the model?

Plan law 5: *a lever that does not reach the model returns a confident null*, and this project
has three recorded instances (``registry`` overwriting a config ``loss``; ``random_seed``
defaulting to 1 inside neuralforecast; ``TrainingArguments.seed`` defaulting to 42 inside the HF
Trainer). Lightning's own ``StochasticWeightAveraging`` would have been the fourth: it is
epoch-based, and ``pl.Trainer(max_steps=...).max_epochs`` is ``None``, so it cannot initialise
under neuralforecast — while neuralforecast itself raises on the ``max_epochs`` kwarg that would
fix it. Both halves of that are pinned below, so a version bump cannot quietly turn the shipped
lever back into a no-op.

The substantive assertion is ``test_swa_changes_the_trained_weights``: two fits at the identical
seed that differ ONLY in the ``swa`` flag must produce DIFFERENT weights. Asserting that the
callback ran would not be evidence — ``a test that cannot fail is not evidence`` (law 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("neuralforecast")

from src.data.features import STATIC_COLS, futr_exog_list  # noqa: E402
from src.data.loader import NF_ID, NF_TARGET, NF_TIME  # noqa: E402
from src.models.registry import build_nf  # noqa: E402
from src.models.swa import StepCheckpointAverage  # noqa: E402

_TINY = {
    "h": 8,
    "input_size": 16,
    "max_steps": 12,
    "val_check_steps": 100,
    "early_stop_patience_steps": -1,
    "batch_size": 4,
    "windows_batch_size": 8,
    "inference_windows_batch_size": 8,
    "accelerator": "cpu",
    "devices": 1,
    "hidden_size": 8,
    "n_head": 1,
    "enable_progress_bar": False,
    "scaler_type": "identity",
}


def _panel(n_series: int = 3, n_obs: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A tiny panel carrying the SAME exog columns the registry attaches by capability flag."""
    rng = np.random.default_rng(0)
    series = [f"unit_{i:03d}" for i in range(n_series)]
    rows = []
    for sid in series:
        t = pd.date_range("2023-01-01", periods=n_obs, freq="h")
        y = 10 + 3 * np.sin(np.arange(n_obs) / 24 * 2 * np.pi) + rng.normal(0, 0.4, n_obs)
        cols = {NF_ID: sid, NF_TIME: t, NF_TARGET: y}
        for c in futr_exog_list():
            cols[c] = rng.normal(0, 1, n_obs)
        rows.append(pd.DataFrame(cols))
    static_df = pd.DataFrame(
        {NF_ID: series, **{c: rng.normal(0, 1, n_series) for c in STATIC_COLS}}
    )
    return pd.concat(rows, ignore_index=True), static_df


def _fit(swa) -> dict[str, torch.Tensor]:
    """Fit a tiny TFT at a FIXED seed and return its float parameters."""
    cfg = {"model": "TFT", "name": "t", "seed": 42, "random_seed": 42, **_TINY}
    if swa is not None:
        cfg["swa"] = swa
    nf = build_nf(cfg)
    long_df, static_df = _panel()
    nf.fit(df=long_df, static_df=static_df)
    model = nf.models[0]
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.dtype.is_floating_point}


def test_lightning_swa_cannot_run_under_neuralforecast() -> None:
    """The reason src/models/swa.py exists. Both halves, pinned against a version bump."""
    import pytorch_lightning as pl

    trainer = pl.Trainer(
        max_steps=5, logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    # Half 1: Lightning gives max_epochs=None when only max_steps is set, and SWA asserts on it.
    assert trainer.max_epochs is None

    # Half 2: the kwarg that would fix half 1 is refused by neuralforecast.
    from neuralforecast.models import TFT

    with pytest.raises(Exception, match="max_epochs is deprecated"):
        TFT(h=4, input_size=8, max_epochs=3)


def test_swa_changes_the_trained_weights() -> None:
    """LAW 5. Same seed, same data, one flag apart — the weights must actually move."""
    base = _fit(None)
    avg = _fit({"every_n_steps": 2, "tail_fraction": 0.5})

    assert set(base) == set(avg)
    max_delta = max(float((base[n] - avg[n]).abs().max()) for n in base)
    assert max_delta > 1e-6, f"swa=True left the weights unchanged (max|delta| = {max_delta})"


def test_swa_off_is_bit_identical_to_the_incumbent() -> None:
    """The control arm must be untouched, or the A/B measures the plumbing and not the lever."""
    a = _fit(None)
    b = _fit(False)
    for n in a:
        assert torch.equal(a[n], b[n]), f"swa=False perturbed {n}"


class _FakeTrainer:
    def __init__(self, step: int = 0) -> None:
        self.global_step = step


def _drive(cb: StepCheckpointAverage, n: int) -> torch.nn.Linear:
    """Fill a Linear's weight with 0..n-1, snapshotting each, then end training at 99."""
    module = torch.nn.Linear(3, 2)
    cb.on_fit_start(None, module)  # type: ignore[arg-type]
    with torch.no_grad():
        for i in range(n):
            module.weight.fill_(float(i))
            cb._snapshot(module)
        module.weight.fill_(99.0)  # the final weights, also included by on_train_end
    cb.on_train_end(_FakeTrainer(n), module)  # type: ignore[arg-type]
    return module


def test_averaging_is_the_mean_of_the_tail_fraction() -> None:
    """The arithmetic itself, on a module whose 'training' we control exactly."""
    cb = StepCheckpointAverage(every_n_steps=1, tail_fraction=0.25)
    module = _drive(cb, 11)  # 11 taken + the final one = 12; 25% of 12 = 3
    assert cb.n_taken == 12
    assert cb.n_averaged == 3
    # the last three snapshots are weights 9, 10 and the final 99
    assert module.weight.detach().unique().tolist() == pytest.approx([np.mean([9.0, 10.0, 99.0])])


def test_the_window_scales_with_run_length_not_with_step_count() -> None:
    """THE BUG THIS REPLACED. A 2.19x spread in run length must NOT change the fraction averaged.

    The fixed-k version averaged ~52% of the short run and ~24% of the long one, and the measured
    delta tracked that fraction on 3/3 windows. Under a fractional window the ratio is invariant.
    """
    for n in (20, 44):  # the measured 2.19x spread
        cb = StepCheckpointAverage(every_n_steps=1, tail_fraction=0.25)
        _drive(cb, n)
        frac = cb.n_averaged / cb.n_taken
        assert 0.22 <= frac <= 0.30, f"n={n}: averaged {frac:.2%}, expected ~25%"


def test_short_runs_still_average_at_least_the_floor() -> None:
    """A run so short that 25% rounds to one snapshot must still average, not silently no-op."""
    cb = StepCheckpointAverage(every_n_steps=1, tail_fraction=0.25, min_snapshots=2)
    module = _drive(cb, 2)  # 3 snapshots total; 25% -> 1, floored to 2
    assert cb.n_averaged == 2
    assert module.weight.detach().unique().tolist() == pytest.approx([np.mean([1.0, 99.0])])


def test_bad_configurations_are_refused() -> None:
    """A no-op configuration must fail loudly rather than return a confident null."""
    with pytest.raises(ValueError, match="min_snapshots must be >= 2"):
        StepCheckpointAverage(min_snapshots=1)
    with pytest.raises(ValueError, match="every_n_steps must be > 0"):
        StepCheckpointAverage(every_n_steps=0)
    with pytest.raises(ValueError, match="tail_fraction must be in"):
        StepCheckpointAverage(tail_fraction=0.0)
    with pytest.raises(ValueError, match="tail_fraction must be in"):
        StepCheckpointAverage(tail_fraction=1.5)


def test_swa_is_not_a_model_constructor_key() -> None:
    """`swa` must be routed, never forwarded — TFT swallows unknown kwargs into trainer_kwargs."""
    from src.models.registry import _NON_MODEL_KEYS, build_model

    assert "swa" in _NON_MODEL_KEYS
    model = build_model({"model": "TFT", "name": "t", "swa": True, **_TINY})
    assert "swa" not in model.trainer_kwargs
    cbs = model.trainer_kwargs.get("callbacks") or []
    assert any(isinstance(c, StepCheckpointAverage) for c in cbs)
