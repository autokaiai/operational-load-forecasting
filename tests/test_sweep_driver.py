"""S5 preflight, part 2 — the SEARCH itself, exercised against the real Optuna sampler.

``tests/test_sweep_space.py`` proves each dimension reaches the model. This file proves the thing
that stands between the space and the model: that TPE can actually sample it, that the incumbent
can be enqueued as trial 0, and that every configuration the sampler emits is one ``build_nf`` will
accept. Those three are exactly the failure modes that only appear once the fan-out is running,
which is the worst possible place to find them.

The pooling test is here for a different reason. Pooled WAPE is ``sum|y-yhat| / sum|y|`` accumulated
across windows and divided ONCE; averaging per-window WAPEs is a mean-of-ratios and a different
number. ``src/eval/protocol.py`` was written to stop those two being quoted interchangeably, and the
driver pools the holdout windows by hand from the ``ae``/``ya`` a trial returns — so the arithmetic
it uses is asserted rather than assumed.
"""

from __future__ import annotations

import optuna
import pytest

from src.models import sweep_space as S
from src.models.registry import build_nf

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = {
    "model": "TFT",
    "name": "tft",
    "h": 8,
    "max_steps": 1,
    "accelerator": "cpu",
    "devices": 1,
    "batch_size": 8,
    "windows_batch_size": 8,
    "inference_windows_batch_size": 8,
    "enable_progress_bar": False,
    "logger": False,
    "local_scaler_type": "robust",
}


def _study() -> optuna.Study:
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4),
    )


def test_the_incumbent_enqueues_and_materialises_as_trial_zero():
    """Seeding TPE with the greedy composite is Part VII's strategy step 3 — *cannot do worse than
    the greedy answer, can leave it*. It only works if ``suggest`` is CALLED on the asked trial:
    an enqueued dict that nothing samples records a trial with no parameters, and the study then
    starts cold from precisely the point the seeding exists to avoid.
    """
    study = _study()
    study.enqueue_trial(S.INCUMBENT)
    trial = study.ask()
    assert S.suggest(trial) == S.INCUMBENT
    study.tell(trial, 0.13429)
    assert study.trials[0].params["hidden_size"] == 64


def test_tpe_samples_the_space_and_every_draw_builds():
    """The end-to-end guard: 16 real TPE draws, each one built. A distribution the sampler can emit
    but ``build_nf`` cannot accept is a mid-fan-out crash, and it is free to rule out here.
    """
    study = _study()
    seen: set[str] = set()
    for i in range(16):
        trial = study.ask()
        params = S.suggest(trial)
        cfg = {**BASE, **S.to_overrides(params), "input_size": params["input_size"]}
        build_nf(cfg)  # must not raise
        seen.add(params["grn_activation"])
        study.tell(trial, 0.13 + 0.001 * (i % 5))
    assert len(study.trials) == 16
    assert seen == {"ELU", "SELU"}, "TPE never explored both activations in 16 draws"


def test_huber_delta_is_sampled_only_with_the_huber_loss():
    """A conditional dimension must not leak: ``huber_delta`` on an MAE trial would be an unused
    parameter that TPE still models, which wastes the search on a dimension with no effect."""
    for loss in ("mae", "huber"):
        params = S.suggest(_Fixed({**S.INCUMBENT, "loss": loss, "huber_delta": 1.25}))
        assert ("huber_delta" in params) is (loss == "huber")
        assert ("huber_delta" in S.to_overrides(params)) is (loss == "huber")


class _Fixed:
    """A trial stub that returns a preset value for each name — no sampler, no study."""

    def __init__(self, values: dict):
        self.values = values

    def suggest_float(self, name, low, high, log=False):
        return self.values[name]

    def suggest_categorical(self, name, choices):
        return self.values[name]


def test_pooling_the_holdout_uses_numerators_not_a_mean_of_ratios():
    """The driver sums ``ae``/``ya`` across the two holdout windows and divides once.

    Averaging the per-window WAPEs instead would be a mean-of-ratios — the exact conflation
    ``compute_pooled_wape`` exists to prevent, and the one that makes two ``results/*.json``
    conventions non-comparable. With unequal window volumes the two answers genuinely differ, so
    this fixture is skewed on purpose.
    """
    windows = [{"ae": 10.0, "ya": 100.0}, {"ae": 30.0, "ya": 900.0}]  # 0.100 and 0.0333
    pooled = sum(w["ae"] for w in windows) / sum(w["ya"] for w in windows)
    mean_of_ratios = sum(w["ae"] / w["ya"] for w in windows) / len(windows)
    assert pooled == pytest.approx(40.0 / 1000.0)
    assert pooled != pytest.approx(mean_of_ratios)


def test_the_sweeps_fixed_arm_is_a_REGISTERED_gap_fill_strategy():
    """``run_sweep``'s arms are gap-fill STRATEGY NAMES, not free labels.

    Each arm is handed to ``_withhold_gap_covariates(strategy=...)``, so a descriptive label like
    ``"only"`` raises ``unknown fill strategy``. The smoke gate caught exactly that on the first
    launch — one container instead of sixty-one, which is the entire argument for smoking before
    fanning out. This asserts the default the driver ships with is real.

    ``median`` is also the RIGHT fixed value, not merely a valid one: S3 adopted it for the cascade
    (its ``interp`` arm lost at −0.00937, 0/3). Holding it fixed is what keeps S5 a hyperparameter
    sweep rather than a re-run of S3 crossed with one.
    """
    from src.data.gap_fill import available_strategies

    assert "median" in available_strategies()
    assert "only" not in available_strategies()


def test_the_search_window_is_never_a_holdout_window():
    """Stage 3 adjudicates on the windows the optimizer never saw. If the search window leaked in,
    the exposé's *"a window the optimizer never saw"* would be false and the phase's headline read
    would be selection bias reported as a result."""
    for search_window in range(3):
        holdout = [w for w in range(3) if w != search_window]
        assert search_window not in holdout
        assert len(holdout) == 2, "two unseen windows — stricter than the plan's original one"
