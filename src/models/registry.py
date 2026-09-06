"""Build a neuralforecast model + wrapper from a plain config dict.

Centralises every architecture in the vanilla family sweep behind one ``build_nf`` so a config
only names the model and its hyperparameters. Exogenous lists come from ``src.data.features``
(never the config) and are attached per model from its neuralforecast capability flags
(``EXOGENOUS_FUTR`` / ``EXOGENOUS_STAT``), so a covariate-free model (DLinear, PatchTST) is
never handed exog it would reject, and a futr-only model (TimesNet, Informer) never gets a
static list.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, HuberLoss
from neuralforecast.models import (
    KAN,
    LSTM,
    MLP,
    NHITS,
    TFT,
    Autoformer,
    BiTCN,
    DLinear,
    FEDformer,
    Informer,
    NBEATSx,
    PatchTST,
    TiDE,
    TimesNet,
    TSMixerx,
    iTransformer,
)

from src.data.features import futr_exog_list, stat_exog_list
from src.models.mamba_tft import MambaTFT

# Every architecture in the sweep, keyed by its exact neuralforecast class name (the value of
# `model:` in a config). Exog wiring is derived from each class's capability flags in
# build_model — never hand-maintained here.
MODELS = {
    # Sprint-1 POC
    "DLinear": DLinear,
    "TiDE": TiDE,
    "TSMixerx": TSMixerx,
    "TFT": TFT,
    # transformers
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
    "Informer": Informer,
    "Autoformer": Autoformer,
    "FEDformer": FEDformer,
    # MLP / basis-expansion
    "NHITS": NHITS,
    "NBEATSx": NBEATSx,
    "MLP": MLP,
    "KAN": KAN,
    # conv / frequency
    "BiTCN": BiTCN,
    "TimesNet": TimesNet,
    # recurrent
    "LSTM": LSTM,
    # IDEAS #16 — OUR model, not neuralforecast's: the shipped TFT with its two LSTM encoders
    # replaced by a selective state-space (Mamba) stack. It subclasses the installed TFT and
    # adopts its sub-modules (see src/models/mamba_tft.py); the package is never modified.
    "MambaTFT": MambaTFT,
}

# Config keys consumed by the wrapper / runner rather than the model constructor.
#
# ANYTHING ADDED HERE IS LOAD-BEARING. ``TFT.__init__`` ends in ``**trainer_kwargs``, so a config
# key that is neither a constructor parameter nor listed here is **accepted at construction** and
# only raises from inside Lightning at ``nf.fit`` — measured:
#
#     TFT(h=4, input_size=8, hiden_size=99)          -> builds, stores it in trainer_kwargs
#     nf.fit(...)                                    -> TypeError: Trainer.__init__() got an
#                                                       unexpected keyword argument 'hiden_size'
#
# i.e. ``build_nf(cfg)`` is NOT validation, only a fit is. ``tests/test_sweep_space.py`` asserts
# every key the S5 sampler can emit is either a constructor parameter or a member of this set.
_NON_MODEL_KEYS = {
    "model",
    "name",
    "seed",
    "freq",
    "local_scaler_type",
    "loss",
    "val_size",
    "sample_weight",  # data-side flag (volume-weighted L1); consumed in nf_fit_and_sweep
    # S5: consumed by build_loss below, never by the model constructor.
    "huber_delta",
    "loss_mask_gap",
    "score_len",
    # Final-push lane 1G: a data-side target transform (train on ``y - <channel>``, add the channel
    # back at inference). Consumed in nf_fit_and_sweep, exactly like `sample_weight`, and it must
    # never reach a model constructor.
    "residual_target",
    # Checkpoint averaging (S5 Tier D, never spawned). Consumed by build_model below, which turns
    # it into a Lightning callback on `trainer_kwargs["callbacks"]`. NOT a constructor parameter.
    "swa",
}

# Point losses S5 may sweep. Deliberately point-only: a quantile head changes the output width and
# the prediction column, which is S7's experiment, not a dimension of this one. S7 becomes a config
# value rather than a new code path because the plumbing below already exists by then.
POINT_LOSSES = {"mae": MAE, "huber": HuberLoss}


def horizon_weight(cfg: dict, h: int | None = None) -> torch.Tensor | None:
    """Mask the training loss to the GRADED tail of the horizon.

    We fit at ``h = 672`` (336h gap + 336h scored) but only the last 336 steps are ever scored, so
    by default two thirds of the loss signal is spent on a block nobody grades. ``horizon_weight``
    is a first-class ``BasePointLoss`` argument — a length-``h`` vector that ``_compute_weights``
    multiplies into the mask — so this needs **no custom loss class**, which is why it stopped
    being the one dimension in that phase carrying new numerics.

    Returns ``None`` (i.e. the unweighted default) when the flag is off or when the horizon has no
    gap to mask, so a non-gapped fit silently does the right thing instead of asserting.
    """
    if not cfg.get("loss_mask_gap"):
        return None
    h = int(h if h is not None else cfg["h"])
    scored = int(cfg.get("score_len", 336))
    if h <= scored:
        return None
    return torch.tensor([0.0] * (h - scored) + [1.0] * scored)


def build_loss(cfg: dict, h: int | None = None):
    """Resolve the training loss from config. Defaults to ``MAE()`` — the historical behaviour.

    Until S5 this function did not exist: ``build_model`` set ``kwargs["loss"] = MAE()`` and
    ``"loss"`` was dropped as a non-model key, so a config ``loss`` string was read by nothing.
    Sweeping it would have trained N identical models and returned a confident null —
    the ``linear_tree``-under-L1 failure mode, which is why the reachability tests exist.

    Absent a ``loss`` key this returns ``MAE()`` with no horizon weight, so **every recorded number
    reproduces bit-for-bit**; the S4 control measured that drift guard at ``max|Δpred| = 0``.
    """
    spec = cfg.get("loss") or "mae"
    if not isinstance(spec, str):
        return spec  # an already-built loss instance; callers that construct their own win
    name = spec.lower()
    if name not in POINT_LOSSES:
        raise ValueError(f"Unknown loss {spec!r}; choose from {sorted(POINT_LOSSES)}")
    kwargs: dict = {}
    if name == "huber":
        kwargs["delta"] = float(cfg.get("huber_delta", 1.0))
    weight = horizon_weight(cfg, h)
    if weight is not None:
        kwargs["horizon_weight"] = weight
    return POINT_LOSSES[name](**kwargs)


def supports_futr(name: str) -> bool:
    """True if the model conditions on known-future covariates (drives futr_df at inference)."""
    return bool(MODELS[name].EXOGENOUS_FUTR)


def supports_stat(name: str) -> bool:
    """True if the model accepts static per-series covariates (drives static_df at fit time)."""
    return bool(MODELS[name].EXOGENOUS_STAT)


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch for a reproducible single-seed POC run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict):
    """Instantiate one neuralforecast model from a config dict (loss + exog handled here)."""
    name = cfg["model"]
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")

    kwargs = {k: v for k, v in cfg.items() if k not in _NON_MODEL_KEYS}
    # L1 by default — the WAPE training proxy. A config `loss` is now honoured (S5/#55) instead of
    # being silently overwritten; `valid_loss` stays unset so early stopping monitors the SAME
    # objective, which is what makes an aligned loss also align the early-stop monitor.
    kwargs["loss"] = build_loss(cfg, cfg.get("h"))
    # Quiet, headless defaults; a config may override (e.g. accelerator="cpu" for tests).
    kwargs.setdefault("enable_progress_bar", False)
    kwargs.setdefault("logger", False)

    # Exog is derived from the model's capability flags, never the config: attach futr/static
    # only to architectures that accept them. We keep the conditioning set at futr+stat (gated
    # by capability) for parity with the recorded baselines and never attach hist_exog.
    cls = MODELS[name]
    for key in ("futr_exog_list", "stat_exog_list", "hist_exog_list"):
        kwargs.pop(key, None)
    if cls.EXOGENOUS_FUTR:
        kwargs["futr_exog_list"] = futr_exog_list()
    if cls.EXOGENOUS_STAT:
        kwargs["stat_exog_list"] = stat_exog_list()

    # S5 Tier D, finally run: checkpoint averaging. `swa` is a dict of StepCheckpointAverage kwargs
    # (or `True` for the defaults); anything falsy leaves the model bit-identical to the incumbent.
    #
    # It has to be built HERE rather than named in a YAML because a callback is a Python object, and
    # it goes onto `callbacks` because neuralforecast APPENDS its EarlyStopping to whatever list it
    # finds there (`_base_model.py`) rather than overwriting it — so both callbacks survive.
    #
    # Lightning's own StochasticWeightAveraging cannot be used: it is epoch-based and asserts
    # `trainer.max_epochs is not None`, which is None under neuralforecast's step-based training,
    # and neuralforecast raises on a `max_epochs` trainer kwarg. See src/models/swa.py.
    swa = cfg.get("swa")
    if swa:
        from src.models.swa import StepCheckpointAverage

        opts = {} if swa is True else dict(swa)
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(StepCheckpointAverage(**opts))
        kwargs["callbacks"] = callbacks

    return cls(**kwargs)


def build_nf(cfg: dict) -> NeuralForecast:
    """Seed, build the model, and wrap it in a NeuralForecast (hourly freq, per-series scaler)."""
    set_seed(int(cfg.get("seed", 42)))
    model = build_model(cfg)
    return NeuralForecast(
        models=[model],
        freq=cfg.get("freq", "h"),
        local_scaler_type=cfg.get("local_scaler_type"),
    )
