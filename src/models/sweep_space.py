"""Plan S5 — the TFT backbone search space, declared once so the tests can iterate it.

The exposé promised *"a two-stage Bayesian hyperparameter search (Optuna TPE, ~40 trials) against
vanilla TFT as the null"* and no step ever ran it: ten levers were spent on a tree worth 0.1585
while the model that ships sits at a hand-set ``hidden_size: 64``.

WHY THIS IS A DATA STRUCTURE AND NOT A ``suggest_*`` BLOCK INSIDE THE DRIVER
---------------------------------------------------------------------------
Because the expensive failure in this phase is not a crash, it is a **lever that does not reach the
model**. This project has now hit that three times — ``linear_tree`` silently broken under L1,
``get_activation_fn("GELU")`` silently returning ``F.elu``, and ``configs/tft_chronos.yaml`` inert
while looking functional — and the S5 audit found three more sitting in our own code:

  * ``registry.py`` dropped a config ``loss`` and overwrote it with ``MAE()``,
  * ``nf_fit_and_sweep`` never attached ``add_volume_sample_weight``,
  * TFT's own ``random_seed`` is never passed, so seed-bagging could average five identical models.

Each would have produced a *confident null* — 32 trials, a clean CI, and nothing under test. So the
space is declared as data, and ``tests/test_sweep_space.py`` walks it: every emitted key must be a
real constructor parameter, every dimension must be shown to change the built model or the fitted
predictions, and every corner must build.

THE OTHER TRAP, MEASURED
------------------------
``TFT.__init__`` ends in ``**trainer_kwargs``, so a misspelled key is **accepted at construction**
and only raises inside Lightning at ``nf.fit``. In a 32-container fan-out that is a late failure in
every container at once. ``sampled_keys()`` exists so the totality test can rule it out statically,
which is stronger than catching it in a fit: a static check cannot be defeated by a value that
happens to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The graded slice. 672 = 336h gap + 336h scored; only the tail is ever scored.
SCORE_LEN = 336


@dataclass(frozen=True)
class Dim:
    """One swept dimension.

    ``reach`` names how ``tests/test_sweep_space.py`` proves the dimension reaches the model:

      ``attr``   the built model carries the value on a named attribute (default: the dim's name)
      ``hparam`` the value is only visible in ``model.hparams`` (it is consumed by a submodule)
      ``params`` two values must produce a different parameter count
      ``loss``   two values must produce a different loss VALUE on a fixed synthetic batch
      ``fit``    data-side; only a real 2-step fit can show it (predictions must differ)
    """

    name: str
    kind: str  # "float" | "int" | "cat"
    reach: str
    values: tuple = ()
    low: float = 0.0
    high: float = 0.0
    log: bool = False
    attr: str = ""
    note: str = ""
    depends_on: tuple[str, Any] = field(default=())

    def probe(self) -> tuple:
        """Two values that must be distinguishable — what the reachability test compares."""
        return (self.values[0], self.values[-1]) if self.values else (self.low, self.high)


# --------------------------------------------------------------------------- the space
# Ordered by how much new code each needed, which is also the risk order: everything above
# `sample_weight` was already wired straight through `build_model` to the constructor.
DIMENSIONS: tuple[Dim, ...] = (
    Dim(
        "learning_rate",
        "float",
        "attr",
        low=3e-4,
        high=3e-3,
        log=True,
        note="class 1, and the one every other continuous dim is coupled to",
    ),
    Dim(
        "hidden_size",
        "int",
        "params",
        values=(32, 64, 128, 192),
        note="the ship model is at a hand-set 64 and nobody ever moved it",
    ),
    Dim("dropout", "float", "hparam", low=0.0, high=0.3),
    Dim(
        "attn_dropout",
        "float",
        "hparam",
        low=0.0,
        high=0.3,
        note="never touched, and it is the TFT's own attention regulariser",
    ),
    Dim("n_head", "int", "params", values=(2, 4, 8)),
    Dim(
        "input_size",
        "int",
        "attr",
        values=(336, 504, 672),
        note="lookback has never been swept at all",
    ),
    Dim(
        "grn_activation",
        "cat",
        "attr",
        values=("ELU", "SELU"),
        note="the uncommitted sweep's SELU 0.1372 vs ELU 0.1411, confirmed on the cube",
    ),
    Dim(
        "scaler_type",
        "cat",
        "params",
        values=("identity", "robust", "revin"),
        note="#42. 'revin' shares standard's callables but adds learnable affine params, "
        "which is why the reachability proof is a parameter count",
    ),
    Dim(
        "sample_weight",
        "cat",
        "fit",
        values=("none", "volume"),
        note="3.3 / #38 — the aligned loss. Data-side, so only a fit can show it.",
    ),
    Dim(
        "loss",
        "cat",
        "loss",
        values=("mae", "huber"),
        note="#55, and the plumbing it needs is also S7's prerequisite",
    ),
    Dim(
        "huber_delta",
        "float",
        "loss",
        low=0.5,
        high=2.0,
        depends_on=("loss", "huber"),
        note="only sampled when the loss is Huber",
    ),
    Dim(
        "loss_mask_gap",
        "cat",
        "loss",
        values=(False, True),
        note="#51 — mask the loss to the graded 336. A BasePointLoss horizon_weight, so no "
        "custom loss class is needed.",
    ),
)

DIMS_BY_NAME = {d.name: d for d in DIMENSIONS}

# Trial 0, enqueued via `study.enqueue_trial`. TPE starts from a known-good point rather than cold
# — it cannot do worse than the incumbent and it can leave it (Part VII, strategy step 3).
#
# It is also the CONTROL, re-run in the same batch. S4's lesson was exactly that, and it paid:
# the freshly-refitted control reproduced its stored cube at max|Δpred| = 0. Here the drift guard
# costs nothing at all, because the seeded trial had to run anyway.
INCUMBENT: dict[str, Any] = {
    "learning_rate": 0.001,
    "hidden_size": 64,
    "dropout": 0.1,
    "attn_dropout": 0.0,
    "n_head": 4,
    "input_size": 504,
    "grn_activation": "ELU",  # shipped setting; SELU reverted 2026-08-05, see docs/method.md
    "scaler_type": "identity",
    "sample_weight": "none",
    "loss": "mae",
    "loss_mask_gap": False,
}


def sampled_keys() -> set[str]:
    """Every config key a trial can emit. The totality test's left-hand side."""
    return {d.name for d in DIMENSIONS}


def suggest(trial) -> dict[str, Any]:
    """Sample one config from the space (an ``optuna.Trial``, or anything with the same API)."""
    params: dict[str, Any] = {}
    for dim in DIMENSIONS:
        if dim.depends_on:
            key, required = dim.depends_on
            if params.get(key) != required:
                continue
        if dim.kind == "float":
            params[dim.name] = trial.suggest_float(dim.name, dim.low, dim.high, log=dim.log)
        elif dim.kind == "int":
            params[dim.name] = trial.suggest_categorical(dim.name, list(dim.values))
        else:
            params[dim.name] = trial.suggest_categorical(dim.name, list(dim.values))
    return params


def to_overrides(params: dict[str, Any]) -> dict[str, Any]:
    """Project sampled parameters onto config overrides that ``build_nf`` understands.

    Deliberately thin. Every dimension is already a config key that ``build_model`` either forwards
    to the constructor or that ``_NON_MODEL_KEYS`` routes to ``build_loss`` — the projection exists
    so a future dimension that DOES need translating has one obvious place to live, not so that
    today's dimensions get renamed on the way through.
    """
    out = dict(params)
    if out.get("sample_weight") == "none":
        out["sample_weight"] = None
    if out.get("loss") != "huber":
        out.pop("huber_delta", None)
    out["score_len"] = SCORE_LEN
    return out


def validate_space() -> None:
    """Structural invariants of the space itself. Cheap, and it runs before every launch.

    ``n_head`` must divide ``hidden_size`` — multi-head attention splits the hidden dimension, so a
    pair that does not divide is a runtime error deep inside the model rather than a bad score. The
    current values all happen to satisfy it; this asserts that an edit cannot silently break it.
    """
    heads = DIMS_BY_NAME["n_head"].values
    widths = DIMS_BY_NAME["hidden_size"].values
    bad = [(w, h) for w in widths for h in heads if w % h]
    if bad:
        raise ValueError(f"hidden_size must be divisible by n_head; offending pairs: {bad}")

    missing = sampled_keys() - set(INCUMBENT) - {"huber_delta"}
    if missing:
        raise ValueError(f"INCUMBENT is missing a value for swept dimension(s): {sorted(missing)}")

    for dim in DIMENSIONS:
        if dim.kind == "cat" and len(dim.values) < 2:
            raise ValueError(f"{dim.name}: a categorical dimension with <2 values sweeps nothing")
        if dim.kind == "float" and not dim.high > dim.low:
            raise ValueError(f"{dim.name}: empty float range [{dim.low}, {dim.high}]")
