"""S5 preflight — prove every swept dimension REACHES the model before spending a GPU-hour.

The expensive failure in a hyperparameter sweep is not a crash, it is a lever that does not reach
the model: 32 trials run, a clean confidence interval comes back, and nothing was under test. This
project has hit that failure three times already (``linear_tree`` silently broken under L1,
``get_activation_fn("GELU")`` returning ``F.elu``, ``configs/tft_chronos.yaml`` inert while looking
functional), and the S5 audit found three more in our own code — the config ``loss`` overwritten by
``MAE()``, ``add_volume_sample_weight`` never reaching any member path, and TFT's ``random_seed``
never passed. All three are fixed; these tests are what stop them coming back.

Everything here runs on CPU in about a minute, and it runs TWICE: once locally before launch, and
once inside the Modal driver as its first gate. A search space that silently shrinks — a dimension
quietly dropped because its plumbing broke — is how you get a result nobody can interpret, so the
preflight fails the run rather than trimming the space.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.features import futr_exog_list, nan_col_list, stat_exog_list
from src.models import sweep_space as S
from src.models.registry import _NON_MODEL_KEYS, MODELS, build_loss, build_nf, horizon_weight

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

BASE = {
    "model": "TFT",
    "name": "tft",
    "h": 8,
    "input_size": 24,
    "max_steps": 2,
    "val_check_steps": 2,
    "accelerator": "cpu",
    "devices": 1,
    "batch_size": 8,
    "windows_batch_size": 8,
    "inference_windows_batch_size": 8,
    "enable_progress_bar": False,
    "logger": False,
    "local_scaler_type": "robust",
    "scaler_type": "identity",
}


def _cfg(**over) -> dict:
    """A minimal, genuinely-fittable TFT config with the sweep's overrides applied."""
    return {
        **BASE,
        **S.to_overrides({**S.INCUMBENT, **over}),
        "input_size": over.get("input_size", BASE["input_size"]),
    }


def _frame(n_series: int = 3, n_hours: int = 120) -> pd.DataFrame:
    """nf-shaped frame carrying every futr and static column the registry will attach."""
    idx = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    rows = []
    for s in range(n_series):
        d = pd.DataFrame({"unique_id": f"unit_{s:03d}", "ds": idx})
        # Series-specific level and shape, so per-series MAD differs and a volume weight can bite.
        d["y"] = (s + 1) * (10.0 + np.sin(np.arange(n_hours) / 24.0) * (s + 1))
        for c in futr_exog_list():
            d[c] = np.linspace(0, 1, n_hours) + s
        for c in stat_exog_list():
            d[c] = float(s)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def _static(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("unique_id", as_index=False)[stat_exog_list()].first()


def _fit_predict(cfg: dict, df: pd.DataFrame) -> np.ndarray:
    """Two training steps, then predict. Small, but it exercises the real fit path."""
    from src.data.loader import add_volume_sample_weight

    train = add_volume_sample_weight(df) if cfg.get("sample_weight") == "volume" else df
    nf = build_nf(cfg)
    nf.fit(train, val_size=int(cfg["h"]), static_df=_static(df))
    return nf.predict(futr_df=None if not nf.models[0].futr_exog_list else _futr(df, cfg))[
        "TFT"
    ].to_numpy()


def _futr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The h future rows the model requires for its known-future channel."""
    h = int(cfg["h"])
    out = []
    for uid, g in df.groupby("unique_id"):
        last = g["ds"].max()
        nxt = pd.date_range(last + pd.Timedelta(hours=1), periods=h, freq="h")
        f = pd.DataFrame({"unique_id": uid, "ds": nxt})
        for c in futr_exog_list():
            f[c] = float(g[c].iloc[-1])
        out.append(f)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ the space's own invariants
def test_the_space_is_structurally_valid():
    """n_head divides hidden_size, no dimension sweeps a single value, INCUMBENT is complete."""
    S.validate_space()


def test_the_incumbent_is_a_point_in_the_space():
    """Trial 0 is enqueued from INCUMBENT, so a value outside the space would be unreachable by TPE
    and the study would start somewhere it can never return to."""
    for name, value in S.INCUMBENT.items():
        dim = S.DIMS_BY_NAME[name]
        if dim.values:
            assert value in dim.values, f"{name}={value!r} is not one of {dim.values}"
        else:
            assert dim.low <= value <= dim.high, f"{name}={value!r} outside [{dim.low}, {dim.high}]"


# ------------------------------------------------------------------ THE TOTALITY GUARD
def test_every_sampled_key_is_a_real_constructor_parameter():
    """THE primary guard, and it is static because a static check cannot be defeated by luck.

    ``TFT.__init__`` ends in ``**trainer_kwargs``, so a key that is neither a constructor parameter
    nor a member of ``_NON_MODEL_KEYS`` is swallowed at construction and only raises from inside
    Lightning at ``nf.fit`` — i.e. in all 32 containers at once, after the image pull. Asserting
    that the projection is total rules the whole failure class out before launch.
    """
    accepted = set(inspect.signature(MODELS["TFT"].__init__).parameters) | _NON_MODEL_KEYS
    emitted = set(S.to_overrides({**S.INCUMBENT, "loss": "huber", "huber_delta": 1.0}))
    orphans = emitted - accepted
    assert not orphans, (
        f"{sorted(orphans)} would land in TFT's **trainer_kwargs and raise at fit time. "
        "Add each to registry._NON_MODEL_KEYS (if build_loss/the runner consumes it) or rename "
        "it to the real constructor parameter."
    )


def test_a_typoed_hyperparameter_is_accepted_at_construction(recwarn):
    """TRIPWIRE on the trap itself, so a future upstream fix is noticed rather than assumed.

    Measured on neuralforecast 3.1.9: the model BUILDS with a misspelled kwarg. If this ever starts
    raising at construction, the totality test above becomes belt-and-braces rather than the only
    thing standing between a typo and a dead fan-out — worth knowing either way.
    """
    from neuralforecast.models import TFT

    model = TFT(h=4, input_size=8, max_steps=1, hiden_size=99, accelerator="cpu", logger=False)
    assert model.trainer_kwargs.get("hiden_size") == 99, (
        "upstream now rejects unknown kwargs at construction — good news; update this tripwire "
        "and relax the totality guard's justification."
    )


# ------------------------------------------------------------------ #50, discharged as an assertion
def test_missing_flags_are_routed_to_the_model_as_known_future_exog():
    """Issue #50: *fix routing first, it changes every trial's input space.*

    Verified 2026-08-05 to be ALREADY TRUE, which is why #50 costs nothing and became this test
    instead of a work item: all 10 ``*_missing`` flags sit in ``futr_exog_list()`` and the registry
    hands that list straight to the TFT, so the VSN can down-gate an imputed signal.
    """
    futr = futr_exog_list()
    flags = [f"{c}_missing" for c in nan_col_list()]
    assert set(flags) <= set(futr), sorted(set(flags) - set(futr))

    model = build_nf(_cfg()).models[0]
    assert set(flags) <= set(model.futr_exog_list), "flags never reached the built model"


# ------------------------------------------------------------------ per-dimension reachability
_ATTR_DIMS = [d for d in S.DIMENSIONS if d.reach in ("attr", "hparam")]
_PARAM_DIMS = [d for d in S.DIMENSIONS if d.reach == "params"]


def _model_value(model, dim):
    """The value the built model actually holds — some land on the object, some only in hparams.

    `dropout` and `attn_dropout` are consumed by submodules rather than stored on the TFT, so
    `model.hparams` is where they are visible. Reading either is still strictly stronger than
    "the constructor accepted it": a value in `trainer_kwargs` appears in NEITHER.
    """
    name = dim.attr or dim.name
    if dim.reach == "attr":
        return getattr(model, name)
    return dict(model.hparams)[name]


@pytest.mark.parametrize("dim", _ATTR_DIMS, ids=lambda d: d.name)
def test_dimension_reaches_the_model_as_a_stored_value(dim):
    """The requested value is on the built model — not merely accepted and discarded."""
    lo, hi = dim.probe()
    got_lo = _model_value(build_nf(_cfg(**{dim.name: lo})).models[0], dim)
    got_hi = _model_value(build_nf(_cfg(**{dim.name: hi})).models[0], dim)
    assert got_lo == pytest.approx(lo) if isinstance(lo, float) else got_lo == lo
    assert got_lo != got_hi, f"{dim.name}: {lo!r} and {hi!r} both produced {got_lo!r}"


@pytest.mark.parametrize("dim", _PARAM_DIMS, ids=lambda d: d.name)
def test_dimension_reaches_the_model_as_a_parameter_count(dim):
    """Capacity-changing dimensions must change the model's size.

    ``scaler_type`` is in here for a reason worth recording: ``revin`` maps to the SAME statistics
    and scaler callables as ``standard`` in neuralforecast 3.1.9 and differs ONLY by registering
    learnable ``revin_weight`` / ``revin_bias``. A parameter count is therefore the honest proof
    that it is a distinct arm, and "RevIN vs robust scaling" is really "standard scaling plus a
    learnable affine" — which is what the write-up should say (#42).
    """
    lo, hi = dim.probe()
    n_lo = sum(p.numel() for p in build_nf(_cfg(**{dim.name: lo})).models[0].parameters())
    n_hi = sum(p.numel() for p in build_nf(_cfg(**{dim.name: hi})).models[0].parameters())
    assert n_lo != n_hi, f"{dim.name}: {lo!r} and {hi!r} both built {n_lo} parameters"


def test_every_corner_of_the_space_builds():
    """Each dimension at each extreme, the rest at the incumbent. Cheap, and exhaustive."""
    for dim in S.DIMENSIONS:
        for value in dim.values or (dim.low, dim.high):
            over = {dim.name: value}
            if dim.name == "huber_delta":
                over["loss"] = "huber"
            build_nf(_cfg(**over))  # must not raise


def test_the_largest_corner_actually_trains():
    """One real fit at the widest, deepest, longest-lookback corner.

    Building is not fitting: shape and memory errors surface in the training step, and this corner
    is the one that would hit them. It is also the cheapest possible answer to "will hidden_size
    192 fit on an L4 at h=672" — the corner runs here on CPU first, so the GPU smoke is confirming
    rather than discovering.
    """
    df = _frame(n_hours=200)
    cfg = _cfg(hidden_size=192, n_head=8, scaler_type="revin", grn_activation="SELU", loss="huber")
    cfg["input_size"] = 96
    preds = _fit_predict(cfg, df)
    assert preds.shape[0] == 3 * cfg["h"] and np.isfinite(preds).all()


# ------------------------------------------------------------------ the loss dimensions
def test_the_default_loss_is_still_a_bare_MAE():
    """THE DRIFT GUARD on the plumbing change. ``build_loss`` now honours a config ``loss`` where
    the old code overwrote it — so the no-``loss`` path must resolve to exactly what shipped, or
    every recorded number in the project moves under us."""
    loss = build_loss({})
    assert type(loss).__name__ == "MAE"
    assert loss.horizon_weight is None


def test_huber_delta_reaches_the_loss_and_changes_its_value():
    y = torch.zeros(2, 8, 1)
    y_hat = torch.full((2, 8, 1), 3.0)
    mae = build_loss({"loss": "mae"})(y, y_hat)
    soft = build_loss({"loss": "huber", "huber_delta": 0.5})(y, y_hat)
    hard = build_loss({"loss": "huber", "huber_delta": 2.0})(y, y_hat)
    assert not torch.isclose(mae, soft) and not torch.isclose(soft, hard)


def test_the_gap_mask_weights_exactly_the_graded_tail():
    """#51. ``h=672`` is 336h of gap plus the 336h that is actually scored, and by default two
    thirds of the loss signal is spent on a block nobody grades."""
    w = horizon_weight({"loss_mask_gap": True, "score_len": S.SCORE_LEN}, 672)
    assert len(w) == 672
    assert (w[:336] == 0).all() and (w[336:] == 1).all()
    # Off, and degenerate horizons, both fall back to the unweighted default rather than asserting.
    assert horizon_weight({}, 672) is None
    assert horizon_weight({"loss_mask_gap": True}, 336) is None


def test_the_gap_mask_changes_the_loss_only_via_the_masked_half():
    """Error in the gap half must become invisible; error in the graded half must not."""
    h, scored = 16, 8
    cfg = {"loss": "mae", "loss_mask_gap": True, "score_len": scored}
    masked, plain = build_loss(cfg, h), build_loss({"loss": "mae"}, h)

    y = torch.zeros(1, h, 1)
    gap_only = torch.zeros(1, h, 1)
    gap_only[:, : h - scored] = 5.0  # error lives entirely in the ignored half
    assert torch.isclose(masked(y, gap_only), torch.tensor(0.0))
    assert plain(y, gap_only) > 0

    tail_only = torch.zeros(1, h, 1)
    tail_only[:, h - scored :] = 5.0
    assert masked(y, tail_only) > 0
    assert not torch.isclose(masked(y, tail_only), plain(y, tail_only))


# ------------------------------------------------------------------ the data-side dimension
def test_volume_sample_weight_changes_the_fitted_model():
    """Plan 3.3 / #38 — the aligned loss, which the backbone had never once trained under.

    ``add_volume_sample_weight`` existed and ``sample_weight: volume`` existed, and NOTHING
    connected them on any member path: the helper was only ever called from ``src/train.py``, which
    no member goes through. This asserts the connection rather than trusting it, and it is a fit
    rather than an attribute check because the change is in the DATA, not on the model.
    """
    df = _frame()
    off = _fit_predict(_cfg(sample_weight="none"), df)
    on = _fit_predict(_cfg(sample_weight="volume"), df)
    assert off.shape == on.shape
    assert not np.allclose(off, on), (
        "the volume weight did not move the fit — neuralforecast consumes a `sample_weight` column "
        "natively, so this failing means the column is not reaching nf.fit"
    )


def test_a_bare_seed_does_NOT_reach_a_neuralforecast_model():
    """THE BUG THIS SUITE WAS WRITTEN TO CATCH, pinned so it cannot come back quietly.

    ``registry.set_seed(cfg["seed"])`` seeds python/numpy/torch — and then
    ``BaseModel.__init__`` calls ``pl.seed_everything(self.random_seed, workers=True)`` and undoes
    it. ``random_seed`` defaults to 1 and ``seed`` is a non-model key, so **every neural run in this
    project has trained at an effective seed of 1** and ``ctx.seed`` moved nothing.

    Nothing recorded is wrong because of it — both arms of every paired A/B sat at the same
    effective seed — but ``seedable=True`` was a false claim for the nf members, and Tier D's
    seed-bagging would have averaged five copies of one model and reported std = 0 as a
    measurement. The historical regime is kept deliberately (see ``seeded_cfg``) so recorded
    numbers still reproduce; this test documents it rather than papering over it.
    """
    from src.models.members import HISTORICAL_NF_SEED

    for seed in (42, 7739):
        model = build_nf({**_cfg(), "seed": seed}).models[0]
        assert model.random_seed == HISTORICAL_NF_SEED


def test_an_explicit_seed_changes_the_fitted_weights():
    """Guards Tier D: seed-bagging only decorrelates errors if the seeds land in different basins.

    ``seeded_cfg`` is what makes a requested seed real, and it is applied only when a caller
    explicitly supplies one — which is exactly the seed-bagging and ``frozen_seeds()`` paths.
    """
    from src.models.members import seeded_cfg

    df = _frame()
    a = _fit_predict(seeded_cfg(_cfg(), 42), df)
    b = _fit_predict(seeded_cfg(_cfg(), 7739), df)
    assert not np.allclose(a, b), "two explicitly-seeded fits produced identical predictions"
