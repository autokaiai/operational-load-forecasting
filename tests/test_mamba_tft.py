"""MambaTFT (IDEAS #16) — the encoder swap must be exactly what it claims to be.

Six claims, each of which has to be able to FAIL (project law 6):

1. The chunked associative scan equals the naive ``h_t = a_t h_{t-1} + b_t`` loop, and is invariant
   to the chunk length and to gradient checkpointing. If it is not, the lane measures a bug.
2. Splitting a sequence and carrying the state is **identical** to running it whole — the property
   the history -> future handoff depends on, and the one a forgotten conv tail silently breaks.
3. ``ssm_mode="lstm"`` is numerically identical to neuralforecast's own TFT on shared weights.
   That is what licenses this lane to use the CACHED ``tft_cascade`` cube as its control.
4. The swap is parameter-matched: the SSM encoder pair is within 10% of the LSTM pair it replaces,
   so a win cannot be a capacity win.
5. **Reachability**: the SSM is load-bearing — perturbing the far end of the history moves the
   forecast, and the state handed to the future encoder changes it. This is the test that
   separates "a state-space encoder" from "a state-space module wired to nothing", which is how
   this project has produced three confident nulls before (law 5).
6. The config in ``configs/tft_chronos_mamba.yaml`` builds and takes a training step through the
   ordinary ``build_nf`` path, with no key silently swallowed by ``**trainer_kwargs``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("neuralforecast")

from neuralforecast.models import TFT  # noqa: E402

from src.models.mamba_tft import MambaTFT, SelectiveSSM, _scan_chunk  # noqa: E402

H, L, N_FUTR, N_STAT, HID, B = 8, 16, 3, 2, 8, 2

COMMON = {
    "h": H,
    "input_size": L,
    "hidden_size": HID,
    "n_head": 2,
    "dropout": 0.1,
    "futr_exog_list": [f"f{i}" for i in range(N_FUTR)],
    "stat_exog_list": [f"s{i}" for i in range(N_STAT)],
    "max_steps": 1,
    "random_seed": 0,
}


def _batch(seed: int = 0, batch: int = B):
    g = torch.Generator().manual_seed(seed)
    return {
        "insample_y": torch.randn(batch, L, 1, generator=g),
        "futr_exog": torch.randn(batch, L + H, N_FUTR, generator=g),
        "hist_exog": None,
        "stat_exog": torch.randn(batch, N_STAT, generator=g),
    }


def _build(mode: str, **kw) -> MambaTFT:
    torch.manual_seed(0)
    return MambaTFT(ssm_mode=mode, **{**COMMON, **kw}).eval()


# ---------------------------------------------------------------- 1. the scan is the recurrence
def _naive(a, b, h0):
    h = h0
    out = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def test_scan_chunk_matches_the_naive_recurrence():
    g = torch.Generator().manual_seed(1)
    a = torch.rand(2, 13, 4, 3, generator=g)  # non-power-of-2 length on purpose
    b = torch.randn(2, 13, 4, 3, generator=g)
    c = torch.randn(2, 13, 3, generator=g)
    h0 = torch.randn(2, 4, 3, generator=g)

    y, h_last = _scan_chunk(a, b, c, h0)
    h_ref = _naive(a, b, h0)
    assert torch.allclose(y, torch.einsum("bldn,bln->bld", h_ref, c), atol=1e-5)
    assert torch.allclose(h_last, h_ref[:, -1], atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 5, 64, 4096])
def test_block_output_is_invariant_to_chunk_length(chunk):
    torch.manual_seed(0)
    blk = SelectiveSSM(HID, chunk=64).eval()
    x = torch.randn(2, 37, HID)
    ref, _ = blk(x)
    blk.chunk = chunk
    got, _ = blk(x)
    assert torch.allclose(ref, got, atol=1e-5), f"chunk={chunk} changed the arithmetic"


def test_checkpointing_does_not_change_the_gradient():
    torch.manual_seed(0)
    x = torch.randn(2, 37, HID)
    grads = []
    for flag in (False, True):
        torch.manual_seed(0)
        blk = SelectiveSSM(HID, chunk=8, use_checkpoint=flag).train()
        xi = x.clone().requires_grad_(True)
        blk(xi)[0].square().sum().backward()
        grads.append(xi.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


# --------------------------------------------------- 2. the handoff is a genuine continuation
def test_split_with_carried_state_equals_the_whole_sequence():
    """The property the history -> future handoff rests on. A dropped conv tail breaks THIS test."""
    torch.manual_seed(0)
    blk = SelectiveSSM(HID, chunk=8).eval()
    x = torch.randn(2, 24, HID)
    whole, _ = blk(x)
    first, state = blk(x[:, :10])
    second, _ = blk(x[:, 10:], state)
    assert torch.allclose(whole, torch.cat([first, second], dim=1), atol=1e-5)


# ------------------------------------------------------- 3. the control arm is the plain TFT
def test_lstm_mode_is_bit_identical_to_neuralforecast_tft():
    torch.manual_seed(0)
    ref = TFT(**COMMON).eval()
    arm = _build("lstm")
    arm.load_state_dict(ref.state_dict())
    with torch.no_grad():
        assert torch.equal(ref(_batch()), arm(_batch()))


def test_ssm_mode_actually_replaces_the_encoder():
    arm = _build("ssm")
    assert not isinstance(arm.temporal_encoder.history_encoder, torch.nn.LSTM)
    assert not any("history_encoder.weight_hh" in k for k in arm.state_dict())


@pytest.mark.parametrize("mode", ["bogus", "gru", ""])
def test_unknown_mode_is_rejected_at_construction(mode):
    with pytest.raises(ValueError):
        _build(mode)


# -------------------------------------------------------------------- 4. parameter matching
def test_encoder_pair_is_parameter_matched_to_the_lstm_pair():
    """At the SHIPPED geometry, which is the only one the claim is about.

    ``d_state`` is an absolute width, not a fraction of ``hidden_size``, so the ratio is only
    meaningful at the config's own hidden 64 — the toy models the rest of this file builds at
    hidden 8 are 2.3x, and asserting there would pin a number nothing ships.
    """
    prod = {**COMMON, "hidden_size": 64}
    torch.manual_seed(0)
    lstm = MambaTFT(ssm_mode="lstm", **prod).temporal_encoder
    torch.manual_seed(0)
    ssm = MambaTFT(ssm_mode="ssm", ssm_state_size=16, ssm_expand=2, **prod).temporal_encoder
    n_lstm = sum(p.numel() for p in lstm.history_encoder.parameters()) + sum(
        p.numel() for p in lstm.future_encoder.parameters()
    )
    # The zero-initialised static-context map is excluded: it is the analogue of the LSTM's own
    # (ch, cc) initial state, which nn.LSTM gets for free because the TFT hands it in directly.
    n_ssm = sum(
        p.numel()
        for enc in (ssm.history_encoder, ssm.future_encoder)
        for p in enc.blocks.parameters()
    )
    assert (n_lstm, n_ssm) == (66560, 65536), f"geometry moved: SSM {n_ssm} vs LSTM {n_lstm}"
    assert 0.9 <= n_ssm / n_lstm <= 1.1, f"SSM {n_ssm} vs LSTM {n_lstm} — not a matched swap"


def test_vsn_weights_are_adopted_not_reinitialised():
    """The arm and its control must share every weight outside the encoder at a given seed."""
    ctl, arm = _build("lstm"), _build("ssm")
    for (k, a), (_, b) in zip(
        ctl.temporal_encoder.history_vsn.state_dict().items(),
        arm.temporal_encoder.history_vsn.state_dict().items(),
        strict=True,
    ):
        assert torch.equal(a, b), f"history_vsn.{k} diverged — the arms are not paired"


# ------------------------------------------------------------------------- 5. reachability
def test_history_is_reachable_through_the_ssm():
    arm = _build("ssm")
    b1 = _batch()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b1.items()}
    b2["insample_y"][:, 0] += 5.0  # the FAR end of the history: only the recurrence carries it
    with torch.no_grad():
        assert not torch.allclose(arm(b1), arm(b2), atol=1e-6)


def test_the_state_handed_to_the_future_encoder_is_used():
    torch.manual_seed(0)
    blk = SelectiveSSM(HID, chunk=8).eval()
    x = torch.randn(2, 12, HID)
    zero, _ = blk(x)
    h, tail = blk.initial_state(2, x.device, x.dtype)
    carried, _ = blk(x, (h + 1.0, tail + 1.0))
    assert not torch.allclose(zero, carried, atol=1e-6)


# ------------------------------------------------------------- 6. the config path is not a lie
def test_shipped_config_fits_and_predicts_through_build_nf():
    """The real config through the real path, tiny: keys reach the constructor, a fit runs.

    ``build_nf`` attaches ``futr_exog_list()`` / ``stat_exog_list()`` from the model's capability
    flags, so the frame has to carry the full conditioning set — the same shape the member path
    hands it. This is what catches a config key that ``**trainer_kwargs`` swallowed at construction
    and would only have raised inside Lightning at ``nf.fit``, four GPU-minutes into a container.
    """
    import numpy as np
    import pandas as pd
    import yaml

    from src.data.features import futr_exog_list, stat_exog_list
    from src.data.loader import NF_ID, NF_TARGET, NF_TIME
    from src.models.registry import build_nf

    cfg = {
        **yaml.safe_load(open("configs/base.yaml")),
        **yaml.safe_load(open("configs/tft_chronos_mamba.yaml")),
        "h": H,
        "input_size": L,
        "max_steps": 2,
        "val_check_steps": 2,
        "val_size": H,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "windows_batch_size": 4,
        "inference_windows_batch_size": 4,
        "ssm_state_size": 4,
        "ssm_chunk": 8,
    }
    per, series, rng = 3 * (L + H), ["unit_000", "unit_001"], np.random.default_rng(0)
    frames = []
    for sid in series:
        ds = pd.date_range("2023-01-01", periods=per, freq="h")
        cols = {NF_ID: sid, NF_TIME: ds, NF_TARGET: 10 + rng.normal(0, 1, per)}
        cols.update({c: rng.normal(0, 1, per) for c in futr_exog_list()})
        frames.append(pd.DataFrame(cols))
    long_df = pd.concat(frames, ignore_index=True)
    static_df = pd.DataFrame({NF_ID: series, **{c: [1.0, 2.0] for c in stat_exog_list()}})

    last = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=per - 1)
    fut = []
    for sid in series:
        ds = pd.date_range(last + pd.Timedelta(hours=1), periods=H, freq="h")
        cols = {NF_ID: sid, NF_TIME: ds}
        cols.update({c: rng.normal(0, 1, H) for c in futr_exog_list()})
        fut.append(pd.DataFrame(cols))

    nf = build_nf(cfg)
    nf.fit(long_df, static_df=static_df, val_size=H)
    preds = nf.predict(futr_df=pd.concat(fut, ignore_index=True))
    assert len(preds) == len(series) * H
    assert preds["MambaTFT"].notna().all()
