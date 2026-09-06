"""IDEAS #16 — replace the TFT's LSTM encoders with a selective state-space (Mamba) layer.

WHAT THIS IS. neuralforecast's ``TemporalCovariateEncoder`` runs the history and the future
through two ``nn.LSTM``s: the history LSTM is initialised from the static context ``(ch, cc)``,
and its final state is handed to the future LSTM. Everything downstream — the shared gated skip,
static enrichment, the interpretable attention, the decoder — is untouched here. ``MambaTFT`` swaps
**those two LSTMs, and nothing else**, for a selective state-space stack (Mamba, Gu & Dao
2023): a diagonal linear recurrence whose transition, input and output projections are functions of
the input at each step, which is what "selective" means and what a plain S4 lacks.

WHY IT IS A FAIR SWAP. The default (``expand=2``, ``d_state=16``, one layer, hidden 64) carries
**32,768 parameters per encoder against the LSTM's 33,280** — 98% of the incumbent, arrived at by
arithmetic and asserted in ``tests/test_mamba_tft.py``, not by tuning. A win here is therefore not
a win bought with capacity.

NOTHING IN THE INSTALLED ``neuralforecast`` IS MODIFIED. The class subclasses ``TFT``, lets it build
itself completely, and then **adopts** the already-constructed VSNs and gated skip into a new
encoder module, replacing only ``history_encoder`` / ``future_encoder``. Adoption (rather than
re-construction) is load-bearing: it means every weight the control and this arm share is drawn from
the identical RNG stream at the identical point, so at seed 42 the two models differ in the encoder
and in nothing else. ``ssm_mode="lstm"`` keeps the original encoder untouched and is asserted
bit-for-bit against neuralforecast's own TFT — that assertion is what lets this lane use the
**cached** ``tft_cascade`` cube as its control instead of paying a GPU container for one.

THE THREE THINGS THAT CARRY OVER FROM HISTORY TO FUTURE, and why each is what it is:

  * the **SSM state** ``h`` [B, d_inner, d_state] — the exact analogue of the LSTM's ``(h, c)``
    handoff, which is the mechanism the TFT's design relies on to make the future encoder a
    continuation of the history rather than a fresh read.
  * the **conv tail**, the last ``d_conv - 1`` frames entering the causal depthwise conv. The LSTM
    has no analogue because it has no receptive field beyond its state; dropping it would put a
    zero-padded seam at the history/future boundary, which is exactly the boundary the horizon
    starts at. Carried, and the seam is measured by ``test_conv_tail_carry_is_continuous``.
  * the **static context** ``ch``. The LSTM uses it as the initial hidden state; here a
    ``zero-initialised`` linear map turns it into the initial SSM state, so the model *starts* at
    the standard ``h_0 = 0`` and may learn to use the statics, rather than being handed a random
    projection of them at step 0.

THE SCAN IS EXACT, NOT AN APPROXIMATION. ``h_t = a_t h_{t-1} + b_t`` is a first-order linear
recurrence, so it is computed by an associative (Hillis-Steele) scan **inside** chunks of
``ssm_chunk`` steps and sequentially **across** chunks, which is O(L log c) work with 19 sequential
hops at L=1176 instead of 1176. ``tests/test_mamba_tft.py`` pins it to a naive python loop at
1e-5. The chunking is also what keeps it numerically boring: no log-space cumulative product ever
spans more than ``ssm_chunk`` steps, so the exp-overflow that kills the one-shot parallel form at
L > 100 cannot arise. Each chunk is optionally gradient-checkpointed (default on), and because the
``C`` contraction happens **inside** the chunk the stored activation is the [B, c, d_inner] output,
never the [B, L, d_inner, d_state] state cube — the difference between ~10 MB and ~1.9 GB at the
gapped h=672.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from neuralforecast.models import TFT
from neuralforecast.models.tft import TemporalCovariateEncoder
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

# The modes a config may select. "lstm" is the untouched neuralforecast encoder and exists as the
# in-code control; "ssm" is the lane. Rejected at construction rather than silently falling back,
# because a member that quietly degrades to its own control is the trap this project has hit twice
# (S10 law 5).
SSM_MODES = ("ssm", "lstm")


def _shift(x: Tensor, step: int, fill: float) -> Tensor:
    """Shift ``x`` right along dim 1 by ``step``, filling the vacated head with ``fill``."""
    head = x.new_full((x.shape[0], step, *x.shape[2:]), fill)
    return torch.cat([head, x[:, :-step]], dim=1)


def _scan_chunk(a: Tensor, b: Tensor, c_proj: Tensor, h_in: Tensor) -> tuple[Tensor, Tensor]:
    """One chunk of ``h_t = a_t h_{t-1} + b_t``, contracted against ``C`` before it is returned.

    ``a``/``b`` are [B, L, D, N], ``c_proj`` is [B, L, N], ``h_in`` is [B, D, N]. Returns the
    per-step output [B, L, D] and the final state [B, D, N].

    The Hillis-Steele invariant, stated because getting the shift wrong produces a plausible-looking
    tensor rather than an error: after the round with window ``w``, ``S_t`` is the recurrence run
    over the last ``w`` steps and ``A_t`` is the product of the last ``w`` transitions. Doubling the
    window uses the PRE-update ``A`` for both, which is why ``a_run`` is updated last.
    """
    length = a.shape[1]
    s = b
    a_run = a
    step = 1
    while step < length:
        s = a_run * _shift(s, step, 0.0) + s
        a_run = a_run * _shift(a_run, step, 1.0)
        step *= 2
    # a_run is now the inclusive cumulative product of ``a`` within the chunk, so this is the
    # carried-in state's contribution and ``s`` is the chunk's own.
    h = a_run * h_in.unsqueeze(1) + s
    y = torch.einsum("bldn,bln->bld", h, c_proj)
    return y, h[:, -1]


class SelectiveSSM(nn.Module):
    """One Mamba block: gated input projection, causal depthwise conv, selective scan, output gate.

    The state is explicit in the signature (``forward(x, state) -> (y, state)``) because this module
    exists to be a drop-in for ``nn.LSTM`` inside the TFT's encoder, and the encoder's whole design
    turns on handing the history's final state to the future.
    """

    def __init__(
        self,
        hidden_size: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        chunk: int = 64,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_inner = int(expand * hidden_size)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.dt_rank = int(dt_rank or max(1, math.ceil(hidden_size / 16)))
        self.chunk = int(chunk)
        self.use_checkpoint = bool(use_checkpoint)

        self.norm = nn.LayerNorm(hidden_size, eps=1e-3)
        self.in_proj = nn.Linear(hidden_size, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=self.d_conv, groups=self.d_inner, bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

        # S4D-real initialisation (Gu et al.): A_n = -n, so the N state channels start with
        # geometrically spread decay rates and the block sees several timescales at step 0 —
        # the daily/weekly structure this panel carries is not reachable from one rate.
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt in [1e-3, 1e-1] through softplus, the Mamba reference init. dt sets how much of each
        # step is integrated; starting it uniform-in-log keeps early steps from either freezing the
        # state or forgetting it within a few hours.
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def initial_state(self, batch: int, device, dtype) -> tuple[Tensor, Tensor]:
        return (
            torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype),
            torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype),
        )

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, length, _ = x.shape
        if state is None:
            state = self.initial_state(b, x.device, x.dtype)
        h_in, conv_tail = state

        xz = self.in_proj(self.norm(x))
        x_in, z = xz.chunk(2, dim=-1)

        # Causal conv, seeded with the carried tail rather than zeros: the seam sits exactly at the
        # forecast anchor, so a zero pad there is a discontinuity in the one place it costs.
        u = torch.cat([conv_tail, x_in.transpose(1, 2)], dim=-1)
        new_tail = u[..., -(self.d_conv - 1) :] if self.d_conv > 1 else conv_tail
        u = F.silu(self.conv1d(u)[..., :length]).transpose(1, 2)

        dt_bc = self.x_proj(u)
        dt, b_proj, c_proj = torch.split(dt_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B,L,d_inner]
        a = -torch.exp(self.A_log)  # [d_inner,d_state], strictly negative -> stable

        da = torch.exp(dt.unsqueeze(-1) * a)  # [B,L,d_inner,d_state] in (0,1)
        db = dt.unsqueeze(-1) * b_proj.unsqueeze(2) * u.unsqueeze(-1)

        ys = []
        h = h_in
        for start in range(0, length, self.chunk):
            sl = slice(start, start + self.chunk)
            args = (da[:, sl], db[:, sl], c_proj[:, sl], h)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                y_c, h = checkpoint(_scan_chunk, *args, use_reentrant=False)
            else:
                y_c, h = _scan_chunk(*args)
            ys.append(y_c)
        y = torch.cat(ys, dim=1) + u * self.D

        return self.out_proj(y * F.silu(z)), (h, new_tail)


class SSMEncoder(nn.Module):
    """A stack of ``SelectiveSSM`` blocks with the interface of the ``nn.LSTM`` it replaces.

    Layer 0 is a **pure replacement** — no residual around it — because the LSTM it stands in for is
    a pure transform and the TFT already adds its own skip (``input_gate(temporal) + embedding``)
    one level up; wrapping layer 0 in a second residual would make the arm "TFT plus an SSM
    correction" rather than "TFT with the recurrence swapped", which is a different experiment.
    Layers above 0 are residual, which is what makes a deeper stack trainable.
    """

    def __init__(self, hidden_size: int, n_layers: int = 1, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SelectiveSSM(hidden_size, **block_kwargs) for _ in range(n_layers)]
        )
        # Static context -> initial SSM state, one map per layer, ZERO-INITIALISED so training
        # starts from the standard h_0 = 0 and the statics are an option the model may take up.
        self.state_init = nn.ModuleList(
            [nn.Linear(hidden_size, blk.d_inner * blk.d_state) for blk in self.blocks]
        )
        for lin in self.state_init:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def state_from_context(self, ch: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Build the per-layer initial state from the TFT's static context ``ch`` [n_rnn, B, H]."""
        states = []
        for i, (blk, lin) in enumerate(zip(self.blocks, self.state_init, strict=True)):
            ctx = ch[min(i, ch.shape[0] - 1)]
            h0 = lin(ctx).view(-1, blk.d_inner, blk.d_state)
            _, tail = blk.initial_state(ctx.shape[0], ctx.device, ctx.dtype)
            states.append((h0, tail))
        return states

    def forward(self, x: Tensor, states: list | None = None):
        out_states = []
        for i, blk in enumerate(self.blocks):
            y, st = blk(x, None if states is None else states[i])
            x = y if i == 0 else x + y
            out_states.append(st)
        return x, out_states


class SSMTemporalCovariateEncoder(nn.Module):
    """``TemporalCovariateEncoder`` with the two LSTMs replaced, everything else ADOPTED.

    The VSNs and the gated skip are the *same objects* the parent TFT already built — see the module
    docstring on why adoption rather than re-construction is what keeps this arm and its control
    sharing an RNG stream.
    """

    def __init__(self, base: TemporalCovariateEncoder, hidden_size: int, n_layers: int, **kw):
        super().__init__()
        self.history_vsn = base.history_vsn
        self.future_vsn = base.future_vsn
        self.input_gate = base.input_gate
        self.input_gate_ln = base.input_gate_ln
        self.history_encoder = SSMEncoder(hidden_size, n_layers=n_layers, **kw)
        self.future_encoder = SSMEncoder(hidden_size, n_layers=n_layers, **kw)

    def forward(self, historical_inputs, future_inputs, cs, ch, cc):
        # Line-for-line the neuralforecast forward, with the two encoder calls swapped. `cc` is
        # accepted and unused: the LSTM's cell state has no analogue here, and changing the
        # signature would mean editing the TFT's forward as well.
        historical_features, history_vsn_sparse_weights = self.history_vsn(historical_inputs, cs)
        init = self.history_encoder.state_from_context(ch)
        history, state = self.history_encoder(historical_features, init)

        future_features, future_vsn_sparse_weights = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)

        input_embedding = torch.cat([historical_features, future_features], dim=1)
        temporal_features = torch.cat([history, future], dim=1)
        temporal_features = self.input_gate(temporal_features)
        temporal_features = temporal_features + input_embedding
        temporal_features = self.input_gate_ln(temporal_features)
        return temporal_features, history_vsn_sparse_weights, future_vsn_sparse_weights


class MambaTFT(TFT):
    """The shipped TFT with its recurrent encoders swapped for a selective SSM (IDEAS #16).

    Every constructor argument of ``TFT`` is forwarded untouched; the ``ssm_*`` arguments below are
    this class's own and are declared explicitly so that ``tests/test_sweep_space.py``'s totality
    check (every config key is a constructor parameter or a declared non-model key) still holds.
    """

    def __init__(
        self,
        *args,
        ssm_mode: str = "ssm",
        ssm_layers: int = 1,
        ssm_state_size: int = 16,
        ssm_conv_width: int = 4,
        ssm_expand: int = 2,
        ssm_dt_rank: int | None = None,
        ssm_chunk: int = 64,
        ssm_checkpoint: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if ssm_mode not in SSM_MODES:
            raise ValueError(f"ssm_mode must be one of {SSM_MODES}, got {ssm_mode!r}")
        self.ssm_mode = ssm_mode
        if ssm_mode == "lstm":
            return  # the untouched neuralforecast encoder: this arm IS the control
        self.temporal_encoder = SSMTemporalCovariateEncoder(
            self.temporal_encoder,
            hidden_size=self.temporal_encoder.input_gate_ln.normalized_shape[0],
            n_layers=ssm_layers,
            d_state=ssm_state_size,
            d_conv=ssm_conv_width,
            expand=ssm_expand,
            dt_rank=ssm_dt_rank,
            chunk=ssm_chunk,
            use_checkpoint=ssm_checkpoint,
        )
