"""Checkpoint averaging (SWA) for a STEP-based neuralforecast run.

WHY THIS EXISTS RATHER THAN ``pytorch_lightning.callbacks.StochasticWeightAveraging``. Lightning's
SWA is epoch-based and reads ``trainer.max_epochs`` in three places: ``on_fit_start`` asserts it is
not ``None``, a float ``swa_epoch_start`` is resolved as ``int(max_epochs * frac)``, and the
averaging window is ``swa_start <= current_epoch <= max_epochs - 1``. neuralforecast trains on
``max_steps`` with ``check_val_every_n_epoch=None`` and **raises** on a ``max_epochs`` trainer kwarg
(``_base_model.py``: *"max_epochs is deprecated, use max_steps instead"*). Measured on this venv:

    pl.Trainer(max_steps=5000).max_epochs  ->  None

so Lightning's SWA cannot initialise here, and the one kwarg that would fix it is refused by the
library above it. That is plan law 5 territory — *a lever that does not reach the model returns a
confident null* — which is why this module exists and why ``tests/test_swa.py`` asserts the averaged
weights actually DIFFER from the un-averaged ones rather than asserting the callback merely ran.

WHAT IT DOES. Every ``every_n_steps`` optimizer steps it snapshots the module's float parameters to
CPU, keeping a **rolling** window of the last ``n_snapshots``. At ``on_train_end`` it writes their
mean back into the module. The forecast is then produced by the averaged weights, because
neuralforecast predicts from the same in-memory module it just fitted.

THE WINDOW IS A FRACTION OF THE ACTUAL RUN, and both halves of that matter. This project's TFT
early-stops long before ``max_steps`` — the cross-variate arms at <=749 of 2000, the Mamba arm at
~849 of 5000 — so a schedule keyed to ``max_steps`` would collect nothing at all. But keying it to
a fixed STEP COUNT is wrong too, and was measured wrong: see ``StepCheckpointAverage`` below, where
a fixed 250-step window averaged 52% of the shortest run and 24% of the longest, and the delta
tracked that fraction on 3/3 windows. The window is therefore ``tail_fraction`` of however many
snapshots were actually taken.

WHAT IT DELIBERATELY DOES NOT DO:

  * **No BatchNorm re-estimation pass.** The TFT normalises with ``LayerNorm``, which carries no
    running statistics, so the ``update_bn`` pass Lightning performs is unnecessary here. Asserted
    in the tests against the real model rather than assumed from the architecture diagram.
  * **No LR schedule of its own.** Lightning's SWA installs an ``SWALR`` and *replaces* any existing
    scheduler; this one leaves the optimizer alone, so the averaging is measured against the
    incumbent's own training trajectory and nothing else moves. See ``num_lr_decays`` below.
  * **No buffer averaging.** Only floating-point parameters are averaged; integer buffers (step
    counters and the like) are left at the final model's values.

INTERACTION WITH ``num_lr_decays`` (neuralforecast's own step-wise LR decay, already a native config
key). The two are independent levers and this callback does not touch the schedule, so they compose
— but they are opposed in intent: averaging wants a *late trajectory that still moves* to have
something to average over, and decay flattens exactly that. Run them as separate arms before running
them together, or a combined null cannot be attributed.
"""

from __future__ import annotations

import math
from collections import deque

import pytorch_lightning as pl
import torch


class StepCheckpointAverage(pl.Callback):
    """Average the last ``tail_fraction`` of the trajectory, sampled every ``every_n_steps`` steps.

    THE WINDOW IS A FRACTION, NOT A STEP COUNT, AND THAT IS THE WHOLE DESIGN. The first version of
    this callback averaged a FIXED last-k snapshots, and it was measured wrong in a way worth
    keeping on the record: run 2026-09-03, 3 windows, seed 892, against `casc_s892`.

        win   wall_s    delta            fraction of the run averaged
        W0    148.2     -0.00997         ~52%
        W2    254.6     +0.00190         ~30%
        W1    324.7     +0.01710         ~24%

    The runs early-stop at different points — 2.19x between longest and shortest — so a fixed
    250-step window averaged half of the short run and a quarter of the long one. Rank of wall time
    equalled rank of delta on 3/3 windows: the shorter the run, the more of it was averaged, the
    worse it did. That is a confound with the CONFIGURATION, not a property of checkpoint averaging,
    and averaging half a trajectory means folding in weights that are barely trained.

    Expressing the window as a fraction of whatever actually happened removes the confound. The
    default 0.25 is the convention from the original SWA paper (averaging begins at 75% of
    training), deliberately taken from an external anchor rather than from our own W1 — which
    happened to average ~24% and gain the most, and would have been the number to pick if we were
    fitting the parameter to the result we already saw.

    Parameters
    ----------
    every_n_steps:
        Optimizer steps between snapshots. Must be > 0.
    tail_fraction:
        Fraction of the taken snapshots to average, counted from the end. In (0, 1].
    min_snapshots:
        Floor on the averaging window, so a short run still averages something. Must be >= 2 — a
        window of 1 is a silent no-op, the exact failure mode this module exists to avoid.
    max_snapshots:
        Memory cap on the rolling buffer. At hidden 64 the TFT is ~1.65M floats ≈ 6.6 MB per
        snapshot, so 32 is ~210 MB of CPU RAM. A 5000-step run takes 100 snapshots and averages 25
        of them, which still fits inside the buffer.
    """

    def __init__(
        self,
        every_n_steps: int = 50,
        tail_fraction: float = 0.25,
        min_snapshots: int = 2,
        max_snapshots: int = 32,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError(f"every_n_steps must be > 0, got {every_n_steps}")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
        if min_snapshots < 2:
            raise ValueError(
                f"min_snapshots must be >= 2 (a window of 1 averages nothing), got {min_snapshots}"
            )
        if max_snapshots < min_snapshots:
            raise ValueError(f"max_snapshots ({max_snapshots}) < min_snapshots ({min_snapshots})")
        self.every_n_steps = int(every_n_steps)
        self.tail_fraction = float(tail_fraction)
        self.min_snapshots = int(min_snapshots)
        self.max_snapshots = int(max_snapshots)
        self._snaps: deque[dict[str, torch.Tensor]] = deque(maxlen=self.max_snapshots)
        self._applied = False
        #: Diagnostics, printed at train end. Not having these is why the run length above had to
        #: be inferred from wall-clock instead of read off the run.
        self.n_taken = 0
        self.n_averaged = 0
        self.final_step = 0

    # A model object can be fitted more than once (nf_fit_and_sweep fits once, but the callback
    # travels on `trainer_kwargs` and would otherwise accumulate state across fits).
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._snaps.clear()
        self._applied = False
        self.n_taken = 0
        self.n_averaged = 0
        self.final_step = 0

    def _snapshot(self, pl_module: pl.LightningModule) -> None:
        self._snaps.append(
            {
                name: p.detach().to("cpu", copy=True)
                for name, p in pl_module.named_parameters()
                if p.dtype.is_floating_point
            }
        )
        self.n_taken += 1

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_) -> None:
        step = int(trainer.global_step)
        if step > 0 and step % self.every_n_steps == 0:
            self._snapshot(pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Always include the final weights, so a run shorter than one snapshot interval still
        # averages something rather than silently leaving the model untouched.
        self._snapshot(pl_module)
        self.final_step = int(trainer.global_step)

        # The window: the last `tail_fraction` of everything TAKEN (not of `max_steps`, which the
        # run never reaches, and not of the buffer, which is only a memory cap).
        k = math.ceil(self.tail_fraction * self.n_taken)
        k = max(self.min_snapshots, k)
        k = min(k, len(self._snaps))
        if k < 2:
            print(
                f"[swa] NOT APPLIED: only {len(self._snaps)} snapshot(s) at step {self.final_step}",
                flush=True,
            )
            return

        window = list(self._snaps)[-k:]
        params = dict(pl_module.named_parameters())
        with torch.no_grad():
            for name in window[0]:
                stacked = torch.stack([s[name] for s in window])
                params[name].copy_(stacked.mean(dim=0).to(params[name].device))
        self.n_averaged = k
        self._applied = True
        print(
            f"[swa] averaged {k} of {self.n_taken} snapshots "
            f"(tail_fraction={self.tail_fraction}, every_n_steps={self.every_n_steps}), "
            f"covering ~{k * self.every_n_steps} of {self.final_step} steps",
            flush=True,
        )
