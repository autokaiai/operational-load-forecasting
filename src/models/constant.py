"""Model 0: a checkpoint-driven constant forecaster.

Placeholder architecture that satisfies the submission contract end-to-end before any
real modeling. It holds a single learnable scalar (`bias`) and predicts that scalar for
every horizon step. Swapping in a real model later means replacing this class and the
inference body in ``predict.py`` — the CLI contract stays untouched.
"""

from __future__ import annotations

import torch


class ConstantForecastModel(torch.nn.Module):
    """A one-parameter forecaster that emits a single learned constant for every step."""

    def __init__(self, bias: float = 0.0) -> None:
        """Create the model with its scalar prediction initialized to ``bias``."""
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(bias)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` shifted by the learned scalar (keeps a real tensor op for parity)."""
        return x + self.bias

    def constant(self) -> float:
        """Return the scalar value predicted for every forecast row."""
        return float(self.bias.detach())
