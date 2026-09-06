"""S10 G1: the seed must reach the HF Trainer, not just the global RNGs.

Found by measurement, not by reading: three `--seed` values produced BIT-IDENTICAL full-FT
checkpoints (max|dpred| exactly 0.0000 over 32,256 rows, against 9.24 for a genuinely different
model). `Chronos2Pipeline.fit` has no `seed` parameter; extra kwargs are forwarded verbatim to
`transformers.TrainingArguments`, whose own `seed` defaults to 42 and which the Trainer applies
internally — so `set_seed(892)` was overwritten before a single step ran.

This is S5.2's defect in a second library. There, `pl.seed_everything(self.random_seed)` inside
neuralforecast's `BaseModel` undid `registry.set_seed`, `MemberSpec.seedable=True` was a false
claim, and `member_seed_noise.py` would have reported sigma = 0 AS A MEASUREMENT. The lesson that
generalises is the one these tests encode: **a lever that does not reach the model returns a
confident null**, so assert that two settings differ rather than trusting that they do.
"""

from __future__ import annotations

from src.models.chronos2_finetune import (
    DEFAULTS,
    TRAINING_ARGS_DEFAULT_SEED,
    trainer_extra,
)


def test_the_seed_is_forwarded_to_training_arguments():
    """FAILS on the pre-fix code, where `extra` held only bf16 and the seed never travelled."""
    assert trainer_extra(892, "cuda", no_bf16=False)["seed"] == 892


def test_two_seeds_produce_different_trainer_kwargs():
    """The reachability property itself: distinct seeds must be distinguishable at the boundary."""
    a = trainer_extra(892, "cuda", no_bf16=False)
    b = trainer_extra(7739, "cuda", no_bf16=False)
    assert a != b, "the seed does not reach TrainingArguments — G1 would measure sigma = 0"


def test_the_default_seed_matches_the_trainers_own_so_nothing_recorded_moves():
    """The reason this fix is safe to apply AFTER checkpoints were fitted.

    Our default is 42 and `TrainingArguments`' default is 42, so an unseeded run forwards exactly
    the value the Trainer was already using. Every historical full-FT and LoRA checkpoint therefore
    still reproduces — the fix adds a capability rather than changing a result. If either default
    ever moves, this fails and the claim must be re-checked instead of assumed.
    """
    assert DEFAULTS["seed"] == TRAINING_ARGS_DEFAULT_SEED == 42
    assert trainer_extra(DEFAULTS["seed"], "cpu", no_bf16=True) == {"seed": 42}


def test_bf16_behaviour_is_unchanged():
    """The pre-existing contents of `extra` must survive the refactor untouched."""
    assert trainer_extra(42, "cuda", no_bf16=False)["bf16"] is True
    assert "bf16" not in trainer_extra(42, "cuda", no_bf16=True)
    assert "bf16" not in trainer_extra(42, "cpu", no_bf16=False)
