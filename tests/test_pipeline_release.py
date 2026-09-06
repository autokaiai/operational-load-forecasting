"""The Chronos pipeline must be released after use, or rung 1 silently never runs.

S10's archive makes TWO Chronos-2 forward passes per inference — the zero-shot cascade channel and
then the fine-tuned blend member — at 455.8 MB of weights each. The clean-room run measured what
happens without an explicit release: 5.94 GiB resident, a 1.02 GiB allocation refused, and the
member degrading to rung 2 on an 8 GB card.

That is the DANGEROUS kind of failure, which is why it gets a test rather than a comment. An OOM
here does not crash the submission, it degrades it — so a modest grading GPU would have shipped the
previous model at 0.13163 while every log line still looked healthy and the CSV was perfectly valid.
Same family as #32 and the bag defect: a correct-looking artifact describing the wrong thing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import NF_ID, NF_TIME
from src.models import cascade_inference as ci


class _FakeModel:
    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, device):
        self.moved_to.append(str(device))
        return self


class _FakePipe:
    def __init__(self) -> None:
        self.model = _FakeModel()


@pytest.fixture
def wired(monkeypatch):
    """A generate_channel call with the heavy parts replaced, so only the wiring is under test."""
    pipe = _FakePipe()
    monkeypatch.setattr(ci, "_load_pipeline_from", lambda source, device: pipe)

    hist = pd.DataFrame(
        {NF_ID: ["a"] * 4, NF_TIME: pd.date_range("2023-01-01", periods=4, freq="h")}
    )
    hist["y"] = 1.0
    monkeypatch.setattr(ci, "history_from_bundle", lambda nf: hist)

    futr = pd.DataFrame(
        {NF_ID: ["a", "a"], NF_TIME: pd.date_range("2023-01-01 04:00", periods=2, freq="h")}
    )

    def fake_predict(p, ctx, fut, h, batch_series):
        out = fut[[NF_ID, NF_TIME]].copy()
        out["pred"] = [1.0, 2.0]
        return out

    monkeypatch.setattr("src.models.chronos2_eval._predict", fake_predict)
    monkeypatch.setattr("src.models.chronos2_eval._pick_pred_column", lambda raw: "pred")
    monkeypatch.setattr("src.models.chronos2_oof.base_exog", lambda: [])
    return pipe, futr


def test_the_pipeline_is_released_after_a_successful_forecast(wired):
    """FAILS before the fix: `generate_channel` simply let `pipe` fall out of scope, and the CUDA
    caching allocator keeps those blocks — so the next pipeline loads on top of the first."""
    pipe, futr = wired
    out = ci.generate_channel(None, futr, source="fake", device="cuda")
    assert list(out[ci.CHRONOS_COL]) == [1.0, 2.0]
    assert pipe.model.moved_to == ["cpu"], "the weights were never moved off the GPU"


def test_the_pipeline_is_released_even_when_the_forecast_RAISES(wired):
    """The release is in a `finally` on purpose. An OOM mid-forecast is exactly when the next
    attempt most needs the memory back — `run_fullft`'s batched retry is that next attempt."""
    pipe, futr = wired

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.models.chronos2_eval._predict", boom)
        with _pytest.raises(RuntimeError, match="out of memory"):
            ci.generate_channel(None, futr, source="fake", device="cuda")
    assert pipe.model.moved_to == ["cpu"], "a failed forecast leaked its weights"


def test_release_never_masks_a_successful_forecast(monkeypatch, wired):
    """Cleanup that raises must not turn a completed forecast into a failure."""
    pipe, futr = wired

    def _boom(self, d):
        raise RuntimeError("nope")

    monkeypatch.setattr(_FakeModel, "to", _boom)
    out = ci.generate_channel(None, futr, source="fake", device="cuda")
    assert list(out[ci.CHRONOS_COL]) == [1.0, 2.0]
