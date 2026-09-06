"""The feature cascade is opt-in — and the tests that matter are the ones about it being OFF.

Background. `configs/tft_chronos.yaml` said `chronos2_forecast` is "registered in
src.data.features". It was not: the registering commit never landed, and `git merge-base` shows
The registering commit never reached `main`; it survives only in the sibling cascade worktrees
this project was built across. So running that config **from this repo** produced a plain TFT
that silently ignored the Chronos column — the config was inert while looking functional.

Be precise about what that does *not* mean: a real cascade **was** run, in those sibling worktrees.
The cached predictions differ from plain TFT by up to 9.35 (mean 0.73), so the covariate was
demonstrably active. What was never run is an *honest* one — every cascade to date used the leaky
rolling covariate (see `src.models.cascade_provenance`). Two distinct defects, one config line.

The obvious fix — put the column back in `KNOWN_FUTURE_SIGNALS` — is the one that must not be made.
`futr_exog_list()` is global, so every plain member would then demand a column `train.csv` does not
have. Hence a scoped opt-in, and hence these tests: most of them assert that nothing changes.
"""

from __future__ import annotations

import pytest

import src.data.features as F
from src.models import members as mem


@pytest.fixture(autouse=True)
def _restore_channels():
    """No test may leak an activation into the next one."""
    before = F.active_cascade_channels()
    yield
    F.set_cascade_channels(before)


# --------------------------------------------------------------------------- off by default


def test_no_cascade_channel_is_active_by_default():
    assert F.active_cascade_channels() == []


def test_the_plain_conditioning_set_is_untouched():
    """The regression that would silently break every non-cascade member."""
    futr = F.futr_exog_list()
    assert "chronos2_forecast" not in futr
    assert "chronos2_forecast_missing" not in futr
    assert "chronos2_forecast" not in F.nan_col_list()


def test_the_plain_conditioning_set_is_exactly_the_documented_one():
    expected = [*F.TIME_ENCODINGS, *F.KNOWN_FUTURE_SIGNALS, *F.missing_indicator_cols()]
    assert F.futr_exog_list() == expected


# --------------------------------------------------------------------------- on when asked


def test_activating_a_channel_adds_the_column_and_its_missing_flag():
    with F.cascade_channels("chronos2_forecast"):
        futr = F.futr_exog_list()
        assert "chronos2_forecast" in futr
        # The warm-up NaNs (~6.7% of train_chronos.csv) need imputing and flagging like any other
        # NaN-prone signal, or the model sees NaN for its first 288 hours.
        assert "chronos2_forecast_missing" in futr
        assert "chronos2_forecast" in F.nan_col_list()


def test_the_scope_is_restored_afterwards():
    with F.cascade_channels("chronos2_forecast"):
        assert F.active_cascade_channels() == ["chronos2_forecast"]
    assert F.active_cascade_channels() == []


def test_the_scope_is_restored_even_when_the_body_raises():
    with pytest.raises(RuntimeError), F.cascade_channels("chronos2_forecast"):
        raise RuntimeError("boom")
    assert F.active_cascade_channels() == []


def test_scopes_nest_and_unwind_in_order():
    with F.cascade_channels("chronos2_forecast"):
        with F.cascade_channels():
            assert F.active_cascade_channels() == []
        assert F.active_cascade_channels() == ["chronos2_forecast"]


def test_a_channel_accepts_a_string_or_a_sequence():
    with F.cascade_channels("chronos2_forecast"):
        a = F.active_cascade_channels()
    with F.cascade_channels(["chronos2_forecast"]):
        b = F.active_cascade_channels()
    assert a == b == ["chronos2_forecast"]


def test_an_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="unknown cascade channel"):
        # a real channel in the sibling cascade worktree, but not registered here
        F.set_cascade_channels(["lgbm_forecast"])


def test_env_var_parsing_ignores_blanks(monkeypatch):
    monkeypatch.setenv("CASCADE_CHANNELS", " chronos2_forecast , ,")
    assert F._channels_from_env() == ["chronos2_forecast"]
    monkeypatch.setenv("CASCADE_CHANNELS", "")
    assert F._channels_from_env() == []


# --------------------------------------------------------------------------- the member


def test_the_cascade_member_is_registered_with_both_halves():
    """Declaring a channel without the derived CSV is the silent failure this guards."""
    spec = mem.get_member("tft_cascade")
    assert spec.cascade == ("chronos2_forecast",)
    # Per WINDOW: a cascade covariate is gap-honest only for the cutoff it was anchored at, so
    # there is no single frame that serves all three (src.models.cascade_provenance).
    assert spec.train_csv == "data/derived/train_chronos_cut{cut}.csv"
    assert spec.kind == "neural" and spec.needs_gpu


def test_the_cascade_member_uses_zero_shot_not_the_fine_tune():
    """A standing requirement, and it is a design point rather than a preference.

    The cascade's job is to hand the TFT a *global prior* its Variable Selection Network can learn
    when to trust. A per-window fine-tune would make the covariate a second fitted model whose
    errors correlate with `chronos_ft` — already an ensemble member in its own right.
    """
    src = (mem.Path(__file__).parent.parent / "src" / "models" / "chronos2_oof.py").read_text()
    assert "_load_pipeline(device)" in src, "the OOF generator must load Chronos with NO adapter"
    assert "adapter=" not in src


def test_a_member_declaring_a_channel_without_a_train_csv_is_rejected(monkeypatch):
    monkeypatch.setattr(mem, "_REGISTRY", dict(mem._REGISTRY))
    with pytest.raises(ValueError, match="no train_csv"):
        mem.register(
            mem.MemberSpec(
                name="bad_cascade",
                kind="neural",
                run=lambda ctx: None,
                seedable=True,
                needs_gpu=True,
                cascade=("chronos2_forecast",),
            )
        )


def test_a_member_declaring_an_unknown_channel_is_rejected(monkeypatch):
    monkeypatch.setattr(mem, "_REGISTRY", dict(mem._REGISTRY))
    with pytest.raises(ValueError, match="unknown cascade channel"):
        mem.register(
            mem.MemberSpec(
                name="bad_channel",
                kind="neural",
                run=lambda ctx: None,
                seedable=True,
                needs_gpu=True,
                cascade=("bitcn_forecast",),
                train_csv="data/derived/train_chronos.csv",
            )
        )


def test_member_train_csv_routes_the_cascade_to_its_per_window_frame():
    assert mem.member_train_csv("tft_cascade", "data/raw/train.csv", 3312) == (
        "data/derived/train_chronos_cut3312.csv"
    )
    assert mem.member_train_csv("tft", "data/raw/train.csv") == "data/raw/train.csv"
    assert mem.member_train_csv(None, "data/raw/train.csv") == "data/raw/train.csv"


def test_a_per_window_frame_cannot_be_resolved_without_a_cutoff():
    """Silently defaulting would hand one window another window's covariate — a leak by aliasing."""
    with pytest.raises(ValueError, match="needs a per-window frame"):
        mem.member_train_csv("tft_cascade", "data/raw/train.csv")


def test_running_a_cascade_member_on_a_frame_lacking_the_column_fails_loudly(monkeypatch):
    """The exact silent failure that produced a 'cascade' which was really a plain TFT."""
    import pandas as pd

    monkeypatch.setattr(mem, "_REGISTRY", dict(mem._REGISTRY))
    mem.register(
        mem.MemberSpec(
            name="casc_probe",
            kind="neural",
            run=lambda ctx: None,
            seedable=True,
            needs_gpu=False,
            cascade=("chronos2_forecast",),
            train_csv="data/derived/train_chronos.csv",
        )
    )
    plain = pd.DataFrame({"unique_id": ["u"], "ds": [0], "y": [1.0]})
    with pytest.raises(AssertionError, match="chronos2_forecast"):
        mem.run_member("casc_probe", mem.RunContext(long_df=plain, cut_idx=3648))
