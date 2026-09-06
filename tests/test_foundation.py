"""S2 — the zero-shot foundation harness, exercised with a stub backend and no packages installed.

The whole point of the ``forecast_fn`` split is that the *harness* — windowing, the frame write, the
sidecar, the member arm, the two-channel merge — is testable locally at zero dependency cost, while
only the four model calls need their real package and those only ever run on Modal. So these tests
run the real code path end to end against a deterministic stub.

What is pinned here is what would fail *silently*, because that is the failure mode this covariate
surface has already produced once: plan 3.10 found a covariate whose context ran 336 hours past the
cutoff, described in its own docstring as "leakage-free". A leaky covariate scores **better**, so
nothing downstream complains. Hence: gap honesty checked as a property of the recorded grid rather
than of the numbers, and checked again after the frame is written rather than before.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    ID,
    KNOWN_FUTURE_SIGNALS,
    NAN_COLS,
    STATIC_COLS,
    TARGET,
    TIME,
    cascade_channels,
)
from src.data.loader import NF_ID, NF_TARGET, NF_TIME
from src.eval.splits import SCORE_LEN, add_hour_index
from src.models import cascade_provenance as cp
from src.models import foundation as fnd

N_SERIES = 3
N_HOURS = 1400
CUT = 700
HORIZON = 2 * SCORE_LEN  # 672 — the member runner slices the last SCORE_LEN of it


@pytest.fixture(scope="module")
def raw_csv(tmp_path_factory) -> str:
    """A raw-schema train.csv: what `run_one` reads and what `attach` must round-trip unchanged."""
    rng = np.random.default_rng(7)
    frames = []
    for s in range(N_SERIES):
        hidx = np.arange(N_HOURS)
        g = pd.DataFrame(
            {
                ID: f"unit_{s:03d}",
                TIME: pd.date_range("2023-01-01", periods=N_HOURS, freq="h"),
                TARGET: 20.0 * (s + 1)
                + 5 * np.sin(hidx / 24 * 2 * np.pi)
                + rng.normal(0, 1, N_HOURS),
            }
        )
        # Every known-future column the real `data/raw/train.csv` ships, EXCEPT the `*_missing`
        # flags — those are derived by `src.data.impute.apply_fill`, not supplied. The calendar
        # columns (`hour_sin`, `trend`, ...) are raw here because they are raw in the dataset too;
        # omitting them made the covariate backends fail on a frame the real one would satisfy.
        from src.data.features import MISSING_SUFFIX, futr_exog_list

        base_futr = [c for c in futr_exog_list() if not c.endswith(MISSING_SUFFIX)]
        for col in {*KNOWN_FUTURE_SIGNALS, *NAN_COLS, *base_futr}:
            g[col] = rng.normal(0, 1, N_HOURS)
        for col in STATIC_COLS:
            g[col] = float(s)
        frames.append(g)
    path = tmp_path_factory.mktemp("data") / "train.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return str(path)


def _stub(offset: float = 0.0):
    """A backend that returns a constant per series, so a misplaced row is arithmetically
    visible."""

    def fn(history: pd.DataFrame, h: int, device: str) -> np.ndarray:
        del device
        order = sorted(history[NF_ID].unique())
        return np.vstack([np.full(h, float(i) + offset + 1.0) for i, _ in enumerate(order)])

    return fn


# --------------------------------------------------------------------------- gap honesty


def test_the_horizon_block_is_one_call_anchored_at_the_cutoff(raw_csv, tmp_path):
    """The correction of plan 3.10, expressed as the recorded grid rather than as a docstring."""
    out = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(), device="cpu")
    meta = cp.read_provenance(out, "toto_forecast")
    assert (CUT, HORIZON) in [tuple(b) for b in meta["blocks"]]
    # The whole graded horizon is covered by a block whose context ends at cut-1.
    for hour in (CUT, CUT + HORIZON // 2, CUT + HORIZON - 1):
        assert cp.context_end_for_hour(meta, hour) == CUT - 1
    assert cp.gap_leak_hours(meta, CUT, HORIZON) == 0


def test_a_336h_rolling_grid_over_the_horizon_would_be_refused(raw_csv, tmp_path):
    """The exact shape that leaked, asserted to fail — otherwise the fence is untested.

    ``chronos2_oof``'s default grid puts a block start at ``cut + 336``, so the scored half draws a
    covariate whose context ended 336 hours past the cutoff. It is the flattering direction, so only
    the sidecar can catch it.
    """
    out = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(), device="cpu")
    leaky = {
        "column": "toto_forecast",
        "blocks": [[CUT, SCORE_LEN], [CUT + SCORE_LEN, SCORE_LEN]],
        "n_hours": N_HOURS,
    }
    cp.provenance_path(out).write_text(json.dumps(leaky))
    with pytest.raises(ValueError, match="NOT gap-honest"):
        cp.check_gap_honest(out, "toto_forecast", CUT, HORIZON)


def test_every_block_is_generated_from_a_context_that_stops_before_it(raw_csv, tmp_path):
    """Asserted inside ``_block`` on every call — pinned here so it cannot be relaxed silently."""
    seen: list[tuple[int, int]] = []

    def spy(history: pd.DataFrame, h: int, device: str) -> np.ndarray:
        df = add_hour_index(history)
        seen.append((int(df["_hidx"].max()), h))
        return _stub()(history, h, device)

    fnd.run_one(
        "timesfm", raw_csv, CUT, out_dir=tmp_path, forecast_fn=spy, device="cpu", train_region=True
    )
    meta = cp.read_provenance(fnd.derived_frame("timesfm", CUT, tmp_path), "timesfm_forecast")
    starts = sorted(s for s, _ in [tuple(b) for b in meta["blocks"]])
    # One context per block, each ending exactly one hour before its block starts.
    assert sorted(ctx_end for ctx_end, _ in seen) == [s - 1 for s in starts]


# --------------------------------------------------------------------------- the two modes


def test_horizon_only_leaves_the_train_region_unfilled(raw_csv, tmp_path):
    """Stage 1 pays for the horizon block alone; the train region is Stage 2's cost, named."""
    out = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(), device="cpu")
    df = pd.read_csv(out)
    assert df.groupby(ID)["toto_forecast"].apply(lambda s: s.notna().sum()).eq(HORIZON).all()


def test_train_region_fills_everything_but_the_warm_up(raw_csv, tmp_path):
    out = fnd.run_one(
        "toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(), device="cpu", train_region=True
    )
    df = pd.read_csv(out)
    warmup = CUT - (CUT // SCORE_LEN) * SCORE_LEN
    per_series = df.groupby(ID)["toto_forecast"].apply(lambda s: int(s.isna().sum()))
    # NaN exactly over the hours with no prior origin, and the frame's tail past cut+horizon.
    assert (per_series == warmup + (N_HOURS - CUT - HORIZON)).all()


def test_a_backend_returning_the_wrong_shape_raises(raw_csv, tmp_path):
    def wrong(history, h, device):
        del history, device
        return np.zeros((1, h))

    with pytest.raises(ValueError, match="one row per series"):
        fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=wrong, device="cpu")


def test_attach_preserves_the_raw_schema_plus_exactly_one_column(raw_csv, tmp_path):
    """Every downstream loader reads the raw schema; a richer format would break all of them."""
    out = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(), device="cpu")
    before = list(pd.read_csv(raw_csv, nrows=1).columns)
    after = list(pd.read_csv(out, nrows=1).columns)
    assert after == [*before, "toto_forecast"]


# --------------------------------------------------------------------------- the member arm


def test_the_member_arm_is_the_scored_block_of_the_same_forward_pass(raw_csv, tmp_path):
    """No second model call: the last SCORE_LEN steps of the horizon block ARE the member."""
    from src.models.members import RunContext, get_member, load_window_long

    out = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(), device="cpu")
    spec = get_member("toto_zeroshot")
    long_df = load_window_long(str(out), CUT, member="toto_zeroshot")
    with cascade_channels(*spec.cascade):
        preds = spec.run(RunContext(long_df=long_df, cut_idx=CUT, device="cpu"))

    assert list(preds.columns) == [NF_ID, NF_TIME, NF_TARGET, "toto_zeroshot"]
    assert preds.groupby(NF_ID).size().eq(SCORE_LEN).all()
    assert not preds["toto_zeroshot"].isna().any()
    # The stub emits a per-series constant, so the member's values identify the source row exactly.
    order = sorted(preds[NF_ID].unique())
    for i, sid in enumerate(order):
        assert np.allclose(preds.loc[preds[NF_ID] == sid, "toto_zeroshot"], float(i) + 1.0)


def test_the_member_is_cpu_deterministic_and_fenced():
    from src.models.members import get_member

    spec = get_member("toto_zeroshot")
    assert spec.needs_gpu is False, "the GPU was spent generating the frame, not reading it"
    assert spec.seedable is False, "a mean +- std over identical slices would fabricate precision"
    # It does not *condition* on the channel, it IS the channel — but the declaration is what makes
    # run_member verify provenance, and a leaky forecast scores better as a member too.
    assert spec.cascade == ("toto_forecast",)


# --------------------------------------------------------------------------- ALONGSIDE (2 channels)


def test_merge_frames_builds_a_two_channel_frame_whose_sidecar_covers_both(raw_csv, tmp_path):
    a = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(0), device="cpu")
    b = fnd.run_one("timesfm", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(10), device="cpu")
    merged = fnd.merge_frames(
        {"toto_forecast": a, "timesfm_forecast": b}, tmp_path / "both.csv", CUT, HORIZON
    )

    df = pd.read_csv(merged)
    assert {"toto_forecast", "timesfm_forecast"} <= set(df.columns)
    assert len(df) == N_SERIES * N_HOURS, "the merge must not duplicate or drop rows"
    # Each channel verifies independently: two channels are two honesty claims, not one.
    for col in ("toto_forecast", "timesfm_forecast"):
        assert cp.check_gap_honest(merged, col, CUT, HORIZON)["column"] == col


def test_a_multi_column_sidecar_refuses_to_guess_which_channel_you_meant(raw_csv, tmp_path):
    a = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(0), device="cpu")
    b = fnd.run_one("timesfm", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(10), device="cpu")
    merged = fnd.merge_frames(
        {"toto_forecast": a, "timesfm_forecast": b}, tmp_path / "both.csv", CUT, HORIZON
    )
    with pytest.raises(ValueError, match="name the one you mean"):
        cp.read_provenance(merged)
    with pytest.raises(ValueError, match="no provenance for column"):
        cp.check_gap_honest(merged, "tirex_forecast", CUT, HORIZON)


def test_the_single_column_sidecar_form_still_reads(raw_csv, tmp_path):
    """Every shipped sidecar is in the flat form; a format migration is the last thing a
    fail-closed check should require."""
    out = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(), device="cpu")
    assert "columns" not in __import__("json").loads(cp.provenance_path(out).read_text())
    assert cp.read_provenance(out)["column"] == "toto_forecast"
    assert cp.read_provenance(out, "toto_forecast")["column"] == "toto_forecast"


def test_merging_refuses_a_source_that_is_leaky_at_this_cutoff(raw_csv, tmp_path):
    """Honest-alone must not be launderable into a merged frame nothing rechecks."""
    a = fnd.run_one("toto", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(0), device="cpu")
    b = fnd.run_one("timesfm", raw_csv, CUT, out_dir=tmp_path, forecast_fn=_stub(10), device="cpu")
    # `b` is honest for CUT but not for an earlier cutoff — its horizon block starts at CUT.
    with pytest.raises(ValueError, match="NOT gap-honest"):
        fnd.merge_frames(
            {"toto_forecast": a, "timesfm_forecast": b},
            tmp_path / "both.csv",
            CUT - SCORE_LEN,
            HORIZON,
        )


# --------------------------------------------------------------------------- the registry wiring


@pytest.mark.parametrize("model", fnd.available_backends())
def test_every_backend_has_its_three_registrations_and_its_channel(model):
    from src.data.features import CASCADE_FORECASTS_ALL
    from src.models.members import get_member

    col = fnd.forecast_column(model)
    assert col in CASCADE_FORECASTS_ALL

    member = get_member(f"{model}_zeroshot")
    instead = get_member(f"tft_cascade_{model}")
    alongside = get_member(f"tft_cascade_chronos_{model}")

    assert member.cascade == (col,)
    assert instead.cascade == (col,), "INSTEAD-OF swaps the channel"
    assert alongside.cascade == ("chronos2_forecast", col), "ALONGSIDE carries both"
    # Both cascade arms point at frames that differ, because a two-channel member needs one frame
    # holding both columns — the merge, not either single-channel frame.
    assert instead.train_csv != alongside.train_csv
    for spec in (member, instead, alongside):
        assert "{cut}" in spec.train_csv, "a covariate is gap-honest for ONE cutoff"
        assert spec.status == "untested", "nothing here has been run"


# --- Toto's patch-divisibility precondition ------------------------------------------------
#
# `Toto2Model.forecast` patchifies the context, so its length must be an exact multiple of
# `patch_size`. It does not say so — it dies inside einops with "can't divide axis of length 3312
# in chunks of 32", which is why this cost a fan-out window before it cost a test.


@pytest.mark.parametrize(
    ("cut_idx", "expected"),
    [
        (3648, 0),  # 32 * 114 exactly
        (2976, 0),  # 32 * 93 exactly
        (3312, 16),  # 32 * 103 + 16 <- the one that failed
    ],
)
def test_every_cv_cutoff_yields_a_context_toto_can_patchify(cut_idx, expected):
    """The regression, stated as the three cutoffs the fan-out actually uses.

    Two of the three divide by 32 and one does not — and the smoke runs only 3648, which is one of
    the two that work. A gate that exercises a single point of a parameter the fan-out varies is
    blind to this whole class of bug, so the fix is pinned here rather than in the smoke.
    """
    offset = fnd.toto_context_offset(cut_idx, 32)
    assert offset == expected
    assert (cut_idx - offset) % 32 == 0, "the trimmed context must divide into whole patches"


def test_the_trim_only_ever_drops_the_oldest_hours():
    """Gap honesty lives at the RIGHT-hand end, so that is the end the trim must never touch."""
    hours = np.arange(3312)
    trimmed = hours[fnd.toto_context_offset(len(hours), 32) :]
    assert trimmed[-1] == hours[-1], "the cutoff end is untouched"
    assert len(hours) - len(trimmed) == 16
    assert trimmed[0] == 16, "and exactly the oldest 16 hours went"


def test_the_trim_is_a_no_op_when_the_context_already_divides():
    """Why the two toto frames already on the volume stay valid instead of being regenerated."""
    assert fnd.toto_context_offset(3648, 32) == 0
    assert fnd.toto_context_offset(2976, 32) == 0


def test_a_context_shorter_than_one_patch_is_an_error_not_an_empty_tensor():
    with pytest.raises(ValueError, match="shorter than one"):
        fnd.toto_context_offset(20, 32)
    with pytest.raises(ValueError, match="must be positive"):
        fnd.toto_context_offset(3312, 0)


# --- the covariate path --------------------------------------------------------------------------
#
# The three `_cov` backends only ever run on Modal, so a stub is the only thing that can exercise
# the plumbing for free. What it asserts is exactly what the confound was: that the candidate now
# receives the SAME covariates the control does, both sides of the cutoff, no target after it.


def test_a_covariate_backend_receives_the_controls_exog_on_both_frames(raw_csv, tmp_path):
    """The regression for the confounded screen, stated as what the backend is handed."""
    seen = {}

    def spy(context, future, h, device):
        seen["ctx_cols"] = list(context.columns)
        seen["fut_cols"] = list(future.columns)
        seen["h"] = h
        n = context[fnd.NF_ID].nunique()
        return np.zeros((n, h))

    fnd.register_cov_backend("_spy_cov", spy)
    try:
        fnd.run_one(
            "_spy_cov",
            raw_csv,
            CUT,
            out_dir=tmp_path,
            horizon=HORIZON,
            device="cpu",
            limit_series=2,
        )
    finally:
        fnd._COV_BACKENDS.pop("_spy_cov", None)

    from src.models.chronos2_oof import base_exog

    # The control's list minus its own column. Same set, or the confound is back, smaller.
    assert set(fnd.foundation_exog("_spy_cov")) == set(base_exog())

    exog = fnd.foundation_exog("_spy_cov")
    assert seen["ctx_cols"] == [fnd.NF_ID, fnd.NF_TIME, fnd.NF_TARGET, *exog]
    # The future frame carries the covariates and NOT the target: these are KNOWN-FUTURE covariates,
    # and handing a backend `y` over the horizon is the leak the whole module exists to prevent.
    assert seen["fut_cols"] == [fnd.NF_ID, fnd.NF_TIME, *exog]
    assert fnd.NF_TARGET not in seen["fut_cols"]
    assert seen["h"] == HORIZON


def test_a_target_only_backend_is_never_handed_covariates(raw_csv, tmp_path):
    """The stub takes three arguments; handing it four is a TypeError, not a silent ignore."""
    seen = {}

    def spy(history, h, device):
        seen["cols"] = list(history.columns)
        return np.zeros((history[fnd.NF_ID].nunique(), h))

    fnd.register_backend("_spy_plain", spy)
    try:
        fnd.run_one(
            "_spy_plain",
            raw_csv,
            CUT,
            out_dir=tmp_path,
            horizon=HORIZON,
            device="cpu",
            limit_series=2,
        )
    finally:
        fnd._BACKENDS.pop("_spy_plain", None)
    assert seen["cols"] == [fnd.NF_ID, fnd.NF_TIME, fnd.NF_TARGET]


def test_tirex_has_no_covariate_twin_and_that_is_deliberate():
    """v1 exposes no covariate argument; v2 has one but truncates our 672-step horizon to 320."""
    assert "tirex_cov" not in fnd.available_backends()
    assert not fnd.is_covariate_backend("tirex")
    assert fnd.is_covariate_backend("tabpfn_ts_cov")
    assert fnd.is_covariate_backend("timesfm_cov")


def test_a_covariate_backend_refuses_a_short_future_frame(raw_csv, tmp_path):
    """A covariate backend needs one covariate row per forecast hour, so a short tail must raise."""
    fnd.register_cov_backend("_spy_short", lambda c, f, h, d: np.zeros((c[fnd.NF_ID].nunique(), h)))
    try:
        with pytest.raises(ValueError, match="future hours"):
            fnd.run_one(
                "_spy_short",
                raw_csv,
                N_HOURS - 10,
                out_dir=tmp_path,
                horizon=HORIZON,
                device="cpu",
                limit_series=2,
            )
    finally:
        fnd._COV_BACKENDS.pop("_spy_short", None)


# --------------------------------------------------------------------------- the model slot
#
# Every backend used to build its model inside the per-block call. Harmless with a horizon-only
# frame (one load per model, per window) and not harmless under --train-region, which is ~11: the
# 2026-08-04 timesfm_cov run reached [train 8/8] and died on the horizon block holding 5.20 GiB of
# live torch tensors. Loading in the hot loop was the bug; the OOM was the symptom.


def test_a_repeated_key_loads_once(monkeypatch):
    calls = []
    monkeypatch.setattr(fnd, "_MODEL_SLOT", {})

    def factory():
        calls.append(1)
        return object()

    first = fnd.cached_model("m:336", factory)
    for _ in range(10):
        assert fnd.cached_model("m:336", factory) is first
    assert calls == [1], "the train blocks share one horizon and must share one load"


def test_a_new_key_evicts_the_previous_model(monkeypatch):
    """One slot, not an LRU. The key that varies is the compiled horizon, and the horizon block
    runs last and once — so keeping the h=336 model resident afterwards would hold memory that
    nothing will ask for again."""
    monkeypatch.setattr(fnd, "_MODEL_SLOT", {})
    train = fnd.cached_model("m:336", lambda: "train-model")
    horizon = fnd.cached_model("m:672", lambda: "horizon-model")

    assert horizon != train
    assert list(fnd._MODEL_SLOT) == ["m:672"], "the evicted model must not stay resident"


def test_eviction_releases_before_loading_not_after(monkeypatch):
    """Order matters on a full card: freeing after the new model is built needs both resident at
    once, which is the peak that OOMs."""
    monkeypatch.setattr(fnd, "_MODEL_SLOT", {})
    events = []
    monkeypatch.setattr(fnd, "_release_gpu", lambda: events.append("release"))

    fnd.cached_model("m:336", lambda: events.append("load-a"))
    fnd.cached_model("m:672", lambda: events.append("load-b"))

    assert events == ["release", "load-a", "release", "load-b"]


def test_distinct_backends_do_not_share_a_slot(monkeypatch):
    """The keys carry the backend name, so timesfm and its _cov twin — same class, different
    compile (return_backcast) — cannot be handed each other's model."""
    monkeypatch.setattr(fnd, "_MODEL_SLOT", {})
    assert fnd.cached_model("timesfm:672", lambda: "plain") == "plain"
    assert fnd.cached_model("timesfm_cov:672", lambda: "cov") == "cov"
    assert fnd.cached_model("timesfm_cov:672", lambda: pytest.fail("reloaded")) == "cov"


def test_the_gpu_note_is_silent_without_cuda_and_never_raises():
    """Diagnostics run inside the generation loop, so a broken probe must not break a GPU run."""
    assert isinstance(fnd._gpu_note(), str)
