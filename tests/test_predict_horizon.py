"""S9.1a — the submission path exercised against inputs it has never seen.

The #32 defect was found by reading code and checking indices. That is how it was *found*; it is
not how it gets *closed*. `predict.py` forecasts ``h`` steps from the end of the checkpoint's
stored history, the private index sits **+337h** out, and the old ``_assign_predictions`` matched
**BY POSITION** — so a forecast of entirely the wrong hours produced the right row count, the right
schema, the right series and the requested timestamps in the output. The only symptom was a score.

Three properties every fixture here has, each because a weaker version passes on `main` today:

1. **The gap is non-zero AND not 336.** The spec: *"the timeframe might differ."* A fixture that
   only covers 336 bakes the same assumption into the test that the code had. The leaderboard case
   (gap 0) and an odd gap are both covered, because a hardcoded 336 gets BOTH wrong.
2. **The covariates carry the real ~4.5% scattered-NaN pattern**, so the imputation path actually
   runs instead of being skipped.
3. **Units and hours are SHUFFLED** relative to the index order. Sorted-and-aligned is the one
   arrangement under which a positional match is accidentally correct.

``test_positional_matching_would_have_been_WRONG_here`` is the load-bearing one: it reconstructs
what the old implementation did and asserts it disagrees with the timestamp join. If that ever
stops being true the fixture has drifted back to the easy case and is no longer testing the fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import predict
from src.data.features import ID, KNOWN_FUTURE_SIGNALS, TIME, TIME_ENCODINGS, nan_col_list
from src.data.impute import fit_fill_stats
from src.data.loader import NF_ID, NF_TIME, build_futr_df

SERIES = [f"unit_{i:03d}" for i in range(4)]
SPAN = 12  # hours requested, for the pure join tests where the block size is irrelevant
# The futr tests use the REAL span: 336h is two whole weeks, so the supplied block covers every
# hour-of-week bin twice and S3's `exact` modifier can restore the deterministic columns. A
# shorter block cannot, which is a genuine constraint of the submission path and is guarded.
COV_SPAN = 336
HISTORY = 24 * 21  # enough hours for every hour-of-week bin to appear
NAN_RATE = 0.045  # the measured rate in the 10 NaN-prone columns


def _covariates(start: pd.Timestamp, periods: int, *, seed: int = 0) -> pd.DataFrame:
    """A covariate frame with the dataset's real semantics: exact calendar, linear trend, NaNs."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=periods, freq="h")
    frames = []
    for s, sid in enumerate(SERIES):
        hours = (ts - pd.Timestamp("2023-01-01")).total_seconds().to_numpy() / 3600.0
        f = pd.DataFrame({ID: sid, TIME: ts})
        f["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24.0)
        f["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24.0)
        f["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7.0)
        f["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7.0)
        f["is_weekend"] = (ts.dayofweek >= 5).astype(float)
        f["trend"] = 1.2660764 + 6.939306e-04 * hours
        for j, col in enumerate(KNOWN_FUTURE_SIGNALS):
            if col == "workload_intensity":  # deterministic in (series, hour-of-week), per S3
                f[col] = 5.0 + s + np.sin(2 * np.pi * (ts.dayofweek * 24 + ts.hour) / 168.0)
            elif col in ("promotion_intensity", "maintenance_known"):  # sparse event flags
                f[col] = (rng.random(periods) < 0.03).astype(float)
            else:
                f[col] = 1.0 + j + s + rng.normal(0, 0.1, periods)
        frames.append(f)
    cov = pd.concat(frames, ignore_index=True)
    for col in nan_col_list():  # the real scattered pattern: isolated single hours
        cov.loc[rng.random(len(cov)) < NAN_RATE, col] = np.nan
    return cov


def _shuffled_index(ts: pd.DatetimeIndex, *, seed: int = 7) -> pd.DataFrame:
    """Forecast index in deliberately scrambled (series, hour) order."""
    rows = [{ID: sid, TIME: t.strftime("%Y-%m-%d %H:%M:%S")} for sid in SERIES for t in ts]
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- horizon arithmetic


@pytest.mark.parametrize("gap", [0, 1, 100, 336, 337])
def test_horizon_plan_derives_any_gap_not_just_336(gap: int) -> None:
    """The leaderboard is gap 0 and the grade is gap 336, so neither may be hardcoded."""
    last_observed = pd.Timestamp("2023-06-29 23:00:00")
    first = last_observed + pd.Timedelta(hours=gap + 1)
    fi = _shuffled_index(pd.date_range(first, periods=SPAN, freq="h"))

    plan = predict.horizon_plan(last_observed, fi)
    assert plan["gap"] == gap
    assert plan["span"] == SPAN
    assert plan["h_required"] == gap + SPAN


def test_horizon_plan_rejects_an_index_at_or_before_the_last_observed_hour() -> None:
    last_observed = pd.Timestamp("2023-06-29 23:00:00")
    fi = _shuffled_index(pd.date_range(last_observed, periods=SPAN, freq="h"))
    with pytest.raises(ValueError, match="not after last observed"):
        predict.horizon_plan(last_observed, fi)


# --------------------------------------------------------------------------- the join


def _preds_for(ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Model output over ``ts``: value encodes (series, hour) so a mismatch is detectable."""
    rows = []
    for s, sid in enumerate(SERIES):
        for t in ts:
            rows.append({NF_ID: sid, NF_TIME: t, "tft": 1000 * s + t.dayofyear * 24 + t.hour})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=3).reset_index(drop=True)


def test_the_join_returns_the_value_for_the_REQUESTED_hour() -> None:
    """With a 336-step forecast covering gap+span, the requested tail must come back exactly."""
    last_observed = pd.Timestamp("2023-06-29 23:00:00")
    gap = 100
    horizon = pd.date_range(last_observed + pd.Timedelta(hours=1), periods=gap + COV_SPAN, freq="h")
    requested = horizon[gap:]
    fi = _shuffled_index(requested)

    got = predict.assign_predictions(fi, _preds_for(horizon), "tft")

    want = [
        1000 * SERIES.index(r[ID])
        + pd.Timestamp(r[TIME]).dayofyear * 24
        + pd.Timestamp(r[TIME]).hour
        for _, r in fi.iterrows()
    ]
    np.testing.assert_array_equal(got, np.array(want, dtype=float))


def test_positional_matching_would_have_been_WRONG_here() -> None:
    """The regression this file exists for — and it MUST FAIL on the pre-fix implementation.

    Reconstructs the retired positional match: sort both frames, zip them. With a gap the model's
    horizon starts 100h before the requested block, so position hands back the GAP's values under
    the requested block's labels. Row count, schema and series are all still perfect.
    """
    last_observed = pd.Timestamp("2023-06-29 23:00:00")
    gap = 100
    horizon = pd.date_range(last_observed + pd.Timedelta(hours=1), periods=gap + COV_SPAN, freq="h")
    fi = _shuffled_index(horizon[gap:])
    preds = _preds_for(horizon)

    correct = predict.assign_predictions(fi, preds, "tft")

    # --- what the old implementation did, verbatim in spirit ---
    f = fi.copy()
    f["_order"] = range(len(f))
    f_sorted = f.sort_values([ID, TIME]).reset_index(drop=True)
    p_sorted = (
        preds.rename(columns={NF_ID: ID, NF_TIME: TIME, "tft": "prediction"})
        .sort_values([ID, TIME])
        .reset_index(drop=True)
    )
    # It could not even line up here: 4*112 forecast rows against 4*12 requested. The historical
    # fixture hid that by making the counts match while the HOURS did not, so emulate that case.
    per_series = len(p_sorted) // len(SERIES)
    take = (
        p_sorted.groupby(ID, group_keys=False)
        .apply(lambda g: g.head(len(f_sorted) // len(SERIES)), include_groups=True)
        .reset_index(drop=True)
    )
    f_sorted["positional"] = take["prediction"].to_numpy()
    positional = f_sorted.sort_values("_order")["positional"].to_numpy()

    assert per_series == gap + COV_SPAN
    assert not np.array_equal(correct, positional), (
        "positional matching agrees with the timestamp join — the fixture has drifted back to the "
        "aligned case and is no longer testing the #32 fix"
    )


def test_a_short_forecast_RAISES_instead_of_silently_mislabelling() -> None:
    """The #32 defect made loud: a model that cannot reach the requested hours must stop."""
    last_observed = pd.Timestamp("2023-06-29 23:00:00")
    horizon = pd.date_range(last_observed + pd.Timedelta(hours=1), periods=SPAN, freq="h")
    fi = _shuffled_index(horizon + pd.Timedelta(hours=336))  # requested block is far away
    with pytest.raises(ValueError, match="got no prediction"):
        predict.assign_predictions(fi, _preds_for(horizon), "tft")


# --------------------------------------------------------------------------- the futr frame


@pytest.mark.parametrize("gap", [0, 100, 336])
def test_futr_frame_spans_the_whole_horizon_with_no_NaN(gap: int) -> None:
    """``nf.predict`` needs one futr row per forecast STEP, and with a gap only the tail is
    supplied by any file. A NaN here propagates to a NaN loss, so the builder must leave none."""
    start = pd.Timestamp("2023-01-01")
    cov_all = _covariates(start, HISTORY + gap + COV_SPAN)
    last_observed = start + pd.Timedelta(hours=HISTORY - 1)

    # Only the REQUESTED block is supplied — the gap is covariate-absent, as at test time.
    requested = pd.date_range(
        last_observed + pd.Timedelta(hours=gap + 1), periods=COV_SPAN, freq="h"
    )
    supplied = cov_all[cov_all[TIME].isin(requested)]
    fill_stats = fit_fill_stats(cov_all[cov_all[TIME] <= last_observed])

    horizon = pd.date_range(last_observed + pd.Timedelta(hours=1), periods=gap + COV_SPAN, freq="h")
    target = pd.DataFrame([{ID: sid, TIME: t} for sid in SERIES for t in horizon])

    futr = build_futr_df(supplied, target, fill_stats)
    assert len(futr) == len(SERIES) * (gap + COV_SPAN)
    assert not futr.isna().any().any(), "a NaN in futr_exog becomes a NaN loss"
    assert list(futr.columns[:2]) == [NF_ID, NF_TIME]


def test_absent_rows_are_flagged_as_reconstructed_and_supplied_rows_are_not() -> None:
    """`*_missing` must mean "this row was reconstructed" — the same 1.0
    ``_withhold_gap_covariates`` writes. If ``apply_fill`` were run after the gap reconstruction it
    would see filled values and record a confident 0 exactly where everything was imputed."""
    start = pd.Timestamp("2023-01-01")
    gap = 48
    cov_all = _covariates(start, HISTORY + gap + COV_SPAN, seed=11)
    last_observed = start + pd.Timedelta(hours=HISTORY - 1)
    requested = pd.date_range(
        last_observed + pd.Timedelta(hours=gap + 1), periods=COV_SPAN, freq="h"
    )
    supplied = cov_all[cov_all[TIME].isin(requested)]
    fill_stats = fit_fill_stats(cov_all[cov_all[TIME] <= last_observed])

    horizon = pd.date_range(last_observed + pd.Timedelta(hours=1), periods=gap + COV_SPAN, freq="h")
    target = pd.DataFrame([{ID: sid, TIME: t} for sid in SERIES for t in horizon])
    futr = build_futr_df(supplied, target, fill_stats)

    flag = "demand_forecast_missing"
    gap_rows = futr[futr[NF_TIME] < requested[0]]
    assert (gap_rows[flag] == 1.0).all(), "gap rows must be flagged as reconstructed"
    # Supplied rows keep the honest per-row answer, so at a 4.5% NaN rate they are mostly 0.
    supplied_rows = futr[futr[NF_TIME] >= requested[0]]
    assert supplied_rows[flag].mean() < 0.5


def test_the_deterministic_calendar_is_restored_exactly_over_the_gap() -> None:
    """Five encodings via S3's ``exact`` detector, ``trend`` via the fitted line — both to
    floating point, because both are computable and neither may be median-filled."""
    start = pd.Timestamp("2023-01-01")
    gap = 72
    cov_all = _covariates(start, HISTORY + gap + COV_SPAN, seed=5)
    last_observed = start + pd.Timedelta(hours=HISTORY - 1)
    requested = pd.date_range(
        last_observed + pd.Timedelta(hours=gap + 1), periods=COV_SPAN, freq="h"
    )
    supplied = cov_all[cov_all[TIME].isin(requested)]
    fill_stats = fit_fill_stats(cov_all[cov_all[TIME] <= last_observed])

    horizon = pd.date_range(last_observed + pd.Timedelta(hours=1), periods=gap + COV_SPAN, freq="h")
    target = pd.DataFrame([{ID: sid, TIME: t} for sid in SERIES for t in horizon])
    futr = build_futr_df(supplied, target, fill_stats).rename(columns={NF_ID: ID, NF_TIME: TIME})

    truth = cov_all.merge(futr[[ID, TIME]], on=[ID, TIME])
    got = futr.merge(truth[[ID, TIME, *TIME_ENCODINGS]], on=[ID, TIME], suffixes=("", "_true"))
    for col in TIME_ENCODINGS:
        np.testing.assert_allclose(
            got[col].to_numpy(), got[f"{col}_true"].to_numpy(), atol=1e-8, err_msg=col
        )


def test_a_SHORT_covariate_block_still_restores_the_calendar_exactly() -> None:
    """The bug this file's first run caught, now pinned as fixed.

    S3's ``exact`` modifier needs ``EXACT_MIN_OBS=3`` observations per (series, hour-of-week) bin.
    The submission supplies 336h = two whole weeks = exactly **2**, so it misses by one and every
    encoding falls through to a per-series median — a CONSTANT ``hour_sin`` across the gap, smooth
    and undetectable. ``src.data.calendar`` computes them instead, so block length is irrelevant:
    12 supplied hours restore the gap as exactly as 336 do.
    """
    start = pd.Timestamp("2023-01-01")
    gap, short = 48, 12
    cov_all = _covariates(start, HISTORY + gap + short, seed=2)
    last_observed = start + pd.Timedelta(hours=HISTORY - 1)
    requested = pd.date_range(last_observed + pd.Timedelta(hours=gap + 1), periods=short, freq="h")
    supplied = cov_all[cov_all[TIME].isin(requested)]
    fill_stats = fit_fill_stats(cov_all[cov_all[TIME] <= last_observed])

    horizon = pd.date_range(last_observed + pd.Timedelta(hours=1), periods=gap + short, freq="h")
    target = pd.DataFrame([{ID: sid, TIME: t} for sid in SERIES for t in horizon])
    futr = build_futr_df(supplied, target, fill_stats).rename(columns={NF_ID: ID, NF_TIME: TIME})

    truth = cov_all.merge(futr[[ID, TIME]], on=[ID, TIME])
    got = futr.merge(truth[[ID, TIME, *TIME_ENCODINGS]], on=[ID, TIME], suffixes=("", "_true"))
    for col in TIME_ENCODINGS:
        np.testing.assert_allclose(
            got[col].to_numpy(), got[f"{col}_true"].to_numpy(), atol=1e-8, err_msg=col
        )
    # and the gap block is genuinely varying, not a constant the median would have produced
    gap_rows = got[got[TIME] < requested[0]]
    assert gap_rows["hour_sin"].nunique() > 1


def test_a_DIFFERENT_calendar_convention_RAISES_rather_than_being_extrapolated() -> None:
    """The closed form is trusted on absent rows only because it is checked on supplied ones.

    If the dataset ever encoded the hour differently, silently applying sin(2*pi*h/24) across the
    gap would be the same class of error as a median-filled constant. So it fails closed.
    """
    start = pd.Timestamp("2023-01-01")
    gap, short = 48, 24
    cov_all = _covariates(start, HISTORY + gap + short, seed=4)
    last_observed = start + pd.Timedelta(hours=HISTORY - 1)
    requested = pd.date_range(last_observed + pd.Timedelta(hours=gap + 1), periods=short, freq="h")
    supplied = cov_all[cov_all[TIME].isin(requested)].copy()
    supplied["hour_sin"] = supplied["hour_sin"] * 0.5  # a different convention
    fill_stats = fit_fill_stats(cov_all[cov_all[TIME] <= last_observed])

    horizon = pd.date_range(last_observed + pd.Timedelta(hours=1), periods=gap + short, freq="h")
    target = pd.DataFrame([{ID: sid, TIME: t} for sid in SERIES for t in horizon])
    with pytest.raises(ValueError, match="do not match the closed form"):
        build_futr_df(supplied, target, fill_stats)


def test_an_ACTIVE_CASCADE_CHANNEL_survives_the_futr_build() -> None:
    """A cascade channel must reach the model, and it nearly did not.

    With the channel active, ``futr_exog_list()`` gains ``chronos2_forecast`` (and its
    ``*_missing`` flag). The builder originally selected its source columns from a hardcoded
    ``TIME_ENCODINGS + KNOWN_FUTURE_SIGNALS``, so the channel was **dropped at the merge** even
    though the covariate frame carried it — silently, until the completeness check fired. That
    channel is worth **+8.3%** to the ship member, which is the whole margin of the project's best
    model, so it gets a test rather than a comment.
    """
    from src.data.features import cascade_channels, futr_exog_list

    start = pd.Timestamp("2023-01-01")
    gap = 24
    with cascade_channels("chronos2_forecast"):
        cov_all = _covariates(start, HISTORY + gap + COV_SPAN, seed=9)
        # the channel is a column on the covariate frame, exactly as validation_input_chronos.csv
        cov_all["chronos2_forecast"] = 7.0 + np.arange(len(cov_all)) % 13

        last_observed = start + pd.Timedelta(hours=HISTORY - 1)
        requested = pd.date_range(
            last_observed + pd.Timedelta(hours=gap + 1), periods=COV_SPAN, freq="h"
        )
        supplied = cov_all[cov_all[TIME].isin(requested)]
        fill_stats = fit_fill_stats(cov_all[cov_all[TIME] <= last_observed])

        horizon = pd.date_range(
            last_observed + pd.Timedelta(hours=1), periods=gap + COV_SPAN, freq="h"
        )
        target = pd.DataFrame([{ID: sid, TIME: t} for sid in SERIES for t in horizon])

        futr = build_futr_df(supplied, target, fill_stats)
        assert "chronos2_forecast" in futr.columns, "the cascade channel was dropped"
        assert "chronos2_forecast_missing" in futr.columns
        assert set(futr.columns) == {NF_ID, NF_TIME, *futr_exog_list()}
        assert not futr.isna().any().any()
        # the supplied block keeps the real channel values, not a median
        got = futr[futr[NF_TIME].isin(requested)]["chronos2_forecast"]
        assert got.nunique() > 1


# ------------------------------------------------------- the WHOLE CLI, at an arbitrary gap


@pytest.mark.parametrize("gap", [0, 17, 96])
def test_the_cli_handles_ANY_gap_end_to_end(tmp_path, gap: int) -> None:
    """S9.1a closed at the CLI level, not just at component level.

    Everything above tests `horizon_plan`, the join and `build_futr_df` in isolation. This runs
    the real ``predict.py`` against a real checkpoint at three gaps — including an ODD one that
    matches neither the leaderboard's 0 nor the grade's 336 — and asserts the numbers written out
    are the model's forecast for the REQUESTED hours.

    The comparison is what makes it bite. Index completeness and row count were both satisfied by
    the retired positional match while it described entirely the wrong hours, so this recomputes
    the forecast independently and asserts equality at the requested timestamps.
    """
    pytest.importorskip("neuralforecast")

    import subprocess
    import sys
    from pathlib import Path

    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import MLP

    from src import bundle
    from src.data.features import futr_exog_list

    repo_root = Path(__file__).resolve().parent.parent
    span = 24
    start = pd.Timestamp("2023-01-01")
    rng = np.random.default_rng(3)

    # history + the whole horizon, so the supplied block can be sliced out of a coherent frame
    cov_all = _covariates(start, HISTORY + gap + span, seed=gap + 1)
    last_observed = start + pd.Timedelta(hours=HISTORY - 1)
    hist = cov_all[cov_all[TIME] <= last_observed].copy()
    hist["target"] = 10 + rng.normal(0, 0.5, len(hist))

    fill_stats = fit_fill_stats(hist)
    from src.data.impute import apply_fill

    long_df = apply_fill(hist, fill_stats).rename(columns={ID: NF_ID, TIME: NF_TIME, "target": "y"})

    futr_cols = futr_exog_list()
    nf = NeuralForecast(
        models=[
            MLP(
                h=gap + span,
                input_size=48,
                max_steps=1,
                loss=MAE(),
                futr_exog_list=futr_cols,
                enable_progress_bar=False,
                logger=False,
                accelerator="cpu",
                devices=1,
            )
        ],
        freq="h",
    )
    nf.fit(long_df[[NF_ID, NF_TIME, "y", *futr_cols]])

    checkpoint = tmp_path / "checkpoint.pt"
    bundle.save(nf, fill_stats=fill_stats, cfg={"model": "MLP", "seed": 42}, out_path=checkpoint)

    # ONLY the requested block is supplied — the gap is covariate-absent, exactly as at test time.
    requested = pd.date_range(last_observed + pd.Timedelta(hours=gap + 1), periods=span, freq="h")
    supplied = cov_all[cov_all[TIME].isin(requested)]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    supplied.to_csv(input_dir / "test_input.csv", index=False)
    fi = _shuffled_index(requested, seed=gap)
    fi.to_csv(input_dir / "forecast_index_test.csv", index=False)

    output_file = tmp_path / "out" / "predictions.csv"
    res = subprocess.run(
        [
            sys.executable,
            "predict.py",
            "--input_dir",
            str(input_dir),
            "--output_file",
            str(output_file),
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert f"gap {gap}h" in res.stdout, res.stdout
    # h is sized to reach the request, so the recursive arm must NOT engage here.
    assert "ROLLOUT" not in res.stdout, res.stdout

    got = pd.read_csv(output_file)
    assert list(got.columns) == ["series_id", "timestamp", "prediction"]
    assert len(got) == len(fi)
    assert (got[[ID, TIME]].astype(str).values == fi[[ID, TIME]].astype(str).values).all()
    assert np.isfinite(got["prediction"]).all()

    # THE LOAD-BEARING ASSERTION: the same hours, recomputed independently.
    future = nf.make_future_dataframe().rename(columns={NF_ID: ID, NF_TIME: TIME})
    future[TIME] = pd.to_datetime(future[TIME])
    want = nf.predict(futr_df=build_futr_df(supplied, future, fill_stats))
    if NF_ID not in want.columns:
        want = want.reset_index()
    want = want.rename(columns={NF_ID: ID, NF_TIME: TIME})
    want[TIME] = pd.to_datetime(want[TIME])

    check = got.copy()
    check[TIME] = pd.to_datetime(check[TIME])
    merged = check.merge(want[[ID, TIME, "MLP"]], on=[ID, TIME], how="left")
    assert merged["MLP"].notna().all(), "the CLI wrote hours the model never forecast"
    # The CLI applies the de-smoothing recalibration to its BLEND OUTPUT at every rung
    # (`predict.DESMOOTH_GAMMA`), so the raw model forecast is no longer what lands in the CSV.
    # Mirror that transform here rather than weaken the assertion: the property under test is that
    # the CLI forecasts THE RIGHT HOURS FROM THE RIGHT MODEL, and it still holds exactly.
    import predict as _p

    _anchor = merged.groupby(ID)["MLP"].transform("mean")
    _want = (_anchor + _p.DESMOOTH_GAMMA * (merged["MLP"] - _anchor)).clip(lower=0.0)
    np.testing.assert_allclose(merged["prediction"].to_numpy(), _want.to_numpy(), rtol=1e-5)


@pytest.mark.parametrize("gap", [20, 60])
def test_a_request_BEYOND_h_rolls_forward_instead_of_failing(tmp_path, gap: int) -> None:
    """The checkpoint's ``h`` is fixed at training time; a longer request used to be a hard raise.

    It must never be answered by forecasting the wrong hours (that is #32), but it need not be
    answered by producing nothing: the bonus's minimum condition is beating the naive baseline, and
    an absent CSV fails that outright. So ``predict.rollout_predict`` feeds the model its own
    forecasts back as history until the horizon covers the request.

    Unreachable on the graded run — train ends 4319, the private block is 4656-4991, so
    ``gap + span = 672`` exactly and the 4992-hour timeline has nothing after it. This is
    insurance, and its ACCURACY is deliberately not asserted: a recursive rollout of a
    direct multi-horizon model is unmeasured, so the contract is completeness, not quality.
    """
    pytest.importorskip("neuralforecast")

    import subprocess
    import sys
    from pathlib import Path

    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import MLP

    from src import bundle
    from src.data.features import futr_exog_list
    from src.data.impute import apply_fill

    repo_root = Path(__file__).resolve().parent.parent
    span, h = 24, 12  # h < gap + span, so the rollout is forced
    assert h < gap + span
    start = pd.Timestamp("2023-01-01")
    rng = np.random.default_rng(5)

    cov_all = _covariates(start, HISTORY + gap + span, seed=gap)
    last_observed = start + pd.Timedelta(hours=HISTORY - 1)
    hist = cov_all[cov_all[TIME] <= last_observed].copy()
    hist["target"] = 10 + rng.normal(0, 0.5, len(hist))
    fill_stats = fit_fill_stats(hist)
    long_df = apply_fill(hist, fill_stats).rename(columns={ID: NF_ID, TIME: NF_TIME, "target": "y"})

    futr_cols = futr_exog_list()
    nf = NeuralForecast(
        models=[
            MLP(
                h=h,
                input_size=48,
                max_steps=1,
                loss=MAE(),
                futr_exog_list=futr_cols,
                enable_progress_bar=False,
                logger=False,
                accelerator="cpu",
                devices=1,
            )
        ],
        freq="h",
    )
    nf.fit(long_df[[NF_ID, NF_TIME, "y", *futr_cols]])

    checkpoint = tmp_path / "checkpoint.pt"
    bundle.save(nf, fill_stats=fill_stats, cfg={"model": "MLP", "seed": 42}, out_path=checkpoint)

    requested = pd.date_range(last_observed + pd.Timedelta(hours=gap + 1), periods=span, freq="h")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    cov_all[cov_all[TIME].isin(requested)].to_csv(input_dir / "test_input.csv", index=False)
    fi = _shuffled_index(requested, seed=gap)
    fi.to_csv(input_dir / "forecast_index_test.csv", index=False)

    output_file = tmp_path / "out" / "predictions.csv"
    res = subprocess.run(
        [
            sys.executable,
            "predict.py",
            "--input_dir",
            str(input_dir),
            "--output_file",
            str(output_file),
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert "ROLLOUT" in res.stdout, res.stdout

    got = pd.read_csv(output_file)
    assert list(got.columns) == ["series_id", "timestamp", "prediction"]
    assert len(got) == len(fi)
    # Index completeness is the contract: every REQUESTED hour, in the requested order, finite.
    assert (got[[ID, TIME]].astype(str).values == fi[[ID, TIME]].astype(str).values).all()
    assert np.isfinite(got["prediction"]).all()
    assert (got["prediction"] >= 0).all()
