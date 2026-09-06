"""S8's panel: the split, the rolled covariate, and the two arms that produce a graded number.

The tests that matter here are the ones that would have failed on the previous implementation:
the covariate must be a *forecast* rather than the lagged target, and the LightGBM roll must be
vectorised without changing its arithmetic. Everything runs on the synthetic fixture with the
dry-run backend, so the whole file is CPU-seconds and needs no Chronos weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.m5_loader import load_m5_long
from src.models.addl_panel import (
    CHRONOS_COL,
    CHRONOS_MISSING,
    NF_ID,
    NF_TARGET,
    NF_TIME,
    attach_covariate,
    blend,
    gapped_split,
    panel_metrics,
    roll_chronos_covariate,
    run_panel,
    score_chronos2,
    score_lgbm,
)


@pytest.fixture(scope="module")
def bundle():
    return load_m5_long(synthetic=True)


@pytest.fixture(scope="module")
def split(bundle):
    return gapped_split(bundle.long_df, bundle.horizon)


# ---------------------------------------------------------------------------- the split


def test_split_is_train_gap_score(bundle, split):
    """train | gap(h) | score(h): the scored block is the tail, the gap is unobserved."""
    assert split.full_h == 2 * split.h
    per = bundle.long_df.groupby(NF_ID).size()
    for uid, total in per.items():
        n_train = int((split.train[NF_ID] == uid).sum())
        n_block = int((split.horizon_block[NF_ID] == uid).sum())
        assert n_train + n_block == total
        assert n_block == split.full_h
    # The scored labels are the final h steps, and none of them is in train.
    labels = split.score_labels
    assert len(labels) == bundle.long_df[NF_ID].nunique() * split.h
    overlap = labels.merge(split.train[[NF_ID, NF_TIME]], on=[NF_ID, NF_TIME])
    assert overlap.empty


def test_train_ends_one_full_horizon_before_the_scored_block(split):
    """The freshest label sits +h before the first scored step — our private-test offset."""
    last_train = split.train.groupby(NF_ID)[NF_TIME].max()
    first_scored = split.score_labels.groupby(NF_ID)[NF_TIME].min()
    gap_steps = (first_scored - last_train).dt.days  # synthetic fixture is daily
    assert set(gap_steps.unique()) == {split.h + 1}


# ------------------------------------------------------------------- the rolled covariate


def test_rolled_covariate_never_sees_its_own_forecast_window(split):
    """Every block's context ends strictly before every step it forecasts."""
    cov, meta = roll_chronos_covariate(None, split.train, [], block=split.h, dry_run=True)
    assert meta["blocks"], "expected at least one rolled block"
    for start, length in meta["blocks"]:
        assert length == split.h
        assert start >= 1, "a block at step 0 would have no context at all"
    starts = [s for s, _ in meta["blocks"]]
    assert starts == sorted(starts, reverse=True), "grid must roll backward from the train end"
    # The blocks tile contiguously back from the train end, leaving only the warm-up.
    n = meta["n_steps"]
    assert starts[0] == n - split.h
    assert meta["warmup_steps"] == n - len(starts) * split.h
    assert 0 <= meta["warmup_steps"] < split.h


def test_covariate_is_a_forecast_not_the_lagged_target(split):
    """THE REGRESSION TEST. The old panel used ``y.shift(h)`` — a noiseless oracle.

    A rolled forecast must differ from the realised target it is predicting; if it ever equals
    it, the arm is training on the answer and its lift is meaningless (and S8's contamination
    detector, which compares that lift against our clean +8.3%, is reading a different quantity).
    """
    cov, _ = roll_chronos_covariate(None, split.train, [], block=split.h, dry_run=True)
    joined = split.train[[NF_ID, NF_TIME, NF_TARGET]].merge(cov, on=[NF_ID, NF_TIME], how="inner")
    assert len(joined) > 0
    assert not np.allclose(joined[NF_TARGET], joined[CHRONOS_COL]), "covariate equals the target"
    # And it must not be the target shifted by h either.
    shifted = split.train.copy()
    shifted["_lag"] = shifted.groupby(NF_ID)[NF_TARGET].shift(split.h)
    chk = shifted[[NF_ID, NF_TIME, "_lag"]].merge(cov, on=[NF_ID, NF_TIME]).dropna()
    assert not np.allclose(chk["_lag"], chk[CHRONOS_COL]), "covariate is the lagged target"


def test_attach_covariate_flags_exactly_the_warmup(split):
    """The warm-up is filled AND flagged — the model is told, not quietly given a number."""
    cov, meta = roll_chronos_covariate(None, split.train, [], block=split.h, dry_run=True)
    attached = attach_covariate(split.train, cov)
    assert attached[CHRONOS_COL].notna().all(), "a NaN futr covariate propagates to a NaN loss"
    per_series_flagged = attached.groupby(NF_ID)[CHRONOS_MISSING].sum()
    assert set(per_series_flagged.unique()) == {float(meta["warmup_steps"])}
    assert len(attached) == len(split.train)


def test_capping_blocks_is_smoke_only_and_grows_the_warmup(split):
    cov, meta = roll_chronos_covariate(
        None, split.train, [], block=split.h, dry_run=True, max_blocks=2
    )
    assert len(meta["blocks"]) == 2
    assert meta["capped"] is True
    attached = attach_covariate(split.train, cov)
    flagged = attached.groupby(NF_ID)[CHRONOS_MISSING].sum().unique()
    assert flagged[0] == meta["n_steps"] - 2 * split.h


# ------------------------------------------------------------------------------ the arms


def _naive_lgbm_roll(bundle, split, model, feat_cols, lags, futr):
    """Row-by-row reference for the vectorised roll — the implementation it replaced."""
    hist = {u: g[NF_TARGET].tolist() for u, g in split.train.groupby(NF_ID, sort=True)}
    rows = []
    for uid, g in split.horizon_block.sort_values([NF_ID, NF_TIME]).groupby(NF_ID, sort=True):
        series = hist[uid][:]
        for _, row in g.iterrows():
            n = len(series)
            f = {f"lag_{lag}": series[n - lag] for lag in lags}
            f["roll_mean_7"] = float(np.mean(series[-7:]))
            ts = pd.to_datetime(row[NF_TIME])
            f["dow"], f["hour"] = ts.dayofweek, ts.hour
            for c in futr:
                f[c] = float(row[c])
            yhat = float(model.predict(pd.DataFrame([f])[feat_cols])[0])
            series.append(yhat)
            rows.append({NF_ID: uid, NF_TIME: row[NF_TIME], "lgbm": yhat})
    return pd.DataFrame(rows)


def test_vectorised_lgbm_roll_matches_the_row_by_row_reference(bundle, split):
    """Vectorising over SERIES must not change the arithmetic — only the number of predict calls.

    A recursive roll cannot be vectorised over time (step t+1 needs step t's prediction), but
    every series advances through the same step together, so the series axis vectorises exactly.
    """
    import lightgbm as lgb

    preds, metrics = score_lgbm(bundle, split, n_estimators=40, seed=42)

    # Rebuild the identical booster, then roll it the slow way.
    lags = (1, 2, 7, 14, split.h)
    futr = [c for c in bundle.futr_exog if c in split.train.columns]
    d = split.train.sort_values([NF_ID, NF_TIME]).copy()
    g = d.groupby(NF_ID, sort=False)[NF_TARGET]
    for lag in lags:
        d[f"lag_{lag}"] = g.shift(lag)
    d["roll_mean_7"] = g.shift(1).rolling(7).mean().reset_index(0, drop=True)
    ds = pd.to_datetime(d[NF_TIME])
    d["dow"], d["hour"] = ds.dt.dayofweek, ds.dt.hour
    tr = d.dropna(subset=[f"lag_{lag}" for lag in lags] + ["roll_mean_7"])
    feat_cols = [*(f"lag_{lag}" for lag in lags), "roll_mean_7", "dow", "hour", *futr]
    model = lgb.LGBMRegressor(
        n_estimators=40, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=42
    )
    model.fit(tr[feat_cols], tr[NF_TARGET])

    ref = _naive_lgbm_roll(bundle, split, model, feat_cols, lags, futr)
    merged = preds.merge(ref, on=[NF_ID, NF_TIME], suffixes=("", "_ref"))
    assert len(merged) == len(preds) == bundle.long_df[NF_ID].nunique() * split.full_h
    np.testing.assert_allclose(merged["lgbm"], merged["lgbm_ref"], rtol=1e-9, atol=1e-9)
    assert 0.0 <= metrics["wape"] < 10.0


def test_chronos2_arm_scores_the_horizon_block(bundle, split):
    from src.models.addl_panel import forecast_block

    cov = forecast_block(
        None, split.train, split.horizon_block, [], split.full_h, dry_run=True, season=7
    )
    preds, metrics = score_chronos2(bundle, split, cov)
    assert len(preds) == bundle.long_df[NF_ID].nunique() * split.full_h
    assert int(preds["is_scored"].sum()) == bundle.long_df[NF_ID].nunique() * split.h
    assert np.isfinite(metrics["wape"])


def test_blend_uses_the_shipped_weight_and_refuses_a_partial_join(bundle, split):
    preds, _ = score_lgbm(bundle, split, n_estimators=20)
    fake_cascade = preds.rename(columns={"lgbm": "cascade"}).copy()
    fake_cascade["cascade"] *= 1.10
    frames = {"lgbm": preds, "cascade": fake_cascade}
    m, metrics = blend(frames, 0.24)
    expected = 0.24 * preds["lgbm"].to_numpy() + 0.76 * fake_cascade["cascade"].to_numpy()
    np.testing.assert_allclose(m["blend"], np.clip(expected, 0, None), rtol=1e-12)
    assert np.isfinite(metrics["wape"])

    frames["cascade"] = fake_cascade.iloc[:-5]
    with pytest.raises(ValueError, match="row count"):
        blend(frames, 0.24)


def test_arms_return_the_FULL_block_with_only_its_tail_scored(bundle, split):
    """Keeping the gap half is what makes the honest weight fit free."""
    preds, _ = score_lgbm(bundle, split, n_estimators=20)
    n_series = bundle.long_df[NF_ID].nunique()
    assert len(preds) == n_series * split.full_h
    assert int(preds["is_scored"].sum()) == n_series * split.h
    # The scored half is strictly LATER than the fitting half, per series.
    for _, g in preds.groupby(NF_ID):
        assert g[g["is_scored"]][NF_TIME].min() > g[~g["is_scored"]][NF_TIME].max()


def test_refit_weight_is_fitted_off_the_scored_rows(bundle, split):
    """THE LEAK GUARD. The weight must be chosen without seeing the rows it is judged on."""
    from src.models.addl_panel import fit_blend_weight

    preds, _ = score_lgbm(bundle, split, n_estimators=20)
    cascade = preds.rename(columns={"lgbm": "cascade"}).copy()
    cascade["cascade"] *= 1.30
    frames = {"lgbm": preds, "cascade": cascade}

    out = fit_blend_weight(frames)
    assert 0.0 <= out["weight"] <= 1.0
    assert out["fit_rows"] == bundle.long_df[NF_ID].nunique() * (split.full_h - split.h)
    assert "gap half" in out["fitted_on"]

    # It must equal the grid optimum computed on the GAP HALF alone...
    m = frames["cascade"][[NF_ID, NF_TIME, NF_TARGET, "is_scored", "cascade"]].merge(
        frames["lgbm"][[NF_ID, NF_TIME, "lgbm"]], on=[NF_ID, NF_TIME]
    )
    fit = m[~m["is_scored"]]
    grid = np.arange(0.0, 1.005, 0.01)
    err = [
        np.abs(fit[NF_TARGET] - np.clip(w * fit["lgbm"] + (1 - w) * fit["cascade"], 0, None)).sum()
        for w in grid
    ]
    assert out["weight"] == pytest.approx(float(grid[int(np.argmin(err))]), abs=1e-9)

    # ...and NOT the optimum on the scored half, or it would be fitting on its own answer.
    scored = m[m["is_scored"]]
    err_s = [
        np.abs(
            scored[NF_TARGET] - np.clip(w * scored["lgbm"] + (1 - w) * scored["cascade"], 0, None)
        ).sum()
        for w in grid
    ]
    w_oracle = float(grid[int(np.argmin(err_s))])
    assert (
        out["metrics"]["wape"]
        >= panel_metrics(
            scored[NF_TARGET],
            np.clip(w_oracle * scored["lgbm"] + (1 - w_oracle) * scored["cascade"], 0, None),
        )["wape"]
    ), "an honest fit can never beat the oracle it was kept away from"


def test_blend_is_absent_when_an_arm_is(bundle, split):
    preds, _ = score_lgbm(bundle, split, n_estimators=20)
    assert blend({"lgbm": preds}, 0.24) == (None, None)


# ----------------------------------------------------------------------------- the driver


def test_run_panel_end_to_end_dry_run(bundle):
    """The whole driver on the fixture: arms score, the blend forms, the detector reports."""
    res = run_panel(bundle, arms=("chronos2", "lgbm"), dry_run=True, seed=42)
    assert res["synthetic"] is True
    assert res["split"]["gap"] == res["split"]["score_len"]
    assert res["split"]["forecast_len"] == 2 * res["split"]["gap"]
    for arm in ("chronos2", "lgbm"):
        assert res["models"][arm]["status"] == "ok", res["models"][arm]
        assert np.isfinite(res["models"][arm]["metrics"]["wape"])
    assert "blend" not in res["models"], "no cascade arm ran, so no blend is possible"
    assert res["contamination_check"]["verdict"] == "not evaluable (an arm failed)"
    assert res["covariate"]["blocks"]


def test_contamination_check_reads_the_lift_against_our_clean_reference():
    from src.models.addl_panel import CLEAN_CASCADE_LIFT, _contamination_check

    ok = {
        "tft": {"status": "ok", "metrics": {"wape": 0.20}},
        "cascade": {"status": "ok", "metrics": {"wape": 0.20 * (1 - CLEAN_CASCADE_LIFT)}},
    }
    out = _contamination_check(ok)
    assert out["cascade_lift_vs_tft"] == pytest.approx(CLEAN_CASCADE_LIFT, abs=1e-4)
    assert out["verdict"] == "consistent with the clean reference"

    suspicious = {
        "tft": {"status": "ok", "metrics": {"wape": 0.20}},
        "cascade": {"status": "ok", "metrics": {"wape": 0.10}},  # +50% lift
    }
    assert "SUSPICIOUS" in _contamination_check(suspicious)["verdict"]


def test_seasonal_mean_uses_the_right_phase_and_beats_seasonal_naive(bundle, split):
    """The seasonal-mean baseline must align on phase, and averaging should beat one draw.

    The horizon block's step index restarts at 0, so the phase has to be offset by the TRAIN
    length — get that wrong and the baseline is a shuffled version of itself, which still looks
    plausible and still scores something.
    """
    from src.models.addl_panel import score_seasonal_mean, score_seasonal_naive

    sm, m_mean = score_seasonal_mean(bundle, split)
    sn, m_naive = score_seasonal_naive(bundle, split)
    assert len(sm) == len(sn) == bundle.long_df[NF_ID].nunique() * split.full_h
    assert sm["seasonal_mean"].notna().all()
    # The fixture has a clean weekly season, so averaging ~50 periods must beat tiling the last.
    assert m_mean["wape"] < m_naive["wape"]

    # Phase check: shifting the baseline by one step must make it worse.
    shifted = sm[sm["is_scored"]].copy()
    shifted["seasonal_mean"] = np.roll(shifted["seasonal_mean"].to_numpy(), 1)
    assert panel_metrics(shifted[NF_TARGET], shifted["seasonal_mean"])["wape"] > m_mean["wape"]


def test_skill_uses_seasonal_naive_because_its_definition_is_corpus_independent():
    from src.models.addl_panel import OUR_REFERENCE, SKILL_BASELINE, _skill_table

    assert SKILL_BASELINE == "seasonal_naive"
    models = {
        "seasonal_naive": {"status": "ok", "metrics": {"wape": 1.0}},
        "cascade": {"status": "ok", "metrics": {"wape": 0.25}},
    }
    out = _skill_table(models)
    assert out["baseline"] == "seasonal_naive"
    assert out["arms"]["cascade"] == 0.75
    # Our own anchors must be read against the SAME baseline, not a different lag.
    assert out["our_reference_skill"]["cascade"] == pytest.approx(
        1 - OUR_REFERENCE["cascade"] / OUR_REFERENCE["seasonal_naive"], abs=1e-4
    )


def test_the_panel_builds_OUR_tft_not_the_library_default():
    """S8 asks whether OUR architecture transfers, so the arm must be our architecture.

    neuralforecast's TFT defaults are a different model — hidden_size 128 against our 64 and
    windows_batch_size 1024 against our 32. The second is also a memory ceiling: the 1024 default
    OOM'd an 8 GiB card on an 8-series fixture, the same failure that killed 11 of S5's 32 trials.
    """
    import yaml

    from src.models.addl_panel import SHIPPED_TFT, build_nf

    shipped = yaml.safe_load(open("configs/tft_chronos.yaml"))
    for key, value in SHIPPED_TFT.items():
        assert shipped[key] == value, f"{key}: panel {value!r} != shipped config {shipped[key]!r}"

    nf = build_nf("TFT", h=8, freq="D", futr_exog=["x"], stat_exog=[], max_steps=1)
    model = nf.models[0]
    assert model.hparams["hidden_size"] == 64
    assert model.hparams["grn_activation"] == "ELU"
    assert model.hparams["n_head"] == 4
    assert model.windows_batch_size == 32
    # LSTM has no GRN and no attention heads; those keys must be dropped, not passed.
    nf_lstm = build_nf("LSTM", h=8, freq="D", futr_exog=["x"], stat_exog=[], max_steps=1)
    assert nf_lstm.models[0].windows_batch_size == 32
