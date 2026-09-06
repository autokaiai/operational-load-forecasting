"""Fences for the two imputation surfaces (plan S3).

The tests that matter most here are not the ones checking a strategy computes what it says. They
are the four that check the machinery cannot silently do the WRONG thing:

* the incumbent path is bit-exact, so adopting a hook does not move a recorded number;
* the gap length is never assumed — the spec's "the timeframe might differ" is a test, not a note;
* the gap reconstruction cannot see past the cutoff, checked by perturbing what it must not read;
* a strategy that cannot honestly serve a surface is REFUSED rather than degraded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import gap_fill as gf
from src.data.features import ID, MISSING_SUFFIX, TIME
from src.data.impute import apply_fill, fit_fill_stats

SIGNALS = ["sig_a", "sig_b"]


def _panel(n_series: int = 4, n_hours: int = 24 * 21, seed: int = 0) -> pd.DataFrame:
    """An hourly panel with a weekly shape, a per-series level, one exact how-of-week col."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2023-01-02")  # a Monday, so hour-of-week 0 is Monday 00:00
    for s in range(n_series):
        ts = pd.date_range(start, periods=n_hours, freq="h")
        how = ts.dayofweek * 24 + ts.hour
        level = 10.0 * (s + 1)
        rows.append(
            pd.DataFrame(
                {
                    ID: f"unit_{s:03d}",
                    TIME: ts,
                    "sig_a": level
                    + np.sin(how / 168 * 2 * np.pi) * 5
                    + rng.normal(0, 0.2, n_hours),
                    "sig_b": rng.normal(5, 1, n_hours),
                    # Deterministic in (series, hour-of-week) — the workload_intensity shape.
                    "exact_col": level + how * 0.5,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _split(df: pd.DataFrame, cut: int, gap_len: int):
    hidx = df.groupby(ID).cumcount()
    return df[hidx < cut], df[(hidx >= cut) & (hidx < cut + gap_len)]


# --------------------------------------------------------------------------- registry contract


def test_unknown_strategy_is_rejected():
    with pytest.raises(KeyError, match="unknown fill strategy"):
        gf.get_strategy("no_such_thing", gf.GAP)


def test_interp_is_refused_on_the_gap_surface():
    """The load-bearing refusal: interp across 336 unanchored hours is silently a forward fill.

    A strategy that degrades instead of erroring reports a number for a method that never ran.
    """
    with pytest.raises(ValueError, match="does not support the 'gap' surface"):
        gf.get_strategy("interp", gf.GAP)
    assert "interp" in gf.available_strategies(gf.SCATTERED)
    assert "interp" not in gf.available_strategies(gf.GAP)


def test_ffill_decay_is_gap_only_and_snaive_is_registered():
    assert "ffill_decay" not in gf.available_strategies(gf.SCATTERED)
    assert "snaive168" in gf.available_strategies(gf.GAP)


def test_modifiers_parse_as_modifiers_not_strategies():
    assert gf.parse_strategy("how168+exact") == ("how168", {"exact"})
    assert gf.parse_strategy("median") == ("median", set())
    assert gf.parse_strategy("chronos2+exact+guard") == ("chronos2", {"exact", "guard"})
    gf.get_strategy("median+exact", gf.GAP)  # resolves to the median strategy


def test_an_unknown_modifier_is_rejected_rather_than_ignored():
    """A typo in a modifier must not silently run the bare strategy and report its number."""
    df = _panel()
    train, gap = _split(df, cut=24 * 14, gap_len=48)
    with pytest.raises(ValueError, match="unknown modifier"):
        gf.reconstruct_block(
            target=gap,
            history=train,
            cols=SIGNALS,
            fill_mask=pd.Series(True, index=gap.index),
            strategy="median+exakt",
            id_col=ID,
            time_col=TIME,
        )


def test_guard_hands_a_sparse_event_flag_back_to_the_median():
    """A forecaster asked for a column that is almost always 0 hallucinates; the median is right.

    Measured on the real data: Chronos-2 is -34.2% against the median on ``maintenance_known``.
    Sparsity is a TRAIN-side property, so this selection needs no knowledge of the gap.
    """
    df = _panel(n_hours=24 * 35)
    df["sparse_flag"] = 0.0
    df.loc[df.sample(frac=0.2, random_state=3).index, "sparse_flag"] = 1.0
    cols = ["sig_a", "sparse_flag"]
    train, gap = _split(df, cut=24 * 28, gap_len=48)

    common = {
        "target": gap,
        "history": train,
        "cols": cols,
        "fill_mask": pd.Series(True, index=gap.index),
        "id_col": ID,
        "time_col": TIME,
    }
    plain = gf.reconstruct_block(strategy="how168", **common)
    guarded = gf.reconstruct_block(strategy="how168+guard", **common)

    # The sparse column falls back to the per-series median (0); the dense one is untouched.
    assert (guarded["sparse_flag"] == 0.0).all()
    assert not (plain["sparse_flag"] == 0.0).all()
    np.testing.assert_allclose(plain["sig_a"].to_numpy(), guarded["sig_a"].to_numpy())


# --------------------------------------------------------------------------- behaviour preserved


def test_gap_median_is_bit_exact_with_the_previous_implementation():
    """Adopting the hook must not move a single recorded number.

    Reimplements the pre-S3 body (per-series train median, global fallback) and demands equality,
    because every figure in the project up to S2 was produced by it.
    """
    df = _panel()
    train, gap = _split(df, cut=24 * 14, gap_len=48)

    med = train.groupby(ID)[SIGNALS].median()
    gmed = train[SIGNALS].median()
    expected = pd.DataFrame(
        {c: gap[ID].map(med[c]).fillna(gmed[c]).to_numpy() for c in SIGNALS}, index=gap.index
    )

    got = gf.reconstruct_block(
        target=gap,
        history=train,
        cols=SIGNALS,
        fill_mask=pd.Series(True, index=gap.index),
        strategy="median",
        id_col=ID,
        time_col=TIME,
    )
    pd.testing.assert_frame_equal(got[SIGNALS], expected[SIGNALS], check_dtype=False)


def test_apply_fill_median_is_bit_exact_with_the_previous_implementation():
    df = _panel()
    df.loc[df.sample(frac=0.05, random_state=1).index, "sig_a"] = np.nan
    stats = fit_fill_stats(df, nan_cols=["sig_a"])

    base = apply_fill(df, stats, nan_cols=["sig_a"])
    same = apply_fill(df, stats, nan_cols=["sig_a"], strategy="median")
    pd.testing.assert_frame_equal(base, same)


# --------------------------------------------------------------------------- the gap is not 336


@pytest.mark.parametrize("gap_len", [1, 17, 200, 500])
def test_the_gap_length_is_never_assumed(gap_len):
    """The spec: "the timeframe might differ". A 336-shaped assumption must not exist anywhere.

    Reconstruction has to cover exactly the rows the caller marks — no more, no fewer — at any
    gap length, including one that is not a multiple of a day or a week.
    """
    df = _panel(n_hours=24 * 50)
    train, gap = _split(df, cut=24 * 21, gap_len=gap_len)
    assert len(gap) == gap_len * df[ID].nunique()

    for strategy in ("median", "how168", "hod24", "ffill_decay", "snaive168"):
        got = gf.reconstruct_block(
            target=gap,
            history=train,
            cols=SIGNALS,
            fill_mask=pd.Series(True, index=gap.index),
            strategy=strategy,
            id_col=ID,
            time_col=TIME,
        )
        assert len(got) == len(gap), strategy
        assert got[SIGNALS].notna().all().all(), f"{strategy} left a hole at gap_len={gap_len}"


def test_only_the_masked_rows_are_reconstructed():
    """A partial mask must leave every other row alone — the scored block keeps REAL covariates."""
    df = _panel()
    train, horizon = _split(df, cut=24 * 14, gap_len=96)
    fill_mask = horizon.groupby(ID).cumcount() < 24  # only the first 24h are "absent"

    from src.models.chronos2_eval import _withhold_gap_covariates

    fut = horizon.rename(columns={ID: "unique_id", TIME: "ds"}).copy()
    train_nf = train.rename(columns={ID: "unique_id", TIME: "ds"})

    import src.models.chronos2_eval as ce

    original = ce.KNOWN_FUTURE_SIGNALS
    ce.KNOWN_FUTURE_SIGNALS = SIGNALS
    try:
        out = _withhold_gap_covariates(fut, train_nf, gap_len=24)
    finally:
        ce.KNOWN_FUTURE_SIGNALS = original

    kept = ~fill_mask.to_numpy()
    np.testing.assert_allclose(out.loc[kept, SIGNALS].to_numpy(), fut.loc[kept, SIGNALS].to_numpy())
    assert not np.allclose(
        out.loc[~kept, SIGNALS].to_numpy(), fut.loc[~kept, SIGNALS].to_numpy()
    ), "the gap rows should have been overwritten"


def test_infer_gap_len_derives_the_gap_and_handles_the_contiguous_case():
    last_obs = pd.Timestamp("2023-06-29 23:00")
    assert gf.infer_gap_len(last_obs, pd.Timestamp("2023-07-14 00:00")) == 336
    assert gf.infer_gap_len(last_obs, pd.Timestamp("2023-06-30 00:00")) == 0  # validation
    assert gf.infer_gap_len(last_obs, pd.Timestamp("2023-06-30 17:00")) == 17
    with pytest.raises(ValueError, match="not after"):
        gf.infer_gap_len(last_obs, pd.Timestamp("2023-06-29 00:00"))


# --------------------------------------------------------------------------- honesty


def test_the_gap_reconstruction_cannot_see_past_the_cutoff():
    """Perturb what it must not read, and demand the answer does not move.

    This is the property the whole surface rests on: at test time nothing after the cutoff exists.
    Checking it by construction is weaker than checking it by perturbation, so we perturb.
    """
    df = _panel()
    train, horizon = _split(df, cut=24 * 14, gap_len=96)
    fill_mask = pd.Series(horizon.groupby(ID).cumcount().to_numpy() < 24, index=horizon.index)

    poisoned = horizon.copy()
    poisoned.loc[~fill_mask, SIGNALS] = 1e6  # absurd values in the scored block

    common = {
        "history": train,
        "cols": SIGNALS,
        "fill_mask": fill_mask,
        "id_col": ID,
        "time_col": TIME,
    }
    for strategy in ("median", "how168", "hod24", "ffill_decay", "snaive168"):
        clean = gf.reconstruct_block(target=horizon, strategy=strategy, **common)
        dirty = gf.reconstruct_block(target=poisoned, strategy=strategy, **common)
        np.testing.assert_allclose(
            clean.loc[fill_mask, SIGNALS].to_numpy(),
            dirty.loc[fill_mask, SIGNALS].to_numpy(),
            err_msg=f"{strategy} read a value from after the cutoff",
        )


# --------------------------------------------------------------------------- the +exact modifier


def test_exact_modifier_reconstructs_a_deterministic_covariate_exactly():
    """Finding 1: a covariate that is an exact function of (series, hour-of-week) is knowable.

    Median-filling it is a defect, not a tuning choice, so the fix is adopted whatever the WAPE
    gates say — and this is what proves it is a fix.
    """
    # Five weeks of history so every (series, hour-of-week) bin clears EXACT_MIN_OBS. A two-week
    # train slice gives 2 observations per bin, which the detector correctly REFUSES to call
    # deterministic — the real slices are 17+ weeks.
    df = _panel(n_hours=24 * 35)
    cols = [*SIGNALS, "exact_col"]
    train, gap = _split(df, cut=24 * 28, gap_len=48)
    common = {
        "target": gap,
        "history": train,
        "cols": cols,
        "fill_mask": pd.Series(True, index=gap.index),
        "id_col": ID,
        "time_col": TIME,
    }
    plain = gf.reconstruct_block(strategy="median", **common)
    fixed = gf.reconstruct_block(strategy="median+exact", **common)

    truth = gap["exact_col"].to_numpy()
    assert np.abs(plain["exact_col"].to_numpy() - truth).max() > 1.0, "the defect should be visible"
    np.testing.assert_allclose(fixed["exact_col"].to_numpy(), truth, atol=1e-9)
    # A genuinely stochastic covariate is untouched by the modifier.
    np.testing.assert_allclose(plain["sig_b"].to_numpy(), fixed["sig_b"].to_numpy(), atol=1e-9)


def test_exact_detection_needs_enough_observations_to_mean_it():
    """A thin slice must not manufacture a false positive: 1 observation per bin is not evidence."""
    df = _panel(n_series=2, n_hours=168)  # exactly one week -> one row per (series, hour-of-week)
    found = gf.exact_how_columns(df[["exact_col"]], df[ID], df[TIME])
    assert found == [], "one observation per bin is trivially constant, not deterministic"

    wide = _panel(n_series=2, n_hours=168 * 4)  # four weeks -> four observations per bin
    assert "exact_col" in gf.exact_how_columns(wide[["exact_col"]], wide[ID], wide[TIME])


# --------------------------------------------------------------------------- scattered surface


def test_interp_beats_median_on_isolated_holes_and_leaves_no_nan():
    """The measured regime: isolated single hours with observed neighbours on both sides."""
    df = _panel()
    truth = df["sig_a"].copy()
    holes = df.groupby(ID).cumcount() % 37 == 11  # isolated, never adjacent
    work = df.copy()
    work.loc[holes, "sig_a"] = np.nan
    stats = fit_fill_stats(work, nan_cols=["sig_a"])

    med = apply_fill(work, stats, nan_cols=["sig_a"], strategy="median")
    itp = apply_fill(work, stats, nan_cols=["sig_a"], strategy="interp")

    assert not itp["sig_a"].isna().any()
    err_med = float((med.loc[holes, "sig_a"] - truth[holes]).abs().sum())
    err_itp = float((itp.loc[holes, "sig_a"] - truth[holes]).abs().sum())
    assert err_itp < err_med, f"interp {err_itp:.3f} should beat median {err_med:.3f}"


def test_the_missing_indicator_still_marks_every_reconstructed_row():
    """Whatever filled it, `*_missing` keeps meaning "this row was reconstructed"."""
    df = _panel()
    holes = df.groupby(ID).cumcount() % 37 == 11
    work = df.copy()
    work.loc[holes, "sig_a"] = np.nan
    stats = fit_fill_stats(work, nan_cols=["sig_a"])

    for strategy in ("median", "interp", "ffill", "how168"):
        out = apply_fill(work, stats, nan_cols=["sig_a"], strategy=strategy)
        flag = out[f"sig_a{MISSING_SUFFIX}"].to_numpy()
        np.testing.assert_array_equal(flag.astype(bool), holes.to_numpy(), strict=False)
        assert not out["sig_a"].isna().any(), strategy


def test_edge_nans_fall_through_to_the_stored_stats():
    """interp refuses to extrapolate, so the train-fitted table backstops the edges.

    That split is what makes a local strategy shippable: no data is baked into the checkpoint,
    and inference still imputes deterministically.
    """
    df = _panel(n_series=2, n_hours=72)
    work = df.copy()
    first = work.groupby(ID).head(1).index
    work.loc[first, "sig_a"] = np.nan  # leading NaN: nothing on the left to interpolate from
    stats = fit_fill_stats(df, nan_cols=["sig_a"])

    out = apply_fill(work, stats, nan_cols=["sig_a"], strategy="interp")
    assert not out["sig_a"].isna().any()
    for idx in first:
        sid = work.loc[idx, ID]
        assert out.loc[idx, "sig_a"] == pytest.approx(stats["sig_a"][str(sid)])


# --------------------------------------------------------------------------- #48's regression fence


def test_withholding_actually_withholds_and_flags_the_whole_gap():
    """Issue #48's third deliverable, as a FENCE rather than a one-off table.

    3.5 answered "does `_withhold_gap_covariates` really withhold?" with a measured `max|dpred|`
    table — evidence, which is not the same thing as a guard. Evidence goes stale the moment the
    code moves; this fails the build instead. It is the same distinction `cascade_provenance` was
    written to enforce on the other covariate surface, and it matters for the same reason: a
    covariate that leaks scores BETTER, so an unchecked one fails in the flattering direction.

    Asserts against the ADOPTED strategy, which after S3's dose-response is still ``median`` — the
    gap arm was a null and the incumbent stayed. If that ever changes, this test is where the
    change has to be made deliberately.
    """
    import src.models.chronos2_eval as ce
    from src.data.gap_fill import reconstruct_block
    from src.models.chronos2_eval import _withhold_gap_covariates

    df = _panel(n_hours=24 * 30)
    gap_len = 72
    train, horizon = _split(df, cut=24 * 21, gap_len=gap_len * 2)
    fut = horizon.rename(columns={ID: "unique_id", TIME: "ds"}).copy()
    train_nf = train.rename(columns={ID: "unique_id", TIME: "ds"})
    flag = f"sig_a{MISSING_SUFFIX}"
    fut[flag] = 0.0

    # The flag list is derived from NAN_COLS, which these synthetic columns are not in, so it is
    # patched alongside the signal list. Patching only one would test the values and silently skip
    # the flags — which is half of what #48 asked for.
    original, original_flags = ce.KNOWN_FUTURE_SIGNALS, ce.missing_indicator_cols
    ce.KNOWN_FUTURE_SIGNALS = SIGNALS
    ce.missing_indicator_cols = lambda: [flag]
    try:
        out = _withhold_gap_covariates(fut, train_nf, gap_len=gap_len)
    finally:
        ce.KNOWN_FUTURE_SIGNALS = original
        ce.missing_indicator_cols = original_flags

    # `_withhold_gap_covariates` sorts, so compare on ITS ordering rather than the input's --
    # otherwise a positional mask silently addresses different rows in the two frames.
    fut_sorted = fut.sort_values(["unique_id", "ds"])
    out = out.loc[fut_sorted.index]
    gap = pd.Series(
        (fut_sorted.groupby("unique_id").cumcount() < gap_len).to_numpy(), index=fut_sorted.index
    )
    expected = reconstruct_block(
        target=fut_sorted,
        history=train_nf,
        cols=SIGNALS,
        fill_mask=gap,
        strategy="median",  # THE ADOPTED STRATEGY — S3's gap arm was a null, the incumbent stayed
        id_col="unique_id",
        time_col="ds",
    )

    g = gap.to_numpy()
    # 1. every gap row carries the imputed value, not the real one
    np.testing.assert_allclose(out.loc[g, SIGNALS].to_numpy(), expected.loc[g, SIGNALS].to_numpy())
    # 2. and it is genuinely DIFFERENT from what was there — a no-op withhold would pass (1) too
    assert not np.allclose(out.loc[g, SIGNALS].to_numpy(), fut_sorted.loc[g, SIGNALS].to_numpy()), (
        "withholding left the real covariates in place"
    )
    # 3. the whole gap is flagged, and nothing beyond it is
    np.testing.assert_array_equal(out[flag].to_numpy() == 1.0, g)


def test_the_adopted_gap_strategy_is_still_the_incumbent():
    """A tripwire on the DEFAULT, so the adopted strategy cannot drift silently.

    S3's dose-response measured six arms off identical weights and not one beat the flat median on
    `tft_cascade` — the ceiling arm (`real`, genuine covariates) came in 0.00064 BELOW it. So the
    incumbent stands, and it stands on evidence rather than inertia. Changing this default is a
    decision someone has to make here, in the open.
    """
    import inspect

    from src.models.chronos2_eval import _withhold_gap_covariates
    from src.models.members import RunContext

    assert inspect.signature(_withhold_gap_covariates).parameters["strategy"].default == "median"
    assert RunContext(long_df=pd.DataFrame(), cut_idx=0).gap_fill == "median"
