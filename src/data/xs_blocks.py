"""Cross-series aggregate block builders — the SINGLE implementation, in `src/` so the
SUBMISSION ARCHIVE can reach it.

MOVED HERE FROM the aggregate-screening script, and the reason is not tidiness. The submission
archive ships `src/` and `predict.py`; it does NOT ship `scripts/`. While these lived under
`scripts/`, `src/data/xs_attach.py` imported them through a re-export shim, which worked in
every CV run (the repo root is on the path) and failed inside the archive with
`ModuleNotFoundError: No module named 'scripts'` — discovered by the submission smoke, not by any
test. A feature the model was FITTED on could not be rebuilt at inference.

The screen imports these from here, so there is exactly one definition of every block and the
numbers on record and the shipped feature cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import KNOWN_FUTURE_SIGNALS, NAN_COLS
from src.data.loader import NF_ID, NF_TARGET
from src.eval.splits import HOUR_IDX, SCORE_LEN

THEIRS = [
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "workload_intensity",
    "shock_risk",
]
DEGENERATE = ["workload_intensity"]
LIVE = [c for c in THEIRS if c not in DEGENERATE]
# Candidates for the slots the degenerate columns free up.
REPLACEMENTS = ["demand_forecast", "staffing_forecast", "event_load_forecast"]


# --------------------------------------------------------------------------- helpers
def _zone_labels(df: pd.DataFrame) -> pd.Series:
    """Recover the zone partition from (zone_sin, zone_cos). Measured: 8 zones x 12 series."""
    z = df.groupby(NF_ID)[["zone_sin", "zone_cos"]].first().round(6)
    _, lab = np.unique(z.to_numpy(), axis=0, return_inverse=True)
    return pd.Series(lab, index=z.index, name="_zone")


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """[time x series] view of one column."""
    return df.pivot_table(index=HOUR_IDX, columns=NF_ID, values=col)


def _attach(df: pd.DataFrame, wide: pd.DataFrame, name: str) -> pd.DataFrame:
    """Merge a [time x series] block back onto the long frame as column `name`."""
    long = wide.stack().rename(name).reset_index()
    return df.merge(long, on=[HOUR_IDX, NF_ID], how="left")


# --------------------------------------------------------------------------- blocks
def block_A1(df, zones):
    """THEIR EXACT FEATURE SET -- the control. system mean + zone mean of all four, degenerate
    columns included, so this reproduces the published +3.3% rather than an improved variant."""
    out, names = df, []
    for c in THEIRS:
        w = _wide(df, c)
        sysm = pd.DataFrame(
            np.repeat(w.mean(axis=1).to_numpy()[:, None], w.shape[1], 1),
            index=w.index,
            columns=w.columns,
        )
        out = _attach(out, sysm, f"xs_sys_{c}")
        names.append(f"xs_sys_{c}")
        zm = w.T.groupby(zones).transform("mean").T
        out = _attach(out, zm, f"xs_zone_{c}")
        names.append(f"xs_zone_{c}")
    return out, names


def block_A2(df, zones):
    """DEGENERATE-FREE + REFILLED. Drop the two calendar re-encodings, spend the slots on three
    further covariates. Same feature count as theirs, strictly more panel information."""
    out, names = df, []
    for c in LIVE + REPLACEMENTS:
        w = _wide(df, c)
        sysm = pd.DataFrame(
            np.repeat(w.mean(axis=1).to_numpy()[:, None], w.shape[1], 1),
            index=w.index,
            columns=w.columns,
        )
        out = _attach(out, sysm, f"xs2_sys_{c}")
        names.append(f"xs2_sys_{c}")
        zm = w.T.groupby(zones).transform("mean").T
        out = _attach(out, zm, f"xs2_zone_{c}")
        names.append(f"xs2_zone_{c}")
    return out, names


def block_A3(df, zones):
    """LEAVE-ONE-OUT system mean. The plain system mean contains the unit's own value (1/96 of it),
    so the model must subtract itself out before the feature means 'what everyone ELSE is doing'."""
    out, names = df, []
    for c in LIVE:
        w = _wide(df, c)
        n = w.shape[1]
        loo = (w.sum(axis=1).to_numpy()[:, None] - w.to_numpy()) / (n - 1)
        out = _attach(out, pd.DataFrame(loo, index=w.index, columns=w.columns), f"xs_loo_{c}")
        names.append(f"xs_loo_{c}")
    return out, names


def block_A4(df, zones):
    """SHARES AND RANKS -- the transform Group 44 did not test.

    They tried DEVIATIONS (own - system, own - zone) and got 0.1414 -> 0.1415, nothing. Their
    diagnosis is right and does not generalise: a difference is a LINEAR combination of two columns
    the model already holds. A share and a within-hour rank are NONLINEAR and SCALE-FREE, which is
    what a volume-weighted pooled WAPE over heterogeneous units actually wants. They tested the one
    transform that provably adds nothing and stopped there."""
    out, names = df, []
    for c in LIVE:
        w = _wide(df, c)
        tot = w.sum(axis=1).to_numpy()[:, None]
        share = w.to_numpy() / np.where(np.abs(tot) < 1e-9, np.nan, tot)
        out = _attach(out, pd.DataFrame(share, index=w.index, columns=w.columns), f"xs_share_{c}")
        names.append(f"xs_share_{c}")
        rank = w.rank(axis=1, pct=True)
        out = _attach(out, rank, f"xs_rank_{c}")
        names.append(f"xs_rank_{c}")
    return out, names


def block_A5(df, zones):
    """CROSS-SECTIONAL DISPERSION. Every aggregate in their design is a MEAN. A mean cannot say
    'the system is unusually disagreeing right now', which is a regime signal in its own right."""
    out, names = df, []
    for c in LIVE:
        w = _wide(df, c)
        for stat, vals in (
            ("std", w.std(axis=1)),
            ("rng", w.quantile(0.9, axis=1) - w.quantile(0.1, axis=1)),
        ):
            b = pd.DataFrame(
                np.repeat(vals.to_numpy()[:, None], w.shape[1], 1), index=w.index, columns=w.columns
            )
            out = _attach(out, b, f"xs_{stat}_{c}")
            names.append(f"xs_{stat}_{c}")
    return out, names


def block_A6(df, zones):
    """MISSINGNESS REGIME. The fraction of units whose `*_missing` flag is set at hour t is a
    system-wide DATA-AVAILABILITY indicator, and nothing in their design touches missingness
    cross-sectionally at all. Also the one block that reaches the 'structure in the missingness
    pattern itself' gap this project has carried unaddressed."""
    out, names = df, []
    flags = [f"{c}_missing" for c in NAN_COLS if f"{c}_missing" in df.columns]
    for c in flags[:5]:
        w = _wide(df, c)
        frac = pd.DataFrame(
            np.repeat(w.mean(axis=1).to_numpy()[:, None], w.shape[1], 1),
            index=w.index,
            columns=w.columns,
        )
        out = _attach(out, frac, f"xs_miss_{c}")
        names.append(f"xs_miss_{c}")
    if flags:
        w = _wide(df, flags[0]).copy()
        anym = df.groupby([HOUR_IDX, NF_ID])[flags].max().max(axis=1).unstack()
        tot = pd.DataFrame(
            np.repeat(anym.mean(axis=1).to_numpy()[:, None], w.shape[1], 1),
            index=anym.index,
            columns=anym.columns,
        )
        out = _attach(out, tot, "xs_miss_any_frac")
        names.append("xs_miss_any_frac")
    return out, names


def block_A7(df, zones):
    """ROLLING AGGREGATES -- crossing their own two wins, which their log never does.

    Centred rolling covariate windows were worth 5.1% on their GBM; cross-series aggregates 3.2%.
    Rolling windows OF THE AGGREGATES -- how the system regime is MOVING, not merely where it is --
    appear nowhere in their log. Centred windows are legal for the same reason the aggregates are:
    every covariate is known across the whole horizon."""
    out, names = df, []
    for c in LIVE:
        w = _wide(df, c)
        sysm = w.mean(axis=1)
        for win in (3, 6, 12, 24):
            r = sysm.rolling(win, center=True, min_periods=1).mean()
            b = pd.DataFrame(
                np.repeat(r.to_numpy()[:, None], w.shape[1], 1), index=w.index, columns=w.columns
            )
            out = _attach(out, b, f"xs_roll{win}_{c}")
            names.append(f"xs_roll{win}_{c}")
    return out, names


def block_A8(df, zones, cut: int = 3648, k: int = 8):
    """LEARNED PARTITION. zone_sin/cos give a GIVEN grouping (8 x 12, and within-zone target corr
    0.583 vs cross-zone 0.535 -- real but a thin margin). The data may hold a better one. Cluster
    the series on their target correlation, FITTED STRICTLY ON _hidx < cut, and take group means."""
    out, names = df, []
    ytr = _wide(df[df[HOUR_IDX] < cut], NF_TARGET)
    corr = ytr.corr().fillna(0.0)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = np.clip(1.0 - corr.to_numpy(), 0, 2)
    np.fill_diagonal(d, 0.0)
    lab = fcluster(linkage(squareform(d, checks=False), "average"), k, criterion="maxclust")
    grp = pd.Series(lab, index=corr.index)
    for c in LIVE:
        w = _wide(df, c)
        gm = w.T.groupby(grp).transform("mean").T
        out = _attach(out, gm, f"xs_clu_{c}")
        names.append(f"xs_clu_{c}")
    return out, names


def block_A9(df, zones):
    """CAPACITY-WEIGHTED system mean. An equal-weighted mean is not the metric's view of 'the
    system': WAPE weights a series by sum|y|, and our per-series scale spread is 2.23x."""
    out, names = df, []
    cap = df.groupby(NF_ID)["nominal_capacity"].first()
    for c in LIVE:
        w = _wide(df, c)
        ws = cap.reindex(w.columns).to_numpy()[None, :]
        wm = (w.to_numpy() * ws).sum(axis=1, keepdims=True) / ws.sum()
        b = pd.DataFrame(np.repeat(wm, w.shape[1], 1), index=w.index, columns=w.columns)
        out = _attach(out, b, f"xs_cap_{c}")
        names.append(f"xs_cap_{c}")
    return out, names


def block_A10(df, zones):
    """ORIGIN-ANCHORED SYSTEM TARGET aggregates -- the only legitimate target channel for a tree.

    A system mean of y AT THE FORECAST HOUR is leakage. Shifted back so the value at hour t is the
    system mean of y over [t-168, t), it is known at any origin >= t and the design only ever reads
    it at the forecast hour... so it must be shifted by the full gapped horizon to stay honest.
    Conservative choice: shift by 672 (the whole gapped horizon), which is knowable at every origin
    this design uses."""
    out, names = df, []
    w = _wide(df, NF_TARGET)
    sysm = w.mean(axis=1)
    for win in (24, 168):
        r = sysm.rolling(win, min_periods=1).mean().shift(2 * SCORE_LEN)
        b = pd.DataFrame(
            np.repeat(r.to_numpy()[:, None], w.shape[1], 1), index=w.index, columns=w.columns
        )
        out = _attach(out, b, f"xs_ytgt{win}")
        names.append(f"xs_ytgt{win}")
    return out, names


def block_A11(df, zones):
    """LAGGED AND LEADING system aggregates -- does the system LEAD this unit?

    A contemporaneous mean cannot express propagation. Leads are legal here for the same reason the
    aggregates are: every covariate is supplied across the whole forecast window, so the system mean
    at t+24 uses nothing unavailable at prediction time. (A TARGET lead would not be legal; this
    block touches covariates only.)"""
    out, names = df, []
    for c in LIVE:
        w = _wide(df, c)
        sysm = w.mean(axis=1)
        for lag in (-24, 24, 168):
            s = sysm.shift(lag)
            tag = f"lead{-lag}" if lag < 0 else f"lag{lag}"
            b = pd.DataFrame(
                np.repeat(s.to_numpy()[:, None], w.shape[1], 1), index=w.index, columns=w.columns
            )
            out = _attach(out, b, f"xs_{tag}_{c}")
            names.append(f"xs_{tag}_{c}")
    return out, names


def block_A12(df, zones, cut: int = 3648, n_pc: int = 3):
    """PRINCIPAL COMPONENTS of the cross-sectional panel -- a LEARNED basis, not a hand-picked mean.

    The system mean is one particular projection of the 96-vector at each hour, chosen by hand. The
    first few PCs are the projections the data itself says carry the variance, and they remove the
    'which four covariates, and why the mean' choice entirely. FITTED STRICTLY ON _hidx < cut."""
    from sklearn.decomposition import PCA

    out, names = df, []
    for c in LIVE:
        w = _wide(df, c)
        tr = w[w.index < cut].to_numpy()
        mu = tr.mean(axis=0, keepdims=True)
        pca = PCA(n_components=n_pc).fit(tr - mu)
        scores = pca.transform(w.to_numpy() - mu)  # [T, n_pc]
        for j in range(n_pc):
            b = pd.DataFrame(
                np.repeat(scores[:, [j]], w.shape[1], 1), index=w.index, columns=w.columns
            )
            out = _attach(out, b, f"xs_pc{j}_{c}")
            names.append(f"xs_pc{j}_{c}")
    return out, names


def block_A13(df, zones):
    """CHRONOS CONSENSUS -- the block nobody else on this leaderboard can run.

    Chronos-2 is fed all 29 covariates but ONE SERIES AT A TIME, so it is exactly as
    cross-sectionally blind as our TFT. The system and zone mean of `chronos2_forecast` is a
    cross-sectional CONSENSUS OF A FOUNDATION MODEL'S FORECASTS -- a second-order feature that
    exists only because we already hold a per-series foundation forecast.

    Scored as a PAIRED sub-experiment against `chronos2_forecast` alone (see `--blocks A13`), or the
    arm would conflate 'adding Chronos' with 'adding its consensus'. Inherits the cascade's per-cut
    provenance fence: `src.models.cascade_provenance`."""
    out, names = df, []
    col = "chronos2_forecast"
    w = _wide(df, col)
    sysm = pd.DataFrame(
        np.repeat(w.mean(axis=1).to_numpy()[:, None], w.shape[1], 1),
        index=w.index,
        columns=w.columns,
    )
    out = _attach(out, sysm, "xs_sys_chronos")
    names.append("xs_sys_chronos")
    zm = w.T.groupby(zones).transform("mean").T
    out = _attach(out, zm, "xs_zone_chronos")
    names.append("xs_zone_chronos")
    rank = w.rank(axis=1, pct=True)
    out = _attach(out, rank, "xs_rank_chronos")
    names.append("xs_rank_chronos")
    return out, names


def block_A14(df, zones, cut: int = 3648):
    """TIME-AXIS AGGREGATION -- per-series WHOLE-HISTORY statistics as continuous statics.

    THE GAP THIS FILLS. Every other block in this screen aggregates ACROSS SERIES at a fixed hour.
    Nothing aggregates ACROSS TIME within a series. Neither did the reference approach: the
    statics are the three the dataset ships (`nominal_capacity`, `zone_sin`, `zone_cos`) and
    nothing about how a series BEHAVES -- its long-run level, its volatility, its seasonal
    strength, how often it goes missing. Fitted STRICTLY ON `_hidx < cut`, then broadcast as a
    constant column per series.

    WHY IT IS NOT TRIVIALLY REDUNDANT WITH THE UNIT CATEGORICAL. The tree already gets `unit` as a
    96-level categorical, so in principle it can memorise any per-series constant. But a categorical
    can only MEMORISE; a continuous static can be SPLIT ON and therefore POOLED -- one split at
    `cv > 0.3` applies to every volatile series at once, including behaviour the categorical would
    have to relearn per level from 1/96 of the rows. That pooling is the hypothesis.

    WHY THE TFT MAY NOT INHERIT A WIN HERE. The TFT carries a static embedding per series AND a
    per-series scaler that already divides out level and scale, so `mean` and `std` are close to
    redundant there in exactly the way Group 44's deviation features were (0.1414 -> 0.1415, a
    linear combination of columns already present). The shape terms (seasonal strength, missingness
    rate) are the ones with a route to the TFT. Screen it here; do not assume transfer."""
    out, names = df, []
    tr = df[df[HOUR_IDX] < cut]
    w = _wide(tr, NF_TARGET)
    cap = df.groupby(NF_ID)["nominal_capacity"].first()

    mu = w.mean()
    sd = w.std()
    stats = {
        "st_level": mu,
        "st_std": sd,
        "st_cv": sd / mu.replace(0.0, np.nan),
        "st_iqr": (w.quantile(0.75) - w.quantile(0.25)) / mu.replace(0.0, np.nan),
        # Volatility at the two horizons the design cares about, scale-free.
        "st_d1": w.diff(1).abs().mean() / mu.replace(0.0, np.nan),
        "st_d168": w.diff(168).abs().mean() / mu.replace(0.0, np.nan),
        # Seasonal strength: how much of the variance the weekly profile explains.
        "st_seas168": 1.0 - (w.diff(168).var() / w.var().replace(0.0, np.nan)),
        "st_seas24": 1.0 - (w.diff(24).var() / w.var().replace(0.0, np.nan)),
        "st_util": mu / cap.reindex(mu.index).replace(0.0, np.nan),
    }
    flags = [f"{c}_missing" for c in NAN_COLS if f"{c}_missing" in df.columns]
    if flags:
        stats["st_missrate"] = tr.groupby(NF_ID)[flags].mean().mean(axis=1)

    full = _wide(df, NF_TARGET)
    for name, vals in stats.items():
        v = vals.reindex(w.columns).astype(float)
        b = pd.DataFrame(
            np.repeat(v.to_numpy()[None, :], len(full), 0), index=full.index, columns=w.columns
        )
        out = _attach(out, b, f"xs_{name}")
        names.append(f"xs_{name}")
    return out, names


def block_A15(df, zones):
    """HIERARCHICAL: a SEASONAL-NAIVE FORECAST OF THE AGGREGATE, not a smoothed history of it.

    A10 already carries origin-anchored ROLLING MEANS of the system target (24h, 168h, shifted 672),
    and a rolling mean deliberately smooths the diurnal and weekly shape away. This block keeps the
    shape: the system mean of `y` at t-672, which -- 672 being 4 x 168 -- is the SAME HOUR OF WEEK,
    and for every forecast hour in the scored block lands at or before the cut. Same fence as A10,
    different information.

    THE HIERARCHICAL ARGUMENT. Summing 96 series cancels idiosyncratic noise, so the aggregate is a
    far more predictable object than any single unit; a cheap forecast of the total is therefore a
    high-quality statement about where each unit is headed. Zone totals sit one level down the same
    hierarchy. The week-over-week delta of the aggregate says whether the system as a whole is
    trending, which no contemporaneous covariate mean can express."""
    out, names = df, []
    shift = 2 * SCORE_LEN  # 672 = 4 x 168: legal AND phase-aligned
    w = _wide(df, NF_TARGET)
    sysm = w.mean(axis=1)

    sn = sysm.shift(shift)
    b = pd.DataFrame(
        np.repeat(sn.to_numpy()[:, None], w.shape[1], 1), index=w.index, columns=w.columns
    )
    out = _attach(out, b, "xs_hier_sys")
    names.append("xs_hier_sys")

    # Week-over-week drift of the aggregate, both legs at or before the cut.
    drift = sysm.shift(shift) - sysm.shift(shift + 168)
    b = pd.DataFrame(
        np.repeat(drift.to_numpy()[:, None], w.shape[1], 1), index=w.index, columns=w.columns
    )
    out = _attach(out, b, "xs_hier_drift")
    names.append("xs_hier_drift")

    # One level down the hierarchy: the zone's own seasonal-naive aggregate.
    zm = w.T.groupby(zones).transform("mean").T.shift(shift)
    out = _attach(out, zm, "xs_hier_zone")
    names.append("xs_hier_zone")
    return out, names


# ---------------------------------------------------- B: PER-SERIES covariate windows (sprint 2B)
# A DIFFERENT FAMILY FROM A1-A15, and the distinction is the whole point. Every A-block aggregates
# ACROSS series; these smooth each series' OWN covariates ALONG TIME and leave the cross-section
# alone. Group 44's log reports the two levers separately -- centred covariate windows +5.1% on
# their GBM (0.1460 -> 0.1414), cross-series aggregates +3.2% -- and we screened only the second.
# A7 is not this: it rolls the SYSTEM MEAN, so it still collapses the cross-section first.
#
# LEGAL FOR THE SAME REASON THE A-BLOCKS ARE. Every covariate is supplied for the entire 672h
# horizon, so a window centred on forecast hour `fc` reads only known-future values. `min_periods=1`
# truncates at the frame edges rather than propagating NaN.
#
# WHY IT SHOULD HELP A TREE MORE THAN A NET. The tree sees one (series, origin, step) row at a time
# with every covariate evaluated POINTWISE at the forecast hour -- `feature_columns()` order is
# lags, wlags, rolls, horizon_step, futr, statics -- so it has no way to see that a covariate is
# spiking rather than merely high. A TFT decoder attends across horizon steps and can construct
# this internally. So a null here would be weaker evidence against the lever than a null on the net.
def _centred_windows(df, cols, wins, prefix):
    """Per-series centred rolling means of `cols`. `_wide` is [time x series], so a DataFrame
    `.rolling` walks the TIME index independently per series column -- no cross-series mixing."""
    out, names = df, []
    for c in cols:
        w = _wide(df, c)
        for win in wins:
            r = w.rolling(win, center=True, min_periods=1).mean()
            name = f"{prefix}{win}_{c}"
            out = _attach(out, r, name)
            names.append(name)
    return out, names


def block_B1(df, zones):
    """THEIR §7 LEVER, on the four covariates they aggregated. 4 x {3,6,12,24} = 16 columns."""
    return _centred_windows(df, THEIRS, (3, 6, 12, 24), "cw")


def block_B2(df, zones):
    """WIDE AND SHALLOW: every known-future signal, one window. 13 columns, not 52 -- if local
    temporal context is what is missing, one well-chosen width on all of them should show it, and
    it costs a quarter of B1's width on a screen where VSN dilution is a measured hazard."""
    return _centred_windows(df, KNOWN_FUTURE_SIGNALS, (24,), "cw")


def block_B3(df, zones):
    """B1 + A9 -- THE ONLY COMBINATION THAT DECIDES ANYTHING. A9 ships; the question is not whether
    covariate windows beat nothing, it is whether they add ON TOP of what we already carry. Screened
    as one block because the two families are orthogonal by construction (one collapses the
    cross-section, the other walks time) and a sum of separately-measured gains is not a
    measurement."""
    out, names = block_B1(df, zones)
    out, a9 = block_A9(out, zones)
    return out, [*names, *a9]


# ------------------------------------------------- C: PER-SERIES HOUR-OF-WEEK CLIMATOLOGY (2B)
# THE THIRD FAMILY, AND THE ONE AIMED AT THE GRADED METRIC. Group 44 call this a "Key decision":
# hour-of-week climatology as a KNOWN-FUTURE feature, fitted strictly on the training region. It is
# absent from `src/` -- `TIME_ENCODINGS` carries hour_sin/cos and dow_sin/cos, which are GLOBAL
# harmonics identical for every series, not a PER-SERIES phase-resolved profile.
#
# WHY IT SHOULD PAY MOST AT GAP 336, WHICH IS WHAT THE GRADE SCORES. A9 attenuated on the near board
# (x0.63) precisely because fresh history crowds it out. Climatology is the opposite shape of lever:
# when the model's own history is two weeks stale, a per-series profile at the MATCHING hour-of-week
# is the only thing that still carries that unit's idiosyncratic shape.
#
# NOT LEAKAGE, AND THE ARGUMENT IS A14's. The profile is a statistic of `_hidx < cut` ONLY,
# evaluated at the forecast hour -- the same standing as A8's clusters, A12's components and
# A14's statics.
# The screen's fence forbids target aggregates read AT the forecast hour; this reads none.
#
# READ A TREE NULL WEAKLY. `lgbm.WEEKLY_LAGS` (672/840/1008) already gives the tree y at the
# matching phase -- a 3-sample, noisy climatology. This block is its denoised form, so the tree
# has partial cover and may show little. The TFT has NO per-series phase-resolved feature at all,
# so a null here would NOT transfer to the neural arm.
def _how_profile(df, cut: int):
    """[time x series] of each series' mean y at the matching hour-of-week, on `_hidx < cut`."""
    hist = df[df[HOUR_IDX] < cut]
    how = hist[HOUR_IDX] % 168
    prof = hist.groupby([hist[NF_ID], how])[NF_TARGET].mean().unstack(fill_value=np.nan)
    prof = prof.apply(lambda r: r.fillna(r.mean()), axis=1)  # a series with an unseen bin
    w = _wide(df, NF_TARGET)  # index/columns template only
    idx = pd.Index(w.index % 168, name=HOUR_IDX)
    return pd.DataFrame(
        prof.reindex(columns=idx.values).T.to_numpy(), index=w.index, columns=prof.index
    ).reindex(columns=w.columns)


def block_C1(df, zones, cut: int = 3648):
    """Per-series hour-of-week climatology. ONE column -- the level at the matching phase."""
    prof = _how_profile(df, cut)
    return _attach(df, prof, "clim_how"), ["clim_how"]


def block_C2(df, zones, cut: int = 3648):
    """C1 plus the SHAPE alone: the profile divided by that series' train-region mean, which strips
    the level a static already carries and leaves only WHEN this unit is busy relative to itself."""
    prof = _how_profile(df, cut)
    lvl = df[df[HOUR_IDX] < cut].groupby(NF_ID)[NF_TARGET].mean()
    shape = prof.div(lvl.reindex(prof.columns).replace(0, np.nan), axis=1)
    out = _attach(df, prof, "clim_how")
    out = _attach(out, shape, "clim_how_shape")
    return out, ["clim_how", "clim_how_shape"]


def block_C3(df, zones, cut: int = 3648):
    """C1 + A9 -- does the phase profile ADD to what ships? The decision block, as B3 is for B1."""
    out, names = block_C1(df, zones, cut)
    out, a9 = block_A9(out, zones)
    return out, [*names, *a9]


# ------------------------------------------- EW: BACKWARD EWMA of the per-unit signals (sprint 2C)
# PORTED VERBATIM from the aggregate-screening script this lane was prototyped in (2026-09-04) so
# there is ONE implementation and `--xs-block` reaches the Chronos trainer and the CV members
# through the same code the tree numbers were measured on.
#
# A THIRD FAMILY. A-blocks collapse the CROSS-SECTION; B-blocks smooth a series' own covariates with
# a CENTRED window; these accumulate BACKWARD along time, per series, and exclude hour t from its
# own feature. Centred and backward are not the same lever: a centred window is a local LEVEL, a
# backward EWMA is one-directional MEMORY.
#
# THE MECHANISM, AND IT MAKES A FALSIFIABLE PREDICTION. Two signals' cross-correlation with the
# target does NOT peak at lag 0 -- `service_irregularity_risk_forecast` at +7h and
# `throughput_disruption_risk_forecast` at +12h -- so they act through a decaying accumulation
# rather than at a point. Backward vs forward EWMA on the residual after removing calendar and
# same-hour signals: +0.125 vs -0.012 and +0.108 vs -0.003 (a SIGN FLIP, i.e. genuine
# one-directional memory), against 0.239/0.249 and 0.250/0.259 for queue/network, which are
# symmetric and are a LEVEL the B-blocks already carry. Half-life 6 dominates 24 and 48: the
# memory is ~6h.
#
#     PREDICTION: LARGE on Chronos, ~NULL on the TFT.
#
# The tree reads each covariate as a bare scalar at the forecast hour and cannot accumulate.
# Chronos-2 ingests covariates per timestep with no recurrent covariate encoder -- the same gap. The
# TFT's LSTM encoder over the covariate sequence IS an EWMA with a learned, input-dependent decay
# (`h_t = f*h_{t-1} + (1-f)*x_t`), so it can build this itself AND tune the half-life, which a fixed
# {6,12,24,48} grid cannot. **If EWMA lands on the TFT the mechanism is WRONG, and that is the more
# interesting result -- report it as such rather than rescuing the hypothesis.**
#
# LEGALITY. Strictly backward and shifted one hour, so a row reads only known-future SIGNAL values
# at hours preceding its own forecast hour. `target` is never touched. Valid at gap 336.
_EW_SIGNALS = [
    "shock_risk",
    "maintenance_known",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
]
# The two whose memory is asymmetric — the ones the mechanism says carry the effect.
_EW_ASYM = ["service_irregularity_risk_forecast", "throughput_disruption_risk_forecast"]
_EW_HL = (6, 12, 24, 48)


def _ewma_back(df, cols, halflives):
    """Backward EWMA per series, shifted one hour so hour t is excluded from its own feature."""
    out = df.sort_values([NF_ID, HOUR_IDX]).reset_index(drop=True)
    g = out.groupby(NF_ID, sort=False)
    names = []
    for c in cols:
        v = out[c].astype(float)
        v = v.fillna(g[c].transform("median")).fillna(float(np.nanmedian(out[c])))
        for hl in halflives:
            n = f"ew_{c}_{hl}"
            out[n] = (
                v.groupby(out[NF_ID], sort=False)
                .transform(lambda s, h=hl: s.ewm(halflife=h, adjust=False).mean())
                .groupby(out[NF_ID], sort=False)
                .shift(1)
            )
            out[n] = out[n].fillna(v)
            names.append(n)
    return out, names


def block_EW(df, zones):
    """Backward EWMA of the 8 per-unit signals at half-lives 6/12/24/48 (32 columns)."""
    return _ewma_back(df, _EW_SIGNALS, _EW_HL)


def block_EW2(df, zones):
    """PARSIMONIOUS: only the two asymmetric-memory signals, half-lives 6 and 12 (4 columns)."""
    return _ewma_back(df, _EW_ASYM, (6, 12))


def block_EWA9(df, zones):
    """EW on top of A9 -- the marginal over what already SHIPS, which is the decision number."""
    out, a9 = block_A9(df, zones)
    out, ew = _ewma_back(out, _EW_SIGNALS, _EW_HL)
    return out, [*a9, *ew]


def block_EW2A9(df, zones):
    """A9 + the FOUR parsimonious columns. 7 columns total.

    THE ARM THE BRIEF MOST WANTS, and the reason is a confound rather than a preference: `EWA9` adds
    32 columns to a Chronos fine-tune held at 2000 steps, so a null there cannot be told apart from
    'a wider input space needs more steps'. Four columns cannot plausibly demand a longer schedule,
    so this arm isolates the FEATURE from the BUDGET. On the tree EW2 was worth ~60% of EW at an
    eighth of the width."""
    out, a9 = block_A9(df, zones)
    out, ew = _ewma_back(out, _EW_ASYM, (6, 12))
    return out, [*a9, *ew]


BLOCKS = {
    "A1": ("their exact set (control)", block_A1),
    "A2": ("degenerate-free + refilled", block_A2),
    "A3": ("leave-one-out system mean", block_A3),
    "A4": ("shares + cross-sectional ranks", block_A4),
    "A5": ("cross-sectional dispersion", block_A5),
    "A6": ("missingness regime", block_A6),
    "A7": ("rolling aggregates (crosses both wins)", block_A7),
    "A8": ("learned partition (corr clusters)", block_A8),
    "A9": ("capacity-weighted system mean", block_A9),
    "A10": ("origin-anchored system target", block_A10),
    "A11": ("lagged + leading system aggregates", block_A11),
    "A12": ("principal components of the panel", block_A12),
    "A13": ("CHRONOS CONSENSUS (paired, needs --chronos)", block_A13),
    "A14": ("time-axis: per-series whole-history statics", block_A14),
    "A15": ("hierarchical: seasonal-naive of the aggregate", block_A15),
    "B1": ("per-series centred covariate windows (their §7)", block_B1),
    "B2": ("per-series centred windows, all signals, w=24", block_B2),
    "B3": ("B1 + A9 -- does it ADD to what ships?", block_B3),
    "C1": ("per-series hour-of-week climatology", block_C1),
    "C2": ("climatology level + shape", block_C2),
    "C3": ("C1 + A9 -- does it ADD to what ships?", block_C3),
    "EW": ("backward EWMA, 8 signals x hl{6,12,24,48} (32 cols)", block_EW),
    "EW2": ("backward EWMA, 2 asymmetric signals x hl{6,12} (4 cols)", block_EW2),
    "EWA9": ("A9 + EW (35 cols)", block_EWA9),
    "EW2A9": ("A9 + EW2 (7 cols) -- isolates the feature from the step budget", block_EW2A9),
}
# Blocks whose builder needs the cutoff (fitted on the train region only).
NEEDS_CUT = {"A8", "A12", "A14", "C1", "C2", "C3"}
# Blocks that require the per-cut cascade frame rather than raw train.csv.
NEEDS_CHRONOS = {"A13"}
