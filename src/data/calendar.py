"""The six deterministic calendar covariates, produced for gap hours nobody supplies a file for.

Why this cannot just call S3's ``exact`` modifier — measured, and it turns on ONE observation
-----------------------------------------------------------------------------------------------
``src.data.gap_fill.exact_how_columns`` detects a column that is an exact function of
(series, hour-of-week) and reconstructs it outright. On the **train slice** it finds five of the six
``TIME_ENCODINGS`` and misses only ``trend``::

    exact_how_columns(train[TIME_ENCODINGS])
        -> ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
    not detected
        -> ['trend']

So in CV the five are already handled and this module would be a duplicate. **At inference they are
not**, and the reason is exact and slightly absurd::

    EXACT_MIN_OBS                                   = 3
    obs per (series, hour-of-week) bin, 4320h train = 25.7   -> detector fires
    obs per (series, hour-of-week) bin, 336h supplied = 2.0  -> detector CANNOT fire

The detector demands three observations per bin before it will call a column deterministic — a
sound rule, since two matching values are thin evidence. But the submission hands us a **336-hour**
covariate block, which is exactly two weeks, which is exactly **two** observations per bin. It
misses by one, silently, and every one of the five then falls through to a per-series median: a
**constant** ``hour_sin`` across the whole gap. Smooth, plausible, and undetectable downstream.

Closed form, VERIFIED — not hardcoded, and not detected either
--------------------------------------------------------------
The way out is neither of the two obvious ones. Baking a lookup into ``checkpoint.pt`` is refused
outright (**the specification: "You should not bake any data into your checkpoint, as the
timeframe might differ"**), and trusting a hardcoded ``sin(2*pi*h/24)`` asserts a convention
nobody checked.

So this module computes the closed form and then **checks it against the rows we were actually
given** before using it on the rows we were not. If the supplied block agrees to 1e-9, the same
formula is trusted across the gap; if it does not, the call raises. That keeps the
*measured-not-asserted* discipline the detector had, while needing three observations from nowhere.
Verified against the real ``validation_input.csv``::

    hour_sin  max|d| 1.110e-16      dow_sin  max|d| 5.551e-17
    hour_cos  max|d| 1.110e-16      dow_cos  max|d| 2.776e-17
    is_weekend max|d| 0.000e+00     trend    max resid 4.441e-16 (linear in absolute hours)

``trend`` is different again
----------------------------
It is **monotone in absolute time, not periodic**, so no hour-of-week rule reaches it at any
observation count, and it has no closed form we could guess. It is therefore **fitted** — a line
through whatever ``(timestamp, trend)`` pairs the caller holds, then extrapolated. A hardcoded slope
would encode *this* dataset's timeframe, the exact assumption that ruling warns about.

Both paths carry a **tripwire** and both fail closed. That is the same reflex as #58's
``get_activation_fn`` check and ``cascade_provenance``'s mandatory sidecar, and it is here because
a wrong deterministic covariate over a 336h gap is a smooth column of believable numbers — the
class of error this project has been caught by six times.

This also sharpens a claim carried elsewhere: that the calendar columns are *"deterministic
functions of the timestamp, so there is nothing to impute and no data to bake."*
The second half is right and is the load-bearing half. The first half is looser than it reads: the
values still have to be **produced**, and on the submission path nothing in the repo produced them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import TIME

# Residual above which the linear model is treated as FALSIFIED rather than extrapolated. The
# measured residual on real data is ~4e-16, so 1e-6 leaves ten orders of margin while still
# catching any genuine change of shape.
TREND_LINEAR_TOL = 1e-6
TREND_COL = "trend"

# Agreement required between the closed form and the rows we were actually handed, before the
# formula is trusted on rows we were not. Measured agreement is ~1e-16, so this is pure margin.
PERIODIC_TOL = 1e-9


def periodic_encodings(ts: pd.Series) -> pd.DataFrame:
    """The five closed-form calendar encodings for any timestamps."""
    ts = pd.to_datetime(ts)
    hour = ts.dt.hour.to_numpy(dtype=float)
    dow = ts.dt.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "is_weekend": (dow >= 5).astype(float),
        },
        index=ts.index,
    )


def verify_periodic(reference: pd.DataFrame, tol: float = PERIODIC_TOL) -> None:
    """Check the closed form against supplied rows. Raises on disagreement.

    This is what earns the right to use the formula on the gap: it is confirmed on every row we
    were given before it is trusted on any row we were not.
    """
    ref = reference.dropna(subset=[TIME])
    if ref.empty:
        raise ValueError("no supplied rows to verify the calendar encodings against")
    want = periodic_encodings(ref[TIME])
    bad = {}
    for col in want.columns:
        if col not in ref.columns:
            continue
        have = pd.to_numeric(ref[col], errors="coerce")
        ok = have.notna()
        if not ok.any():
            continue
        delta = float((have[ok].to_numpy() - want[col].to_numpy()[ok.to_numpy()]).__abs__().max())
        if delta > tol:
            bad[col] = delta
    if bad:
        raise ValueError(
            f"calendar encodings do not match the closed form on the SUPPLIED rows: {bad}. "
            "The dataset's convention differs from sin(2*pi*h/24) / sin(2*pi*dow/7); refusing to "
            "extrapolate it across the gap."
        )


def complete_periodic(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill missing periodic encodings from the timestamp, after verifying the formula.

    Values already present are left untouched — a supplied row keeps the harness's own number.
    """
    out = frame.copy()
    verify_periodic(reference if reference is not None else out)
    want = periodic_encodings(out[TIME])
    for col in want.columns:
        if col not in out.columns:
            out[col] = want[col]
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(want[col])
    return out


def fit_trend(ts: pd.Series, trend: pd.Series) -> tuple[float, float, pd.Timestamp]:
    """Least-squares ``trend = slope * hours_since(origin) + intercept``, with a linearity tripwire.

    Returns ``(slope, intercept, origin)``. Raises if the observed rows are not linear to
    ``TREND_LINEAR_TOL``.
    """
    ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    tr = pd.to_numeric(pd.Series(trend).reset_index(drop=True), errors="coerce")
    ok = ts.notna() & tr.notna()
    ts, tr = ts[ok], tr[ok]
    if len(ts) < 2:
        raise ValueError(f"need >= 2 observed `trend` rows to fit a line, got {len(ts)}")

    origin = ts.min()
    hours = (ts - origin).dt.total_seconds().to_numpy() / 3600.0
    if float(np.ptp(hours)) == 0.0:
        raise ValueError("all observed `trend` rows share one timestamp; cannot fit a slope")

    slope, intercept = np.polyfit(hours, tr.to_numpy(dtype=float), 1)
    resid = float(np.abs(np.polyval([slope, intercept], hours) - tr.to_numpy(dtype=float)).max())
    if resid > TREND_LINEAR_TOL:
        raise ValueError(
            f"`trend` is not linear in time (max residual {resid:.3e} > {TREND_LINEAR_TOL:.0e}); "
            "refusing to extrapolate it across the gap. Inspect the covariate file."
        )
    return float(slope), float(intercept), origin


def complete_trend(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill missing ``trend`` values in ``frame`` by extrapolating the line its observed rows imply.

    Values already present are **left untouched** — a row the harness supplied keeps its own
    own number, and only genuinely absent rows are computed. ``reference`` supplies the observed
    pairs when ``frame`` itself has none (e.g. an all-gap block); it defaults to ``frame``.
    """
    out = frame.copy()
    if TREND_COL not in out.columns:
        out[TREND_COL] = np.nan
    out[TREND_COL] = pd.to_numeric(out[TREND_COL], errors="coerce")
    if not out[TREND_COL].isna().any():
        return out

    ref = out if reference is None or TREND_COL not in reference.columns else reference
    slope, intercept, origin = fit_trend(ref[TIME], ref[TREND_COL])
    hours = (pd.to_datetime(out[TIME]) - origin).dt.total_seconds() / 3600.0
    out[TREND_COL] = out[TREND_COL].fillna(slope * hours + intercept)
    if out[TREND_COL].isna().any():
        raise ValueError("`trend` still NaN after extrapolation")
    return out
