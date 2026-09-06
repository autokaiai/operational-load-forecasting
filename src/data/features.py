"""Column-role registry — the single source of truth for which column plays which role.

Every model and both code paths (train + inference) import their feature lists from here,
so a config only ever names a model, never re-lists columns. Roles are derived from the
real ``train.csv`` / ``validation_input.csv`` schema (25 columns).
"""

from __future__ import annotations

from contextlib import contextmanager

# Identifier / target columns in the raw CSVs (renamed to neuralforecast long-format
# names — unique_id / ds / y — in src.data.loader).
ID = "series_id"
TIME = "timestamp"
TARGET = "target"

# Static per-unit covariates (constant within a series; verified one distinct value/series).
STATIC_COLS = ["nominal_capacity", "zone_sin", "zone_cos"]

# Deterministic calendar features — known for every future timestamp, never NaN.
TIME_ENCODINGS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "trend"]

# Planning / risk signals delivered for the forecast horizon ("known-future covariates").
# Present in validation_input.csv future rows. Order mirrors the CSV for readability.
KNOWN_FUTURE_SIGNALS = [
    "workload_intensity",
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "promotion_intensity",
    "shock_risk",
    "maintenance_known",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
]

# The 10 columns carrying ~4.5% NaN. NaN means "unavailable", not zero: we impute with a
# stored median and flag it with a `*_missing` indicator (see src.data.impute).
NAN_COLS = [
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "shock_risk",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
]

MISSING_SUFFIX = "_missing"

# --------------------------------------------------------------------------- feature cascade
#
# Base-model forecasts of `target`, fed to the TFT as known-future covariates so its Variable
# Selection Network learns when to trust the global prior instead of a fixed output average.
# Generated leakage-free by rolling origin (``src.models.chronos2_oof``) and ABSENT from the raw
# CSVs — they live only in the derived frames (``data/derived/train_chronos.csv``).
#
# **Opt-in, and empty by default.** This is the difference that matters. `futr_exog_list()` is
# global: every nf member and both code paths read it. Registering a cascade column unconditionally
# — which is what an early cascade branch did, and what the sibling cascade worktrees still do —
# makes every plain member demand a column `train.csv` does not have. That registration never
# reached `main`, which is why `configs/tft_chronos.yaml` used to claim the covariate was
# "registered in src.data.features" while it was not, and why running that config from this repo
# silently trains a PLAIN TFT that ignores the Chronos column entirely.
#
# So a channel is active only when a caller asks for it, by either route:
#   * ``with cascade_channels("chronos2_forecast"):`` — in-process, and the one to use when
#     several members share a process: an env var cannot vary per member;
#   * ``CASCADE_CHANNELS=chronos2_forecast`` — for subprocess launches (a remote runner, the CLI).
#
# EVERY CHANNEL HERE IS A ZERO-SHOT FORECAST, and that is a rule rather than a coincidence: a model
# FITTED ON OUR DATA enters only as a blend member, never as the conditioning set (2026-07-31).
# The reason is leakage — an OOF forecast for hour `t` from a model trained on our targets can
# encode target[t], and the fence is expensive and easy to get subtly wrong. The line is
# fitted-on-our-data, not "any model": these weights never saw this dataset, which is what makes
# `tft_cascade` at 0.1341 compatible with the rule rather than an exception to it.
# Do NOT add an LGBM or TFT OOF column here.
CASCADE_FORECASTS_ALL = [
    "chronos2_forecast",  # Chronos-2 ZERO-SHOT (src.models.chronos2_oof); NaN over the warm-up
    # Plan S2 candidates (src.models.foundation) — pretrained on outside corpora, never fitted here.
    # Listed so a channel can be activated; registering a name costs nothing until a member declares
    # it, because the selection is opt-in and empty by default.
    "toto_forecast",  # Datadog, 313M, observability telemetry
    "timesfm_forecast",  # Google, ~200M, Trends / Wikipedia / synthetic
    "tabpfn_ts_forecast",  # TabPFN v3 as a tabular regressor over time features
    "tirex_forecast",  # NX-AI, 35M, xLSTM
    # COVARIATE-AWARE twins. The four above are target-only, and the screen's control
    # (`chronos2_forecast`) is not — `chronos2_oof` feeds it all 29 known-future covariates — so the
    # first screen compared covariate-blind candidates against a covariate-fed control and measured
    # the covariates rather than the models. These re-run the same models on the control's own
    # footing. Separate channels, so the covariate lift is a paired A/B on identical rows.
    # `tirex` has no twin: v1 exposes no covariate argument at all, and v2 (which does) truncates
    # our 672-step horizon to 320.
    "toto_cov_forecast",  # known_dynamic on Toto2Model.forecast — the README was wrong
    "tabpfn_ts_cov_forecast",  # tabular by construction — covariates are just columns
    "timesfm_cov_forecast",  # `forecast_with_covariates`, xreg_mode="xreg + timesfm"
]

_ACTIVE_CASCADE: list[str] = []


def _channels_from_env() -> list[str]:
    import os

    raw = os.environ.get("CASCADE_CHANNELS", "")
    return [c.strip() for c in raw.split(",") if c.strip()]


def set_cascade_channels(channels) -> list[str]:
    """Activate cascade covariates. Returns the previous selection (for restore)."""
    global _ACTIVE_CASCADE
    if isinstance(channels, str):
        channels = [channels]
    unknown = set(channels) - set(CASCADE_FORECASTS_ALL)
    if unknown:
        raise ValueError(
            f"unknown cascade channel(s) {sorted(unknown)}; choose from {CASCADE_FORECASTS_ALL}"
        )
    previous, _ACTIVE_CASCADE = (
        _ACTIVE_CASCADE,
        [c for c in CASCADE_FORECASTS_ALL if c in set(channels)],
    )
    return previous


def active_cascade_channels() -> list[str]:
    """Cascade covariates currently in the conditioning set — empty unless switched on."""
    return list(_ACTIVE_CASCADE)


@contextmanager
def cascade_channels(*channels):
    """Scope a cascade selection to a block, restoring whatever was active before.

    Use this rather than ``set_cascade_channels`` directly: it keeps a cascade member from
    leaking its extra covariate into the next member run in the same process.
    """
    flat = [c for arg in channels for c in ([arg] if isinstance(arg, str) else arg)]
    previous = set_cascade_channels(flat)
    try:
        yield active_cascade_channels()
    finally:
        set_cascade_channels(previous)


set_cascade_channels(_channels_from_env())


# --------------------------------------------------------------------------------------------
# DERIVED CROSS-SERIES AGGREGATES (sprint 2). Same state-not-rebinding pattern as the cascade
# above, and for the same reason, which is worth stating because the obvious alternative is a trap:
# `src/models/registry.py` does `from src.data.features import futr_exog_list` at module import and
# therefore holds its own reference to the function object. Monkeypatching `futr_exog_list` — on
# this module or on the caller's — does NOT reach the model; the aggregate screen gets away with
# rebinding `lgbm.futr_exog_list` only because the tree resolves the name at call time inside that
# one module. A neural arm wired the same way would
# train without the aggregates and return a confident null, which we would then read as
# "cross-series does not transfer to the TFT". Law 5, and the repo carries the identical warning in
# `nan_col_list` for `src.data.impute`.
#
# NOT in `nan_col_list`: these are DERIVED from columns that have already been imputed, so they are
# complete by construction and carry no `*_missing` twin. Appended LAST in `futr_exog_list` so that
# with nothing active the conditioning set is byte-identical to the shipped one and every existing
# checkpoint stays comparable.
_ACTIVE_AGGREGATES: list[str] = []


def set_aggregate_columns(columns) -> list[str]:
    """Activate derived cross-series aggregate covariates. Returns the previous selection."""
    global _ACTIVE_AGGREGATES
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    if any(not isinstance(c, str) or not c.strip() for c in columns):
        raise ValueError(f"aggregate column names must be non-empty strings, got {columns}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"duplicate aggregate column names: {columns}")
    base = {*TIME_ENCODINGS, *KNOWN_FUTURE_SIGNALS, *CASCADE_FORECASTS_ALL}
    clash = sorted(set(columns) & base)
    if clash:
        raise ValueError(f"aggregate column name(s) collide with existing covariates: {clash}")
    previous, _ACTIVE_AGGREGATES = _ACTIVE_AGGREGATES, columns
    return previous


def active_aggregate_columns() -> list[str]:
    """Derived aggregates currently in the conditioning set — empty unless switched on."""
    return list(_ACTIVE_AGGREGATES)


@contextmanager
def aggregate_columns(*columns):
    """Scope an aggregate selection to a block, restoring whatever was active before."""
    flat = [c for arg in columns for c in ([arg] if isinstance(arg, str) else arg)]
    previous = set_aggregate_columns(flat)
    try:
        yield active_aggregate_columns()
    finally:
        set_aggregate_columns(previous)


def nan_col_list() -> list[str]:
    """NaN-prone columns to impute — the base signals plus any ACTIVE cascade channel.

    Call this rather than reading ``NAN_COLS`` directly. ``src.data.impute`` imports names by
    value at module load, so a module-level list cannot pick up a cascade activated later; this
    function is evaluated per call and can.
    """
    return [*NAN_COLS, *_ACTIVE_CASCADE]


def missing_indicator_cols() -> list[str]:
    """Names of the binary `*_missing` columns added during imputation.

    Active cascade columns are included: they carry NaN over the warm-up hours preceding their
    first rolling origin (~6.7% of `train_chronos.csv`), so they are imputed and flagged exactly
    like the planning signals and train/inference see an identical layout.
    """
    return [f"{col}{MISSING_SUFFIX}" for col in nan_col_list()]


# These two lists are the full conditioning set. Which of them a given model actually receives
# is decided in src.models.registry from the model's capability flags: a covariate-free model
# (DLinear, PatchTST) gets neither; a futr-only model (TimesNet, Informer) gets only the futr
# list. Hence there is no model->exog map here — the model classes are the source of truth.
def futr_exog_list() -> list[str]:
    """Every known-future column the exog models condition on (calendar + signals + flags)."""
    return [
        *TIME_ENCODINGS,
        *KNOWN_FUTURE_SIGNALS,
        *_ACTIVE_CASCADE,
        *missing_indicator_cols(),
        *_ACTIVE_AGGREGATES,
    ]


def stat_exog_list() -> list[str]:
    """Static per-unit columns."""
    return list(STATIC_COLS)
