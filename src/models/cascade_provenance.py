"""Provenance for cascade covariates — and the gap-honesty check the cascade needed and lacked.

A cascade column (``chronos2_forecast``) is a *forecast* stored as a covariate, so the only thing
distinguishing an honest value from a leaky one is **which hours its context was allowed to see**.
That fact lives nowhere in the CSV. This module records it beside the frame and enforces it before
any member conditions on it.

**Why this exists.** ``src.models.chronos2_oof`` rolls the origin back in 336h blocks: block ``s``
forecasts ``[s, s+336)`` from context ``[0, s)``. Its docstring called that "leakage-free ... all
consumers inherit the honesty for free". Two different properties were being conflated:

* **context causality** — ``chronos2_forecast[t]`` never saw ``target[t]``. True, and enough for
  *training* a model on the train region.
* **gap honesty** — for a gapped window with cutoff ``c``, no covariate value over the forecast
  horizon ``[c, c+672)`` saw ``target`` at or after ``c``. This is what the graded task requires,
  and the 336h grid does **not** provide it.

Concretely, at ``c = 3648`` the scored block is ``[3984, 4320)`` and its covariate comes from block
``s = 3984`` — a context ending at 3983, i.e. **336 hours past the cutoff**. The TFT was handed a
fresh 336-step-ahead forecast of the block it was supposed to reach blind across a 672h gap. Every
CV cutoff is on the same grid, so every window leaked the same way, over its far half only.

The measured signature matches exactly. Splitting the cached cascade predictions by horizon half
(``results/member_metrics_near_far.json``): steps 1-336 land in ``[c, c+336)``, whose covariate
comes from block ``s = c`` — context ending at ``c-1``, honest — and there the cascade **loses** to
plain TFT (0.1509 vs 0.1491). Steps 337-672 are the leaky half, and there it wins (0.1464 vs
0.1486). The cascade's entire apparent advantage sat in the half where the covariate had seen the
gap, which also resolves the contradiction recorded as F8 (better on CV far and on the near
leaderboard, worse on CV near) without needing to appeal to sampling noise.

**Fail closed.** A frame with no provenance is refused rather than trusted. We cannot tell an
honest generator from the one above by looking at the numbers — that is the whole problem — so
"unknown" has to mean "no".
"""

from __future__ import annotations

import json
from pathlib import Path

PROVENANCE_SUFFIX = ".provenance.json"


def provenance_path(csv_path: str | Path) -> Path:
    """Sidecar path for a derived frame: ``train_chronos.csv`` -> ``train_chronos.provenance.json``.

    A sidecar rather than a column: the grid is a property of the whole frame, and the derived CSV
    must keep the raw schema plus exactly one column so every downstream loader stays unchanged.
    """
    p = Path(csv_path)
    return p.with_suffix(PROVENANCE_SUFFIX)


def write_provenance(
    csv_path: str | Path,
    *,
    column: str,
    blocks: list[tuple[int, int]],
    n_hours: int,
    generator: str,
    zero_shot: bool,
    reconstructed: bool = False,
    note: str = "",
) -> Path:
    """Record how a cascade column was generated, next to the frame that carries it.

    ``blocks`` is ``[(start, length), ...]``: block ``(s, L)`` covers hours ``[s, s+L)`` and was
    forecast from context ``[0, s)``. Lengths vary — a gap-honest frame rolls the train region in
    336h blocks and then covers the whole 672h horizon in one — so the length travels with each
    block rather than sitting beside the list as a single number.
    """
    out = provenance_path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted((int(s), int(length)) for s, length in blocks)
    out.write_text(
        json.dumps(
            {
                "column": column,
                "blocks": [list(b) for b in ordered],
                "n_hours": int(n_hours),
                "generator": generator,
                "zero_shot": bool(zero_shot),
                # True = inferred from the generator's known grid rather than written by the run
                # that produced the frame. Sound only when the frame's NaN warm-up prefix matches.
                "reconstructed": bool(reconstructed),
                "note": note,
            },
            indent=2,
        )
        + "\n"
    )
    return out


def merge_provenance(csv_path: str | Path, parts: dict[str, dict], *, note: str = "") -> Path:
    """Record SEVERAL channels' grids beside one frame — the ALONGSIDE (two-channel) cascade.

    ``parts`` maps ``column -> that column's provenance dict`` (each as
    :func:`write_provenance` would have written it alone).

    Why the shape had to grow. A two-channel cascade member declares
    ``cascade=("chronos2_forecast", "toto_forecast")`` and ``MemberSpec.train_csv`` is a *single*
    path, so both columns arrive on one merged frame — and ``check_gap_honest`` refuses a sidecar
    whose ``column`` does not match the channel it was asked about. One flat sidecar can only name
    one column, so a merged frame would have failed closed on the second channel however honest it
    was. The plan called the two-channel mode "a registration plus a derived frame, no refactor";
    that was right about the registration and wrong about the sidecar, and this is the difference.

    The flat single-column form stays readable and stays the output of ``write_provenance`` — every
    shipped sidecar is in it, and a format migration is the last thing a fail-closed check should
    require.
    """
    out = provenance_path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        col: {
            "blocks": [list(b) for b in sorted((int(s), int(ln)) for s, ln in meta["blocks"])],
            "n_hours": int(meta["n_hours"]),
            "generator": meta.get("generator", ""),
            "zero_shot": bool(meta.get("zero_shot", False)),
            "reconstructed": bool(meta.get("reconstructed", False)),
            "note": meta.get("note", ""),
        }
        for col, meta in parts.items()
    }
    out.write_text(json.dumps({"columns": columns, "note": note}, indent=2) + "\n")
    return out


def read_provenance(csv_path: str | Path, column: str | None = None) -> dict | None:
    """The recorded grid for a derived frame, or None if the sidecar is absent.

    Accepts both sidecar shapes. Pass ``column`` to select one channel out of a multi-column
    sidecar; without it, a multi-column sidecar returns its single entry if it has exactly one and
    raises otherwise — silently picking a channel is how the wrong grid would end up vouching for
    the right column.
    """
    p = provenance_path(csv_path)
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    if "columns" not in meta:
        return meta
    cols = meta["columns"]
    if column is None:
        if len(cols) != 1:
            raise ValueError(
                f"{p.name} describes {len(cols)} columns {sorted(cols)}; name the one you mean."
            )
        column = next(iter(cols))
    if column not in cols:
        raise ValueError(f"{p.name} has no provenance for column {column!r}; has {sorted(cols)}")
    return {"column": column, **cols[column]}


def _blocks(meta: dict) -> list[tuple[int, int]]:
    """``[(start, length), ...]``, accepting the uniform ``block``/``starts`` shorthand too."""
    if "blocks" in meta:
        pairs = [(int(s), int(length)) for s, length in meta["blocks"]]
    else:
        block = int(meta["block"])
        pairs = [(int(s), block) for s in meta["starts"]]
    # Always ascending: the generator rolls the origin *backwards*, so a grid arriving in
    # generation order would be descending and the lookups below read as if it were sorted.
    return sorted(pairs)


def context_end_for_hour(meta: dict, hour: int) -> int | None:
    """Last hour the covariate value at ``hour`` was allowed to condition on (None = uncovered).

    Uncovered hours are the generator's warm-up prefix, where the column is NaN and the shared
    imputation fills it — degraded, but not leaky.
    """
    covering = [s for s, length in _blocks(meta) if s <= hour < s + length]
    return (max(covering) - 1) if covering else None


def gap_leak_hours(meta: dict, cut_idx: int, horizon: int) -> int:
    """How far past the cutoff the covariate looked over ``[cut, cut+horizon)``. 0 = honest.

    The forecast origin is ``cut-1``, so an honest covariate has every context ending at or before
    ``cut-1``. A block starting at ``s > cut`` ends its context at ``s-1``, which is ``s - cut``
    hours of target the member is not allowed to see.
    """
    # A block starting at or before the cutoff is fine however far it reaches — its context still
    # ends before the origin. Only a block that *begins* inside the horizon looks past the cutoff,
    # and the last such block is the one that saw the most, so it sets the leak.
    offenders = [s for s, _ in _blocks(meta) if cut_idx < s < cut_idx + horizon]
    return int(max(offenders) - cut_idx) if offenders else 0


def check_gap_honest(csv_path: str | Path, column: str, cut_idx: int, horizon: int) -> dict:
    """Raise unless the cascade covariate over ``[cut, cut+horizon)`` is gap-honest. Fail closed.

    Returns the provenance dict when the frame passes, so callers can record what they conditioned
    on rather than asserting it separately.
    """
    meta = read_provenance(csv_path, column)
    if meta is None:
        raise ValueError(
            f"cascade covariate {column!r} in {csv_path} has no provenance sidecar "
            f"({provenance_path(csv_path).name}), so its gap honesty cannot be verified. "
            "Refusing rather than assuming: a covariate whose context ran past the cutoff scores "
            "better, not worse, so an unverified frame fails silently in the flattering direction. "
            "Regenerate with `python -m src.models.chronos2_oof --cut-idx <cut>` (which writes the "
            "sidecar), or backfill one with src.models.cascade_provenance.write_provenance if you "
            "can establish the grid independently."
        )
    if meta.get("column") != column:
        raise ValueError(
            f"provenance for {csv_path} describes column {meta.get('column')!r}, not {column!r}"
        )
    leak = gap_leak_hours(meta, cut_idx, horizon)
    if leak:
        raise ValueError(
            f"cascade covariate {column!r} is NOT gap-honest at cut_idx={cut_idx}: over the "
            f"forecast horizon [{cut_idx}, {cut_idx + horizon}) its context runs {leak} hours past "
            f"the cutoff ({len(_blocks(meta))} blocks). It would condition on {leak} hours "
            "of target it cannot see at test time, which inflates the far half of the horizon and "
            "leaves the near half honest — see this module's docstring for the measured signature. "
            f"Regenerate the covariate for this window with a single {horizon}h block anchored at "
            f"the cutoff: `python -m src.models.chronos2_oof --cut-idx {cut_idx} "
            f"--horizon {horizon}`."
        )
    return meta
