"""Locate the S8 additional-dataset Parquet files, on Modal or locally.

The corpora are staged by ``tools/modal_addl_data.py`` into the Modal volume
``tsf-addl-data`` (mounted at ``/data``), and the small derived files are mirrored locally
under ``data/external/``. Both loaders resolve through here so neither has to care which
environment it is running in.

Resolution order:
1. ``$ADDL_DATA_ROOT`` — explicit override (tests, ad-hoc runs). **Exclusive**: when set, the
   defaults below are not consulted, so a test can point at an empty directory and be sure it
   is exercising the missing-corpus path rather than silently finding the real files.
2. ``/data`` — the Modal volume mount point.
3. ``data/external`` — the local mirror, relative to the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_ROOTS = ("/data", str(_REPO_ROOT / "data" / "external"))


def addl_roots() -> list[Path]:
    """Candidate roots holding ``{m5,ecl}/derived/…``, most specific first.

    ``$ADDL_DATA_ROOT`` wins outright when set — see module docstring.
    """
    override = os.environ.get("ADDL_DATA_ROOT")
    if override:
        return [Path(override)]
    return [Path(r) for r in CANDIDATE_ROOTS]


def resolve(relpath: str) -> Path:
    """Return the first existing ``root/relpath``.

    Raises loudly rather than falling back to a fixture — a missing corpus must fail the run,
    not silently downgrade it to synthetic data (that failure mode is what made S8's original
    wiring dangerous; see plan S8.0).
    """
    tried = []
    for root in addl_roots():
        p = root / relpath
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"additional-dataset file {relpath!r} not found. Tried: {tried}. "
        "Stage it with `modal run tools/modal_addl_data.py`, mount the `tsf-addl-data` volume "
        "at /data, or set $ADDL_DATA_ROOT."
    )
