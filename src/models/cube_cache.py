"""Persist a member's three-window prediction cube, keyed by what actually determines it.

Why
---
The paired A/B runner re-ran **both** arms on every invocation. Screening four levers against
``lgbm_stride24`` therefore refit the identical baseline four times: of the ~4.2 h the 2026-07-31
composite batch took, roughly 1.5 h was one model being computed over and over. That was tolerable
while an arm cost 332 s and intolerable once 4.4 pushed it to 1817 s.

The second reason matters more than the first. Every A/B computes a full member cube and then
**throws it away**, so the tuned tree's predictions — the ones the Phase-6 blend search wants —
do not exist anywhere on disk. Keeping them turns "refit the tree for 30 minutes" into a file read.

Fail closed
-----------
A stale cache hit does not waste time, it reports a **wrong number**, and a wrong number is worse
than no number at all. So the key digests everything that determines the predictions — the
member's resolved config and overrides, the source of the fitting code, the seed, the cutoffs, and
the training frame's identity — and a runner that does not *declare* its parameters is not cached
at all. Silence means refit, never reuse.

The one thing the key deliberately does not cover is ``src/models/members.py`` itself. Hashing it
would invalidate every cube whenever a new member is registered, which is exactly when the cache is
most wanted (4.7 registers ``lgbm_best`` and then wants the ``lgbm_stride24`` cube it just spent an
hour on). Per-runner ``member_code`` names the files that actually do the fitting instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.models.members import get_member

CACHE_DIR = Path("results/member_cubes")


def _file_digest(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "absent"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _data_identity(path: str | Path) -> str:
    """Identify the training frame without hashing 326 MB on every call."""
    p = Path(path)
    if not p.exists():
        return "absent"
    st = p.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def cache_key(
    member: str,
    *,
    seed: int | None,
    cutoffs,
    train_csv: str,
    gap_cov: str = "impute",
    gap_fill: str = "median",
    nan_fill: str = "median",
) -> dict | None:
    """The full, human-readable key — or ``None`` if this member must not be cached.

    Returning ``None`` is the fail-closed path: a runner that does not declare
    ``member_overrides`` has parameters we cannot see, so we cannot tell a stale cube from a fresh
    one and must not try.

    The three imputation conditions are part of the key for exactly that reason. They do not
    change a single byte of the member's config, its overrides or its code, so a key without them
    would serve the ``median`` cube to an ``interp`` arm and report the A/B as a flat zero — a
    stale hit is not wasted time, it is a wrong number, and this one would be wrong in the
    direction of "no effect", which is the hardest kind to notice.
    """
    spec = get_member(member)
    run = spec.run
    if not hasattr(run, "member_overrides"):
        return None

    from src.models.members import _load_yaml

    cfg = getattr(run, "member_config", None)
    return {
        "member": member,
        "kind": spec.kind,
        "seed": None if seed is None else int(seed),
        "cutoffs": [int(c) for c in cutoffs],
        "config_path": cfg,
        "config": _load_yaml(cfg) if cfg else {},
        "overrides": dict(run.member_overrides),
        "cascade": list(spec.cascade),
        "train_csv": train_csv,
        "train_identity": _data_identity(train_csv),
        "code": {f: _file_digest(f) for f in getattr(run, "member_code", ())},
    }


def _digest(key: dict) -> str:
    blob = json.dumps(key, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _paths(member: str, key: dict) -> tuple[Path, Path]:
    d = _digest(key)
    return CACHE_DIR / f"{member}__{d}.parquet", CACHE_DIR / f"{member}__{d}.key.json"


def load(member: str, key: dict | None) -> pd.DataFrame | None:
    """Return the cached cube, or ``None`` on any doubt whatsoever."""
    if key is None or os.environ.get("MEMBER_CUBE_CACHE") == "0":
        return None
    cube_path, key_path = _paths(member, key)
    if not (cube_path.exists() and key_path.exists()):
        return None
    # A digest collision would be catastrophic and silent, so compare the key itself, not its hash.
    if json.loads(key_path.read_text()) != json.loads(json.dumps(key, default=str)):
        return None
    df = pd.read_parquet(cube_path)
    if member not in df.columns or df[member].isna().any():
        return None
    return df


def store(member: str, key: dict | None, cube: pd.DataFrame) -> Path | None:
    """Persist ``cube``. A ``None`` key means not cacheable — a no-op, not an error."""
    if key is None or os.environ.get("MEMBER_CUBE_CACHE") == "0":
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cube_path, key_path = _paths(member, key)
    # Write through a temp file: a cube truncated by a Ctrl-C would otherwise be read back as real.
    tmp = cube_path.with_suffix(".parquet.tmp")
    cube.to_parquet(tmp, index=False)
    tmp.replace(cube_path)
    key_path.write_text(json.dumps(key, indent=2, default=str) + "\n")
    return cube_path
