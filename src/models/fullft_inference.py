"""The full fine-tuned Chronos-2 MEMBER, fetched and run at inference time.

What this is, and what it is deliberately not
---------------------------------------------
`chronos_full_ft` is a **blend member**, never a covariate. That is not a stylistic choice: the
standing architectural rule is that *a model fitted on our data enters only as a blend member*,
because an OOF forecast for hour ``t`` produced by a model trained on our targets can encode
``target[t]``. Zero-shot Chronos-2 sits on the permitted side of that line and is why
``chronos2_forecast`` may be a cascade channel; this model does not. Pinned by a test asserting it
appears in no ``futr_exog_list()``.

Why it reuses the cascade generator
-----------------------------------
`cascade_inference.generate_channel` is weight-agnostic — it takes a resolved ``source`` and calls
``Chronos2Pipeline.from_pretrained`` on it — and a full-FT checkpoint is a plain
``model.safetensors`` + ``config.json``, not a PEFT overlay, so it loads through that identical
path. The member and the cascade channel therefore compute the SAME quantity (a Chronos forecast of
the target over exactly the future frame's rows, anchored at the end of the observed history) from
DIFFERENT weights. Reusing the generator means the honest-by-construction context anchoring is
shared rather than reimplemented, and the ship path this module extends stays untouched.

Why the fetch may fail, and what happens then
---------------------------------------------
455.8 MiB — full FT rewrites every weight, so the checkpoint is the whole model against LoRA's
4.6 MiB adapter. It cannot go in the 200 MB archive, so it is hosted and fetched, which is the
route the 2026-06-07 clarification sanctioned. An unreachable fetch degrades to the arm below with
its OWN measured weights — see ``predict.main``. It must never be papered over by renormalising the
three-member weights onto two members: that would be a model no measurement describes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.models.cascade_inference import _log as _cascade_log
from src.models.cascade_inference import generate_channel

#: The member's column name, matching the CV cubes it was measured on.
FULLFT_COL = "chronos_full_ft"

#: Where a bundled copy would live inside an unpacked checkpoint. Absent by default (456 MB does
#: not fit the archive) but tried first, so a fully-offline variant needs a bigger box and no code.
BUNDLED_WEIGHTS_SUBDIR = "chronos2_fullft"

#: The hosted repo, resolved at import time; ``FULLFT_REPO_ID`` overrides it for a rehearsal.
#: An empty value resolves to ``None`` — i.e. the degraded arm below — rather than to a guess.
#: A wrong repo id would fetch SOMEONE ELSE'S weights and produce a confident, plausible, wrong
#: forecast, so the lookup fails closed instead of falling back to a default namespace.
FULLFT_REPO = os.environ.get("FULLFT_REPO_ID", "autokai/chronos2-fullft-dlam-g100")

#: Pin the revision, for the reason `cascade_inference.PINNED_REVISION` documents: an unpinned repo
#: resolves whatever `main` points at on the day, and a changed member would alter the graded run
#: with no symptom at all. Pinned to the upload commit of 2026-08-28. Unlike Chronos-2's pin (added
#: retroactively, and harmless because both snapshots turned out to be the same blob), this one has
#: been in place since the repo's first and only commit.
FULLFT_REVISION = os.environ.get("FULLFT_REVISION", "ff0fda7e0c5f67581c693975763a988b8c8ed4e3")

#: Escape hatch mirroring ``DISABLE_CASCADE`` — take the arm below deliberately, for the clean-room
#: test that rung 2 still reproduces exactly.
_DISABLE_ENV = "DISABLE_FULLFT"


def _log(msg: str) -> None:
    print(f"[fullft] {msg}", flush=True)


def fullft_enabled() -> bool:
    """False when ``DISABLE_FULLFT`` is set to anything truthy."""
    return os.environ.get(_DISABLE_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}


def resolve_source(bundle_dir: Path | None = None, *, allow_download: bool = True) -> str | None:
    """Locate the full-FT weights: bundled -> local HF cache -> hub. ``None`` if unreachable.

    Every step is announced. The difference between this arm and the one below it is ~0.0044 pooled
    WAPE, and that must never be a silent property of the machine the harness happened to run on.
    """
    if bundle_dir is not None:
        local = Path(bundle_dir) / BUNDLED_WEIGHTS_SUBDIR
        if (local / "config.json").exists():
            _log(f"weights: bundled copy at {local}")
            return str(local)

    if not FULLFT_REPO:
        _log("weights: no repo configured (FULLFT_REPO_ID unset) — taking the arm below")
        return None

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _log("weights: huggingface_hub not installed; cannot resolve the full-FT member")
        return None

    rev = FULLFT_REVISION or None
    shown = (FULLFT_REVISION or "default")[:8]

    # The cache first, and explicitly offline, so a machine that already holds the weights never
    # depends on the network being up.
    try:
        path = snapshot_download(FULLFT_REPO, revision=rev, local_files_only=True)
        _log(f"weights: local HF cache at {path} (rev {shown})")
        return path
    except Exception:
        pass

    if not allow_download:
        _log("weights: not cached and downloads disabled")
        return None

    try:
        path = snapshot_download(FULLFT_REPO, revision=rev)
        _log(f"weights: downloaded to {path} (rev {shown})")
        return path
    except Exception as exc:  # offline sandbox, hub outage, rate limit, private repo without token
        _log(f"weights: download failed ({type(exc).__name__}: {exc})")
        return None


def generate_member(
    nf,
    futr_df: pd.DataFrame,
    *,
    source: str,
    device: str = "cuda",
    batch_series: int = 0,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast the target over exactly ``futr_df``'s rows using the full-FT weights.

    Returns ``[unique_id, ds, chronos_full_ft]``. Thin by design: the anchoring, the covariate
    conditioning and the non-negativity clip are the cascade generator's, so the member cannot
    silently drift from the construction its CV numbers were measured under.
    """
    from src.models.cascade_inference import CHRONOS_COL

    _cascade_log(f"(full-FT member) using weights at {source}")
    out = generate_channel(
        nf, futr_df, source=source, device=device, batch_series=batch_series, history=history
    )
    return out.rename(columns={CHRONOS_COL: FULLFT_COL})
