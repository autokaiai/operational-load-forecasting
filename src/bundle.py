"""Self-contained checkpoint: a single ``checkpoint.pt`` holding everything inference needs.

Layout (a zip archive named ``checkpoint.pt``):
- ``nf/`` — ``NeuralForecast.save(save_dataset=True)`` output: model weights, scalers, and the
  per-series last-window target history the model conditions on.
- ``bundle.json`` — sidecar: imputation fill stats, exog/column lists, model name, seed, and the
  neuralforecast version used (asserted on load).

This makes the same code path serve validation and the private test: ``predict.py`` reads
covariates from ``--input_dir`` and the history from this bundle — never from ``train.csv``.
The archive is our own trusted, offline artifact, so loading it (full unpickle via
``NeuralForecast.load``) is safe.
"""

from __future__ import annotations

import json
import tempfile
import warnings
import zipfile
from pathlib import Path

import neuralforecast
from neuralforecast import NeuralForecast

from src.data.features import CASCADE_FORECASTS_ALL, stat_exog_list

SIDECAR_NAME = "bundle.json"
TREE_NAME = "tree.json"
NF_SUBDIR = "nf"


def model_futr_exog(nf: NeuralForecast) -> list[str]:
    """The future-covariate columns the restored models ACTUALLY require.

    **This, not the sidecar, is the authoritative list**, and the difference is not academic: the
    shipped `cascade_bag5.pt` records **29** columns in its sidecar while its models want **31**.
    `save` used to call `src.data.features.futr_exog_list()`, whose result depends on which cascade
    channels happen to be active *in the saving process* — and the S6.5 submission runner saved
    outside that scope, so the sidecar quietly described a different model from the one beside it.

    A value read off the weights cannot drift from the weights. Every model in a bag is trained on
    one frame, so disagreement between them is a corrupt bundle rather than a case to reconcile.
    """
    lists = [list(getattr(m, "futr_exog_list", None) or []) for m in nf.models]
    if not lists:
        return []
    first = lists[0]
    for other in lists[1:]:
        if other != first:
            raise ValueError(
                "models in this bundle disagree about their future covariates "
                f"({first} vs {other}); the bag is not a single model."
            )
    return first


def cascade_channels_of(futr_cols) -> list[str]:
    """Which cascade channels a futr list implies — derived, never declared.

    `predict.py` must reactivate exactly the channels the model was trained with, and the compact
    note's instruction to read them from the sidecar's `config` is not available: the shipped
    checkpoint has no `cascade` key at all. Intersecting the model's own column list with the
    registry is the read that works on checkpoints already written, including that one.
    """
    have = set(futr_cols)
    return [c for c in CASCADE_FORECASTS_ALL if c in have]


def aggregate_columns_of(futr_cols) -> list[str]:
    """Which DERIVED cross-series aggregates a futr list implies — derived, never declared.

    Same philosophy as `cascade_channels_of` and for the same reason: `predict.py` has to rebuild
    exactly the conditioning set the model was fitted on, and the only thing that can be trusted on
    an already-written checkpoint is the model's own column list. So an aggregate is defined here by
    ELIMINATION — anything in the futr list that is not a calendar encoding, a known-future signal,
    a cascade channel, or one of their `*_missing` twins.

    Defining it that way rather than by an `xs_` prefix means a renamed block cannot quietly stop
    being recorded, and an unrecognised column can never be silently dropped from the submission's
    feature set: it is either a known base column or it is an aggregate to be rebuilt.
    """
    from src.data.features import KNOWN_FUTURE_SIGNALS, MISSING_SUFFIX, NAN_COLS, TIME_ENCODINGS

    base = {*TIME_ENCODINGS, *KNOWN_FUTURE_SIGNALS, *CASCADE_FORECASTS_ALL, *NAN_COLS}
    base |= {f"{c}{MISSING_SUFFIX}" for c in base}
    return [c for c in futr_cols if c not in base]


def save(
    nf: NeuralForecast,
    fill_stats: dict,
    cfg: dict,
    out_path: str | Path,
    tree: dict | None = None,
    blend: dict | None = None,
) -> Path:
    """Bundle a fitted NeuralForecast + sidecar into a single zipped ``checkpoint.pt``.

    ``tree`` is the persisted LightGBM member (``scripts.fit_submission_tree``) and ``blend`` its
    weight. Both are optional so every existing caller and every existing checkpoint keeps working;
    when absent, inference is the neural bag alone.

    The tree has to travel *inside* the archive because ``predict.py`` runs against an input dir of
    covariates and a forecast index with **no training data** — there is nothing to refit from at
    inference, so a member that is not in the box cannot be in the blend.
    """
    out_path = Path(out_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nf.save(str(tmp / NF_SUBDIR), overwrite=True, save_dataset=True)
        futr = model_futr_exog(nf)
        sidecar = {
            "model": cfg["model"],
            "seed": int(cfg.get("seed", 42)),
            "fill_stats": fill_stats,
            # Read off the models, NOT off the ambient cascade selection — see model_futr_exog.
            "futr_exog_list": futr,
            "cascade_channels": cascade_channels_of(futr),
            "aggregate_columns": aggregate_columns_of(futr),
            "stat_exog_list": stat_exog_list(),
            "neuralforecast_version": neuralforecast.__version__,
            "config": {k: v for k, v in cfg.items() if k != "loss"},
            "blend": dict(blend) if blend else None,
        }
        (tmp / SIDECAR_NAME).write_text(json.dumps(sidecar, indent=2))
        if tree is not None:
            # Kept beside the sidecar rather than inside it: the serialised booster is a large
            # opaque string and the sidecar is meant to stay readable by eye.
            (tmp / TREE_NAME).write_text(json.dumps(tree))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(tmp.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(tmp))
    return out_path


def load(checkpoint_path: str | Path) -> tuple[NeuralForecast, dict]:
    """Unzip the bundle and restore (NeuralForecast, sidecar). Offline; no train.csv needed."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    tmp = Path(tempfile.mkdtemp())  # persists for the process lifetime (CLI), OS cleans up
    with zipfile.ZipFile(checkpoint_path) as zf:
        zf.extractall(tmp)

    sidecar = json.loads((tmp / SIDECAR_NAME).read_text())
    tree_path = tmp / TREE_NAME
    sidecar["tree"] = json.loads(tree_path.read_text()) if tree_path.exists() else None
    saved_ver = sidecar.get("neuralforecast_version")
    if saved_ver and saved_ver != neuralforecast.__version__:
        warnings.warn(
            f"checkpoint trained with neuralforecast {saved_ver}, "
            f"loading with {neuralforecast.__version__}; behaviour may differ.",
            stacklevel=2,
        )
    nf = NeuralForecast.load(str(tmp / NF_SUBDIR))

    # Reconcile the sidecar against the weights, and let the weights win. Bundles written before
    # this function existed record the ambient futr list rather than the model's, so a checkpoint
    # that is perfectly good can carry a sidecar that is wrong about it — repairing here means
    # `predict.py` never has to know which vintage it was handed.
    futr = model_futr_exog(nf)
    if futr and sidecar.get("futr_exog_list") != futr:
        missing = sorted(set(futr) - set(sidecar.get("futr_exog_list") or []))
        warnings.warn(
            f"sidecar lists {len(sidecar.get('futr_exog_list') or [])} future covariates but the "
            f"models require {len(futr)} (absent from the sidecar: {missing}); "
            "trusting the models.",
            stacklevel=2,
        )
    if futr:
        sidecar["futr_exog_list"] = futr
    sidecar["cascade_channels"] = cascade_channels_of(futr)
    sidecar["aggregate_columns"] = aggregate_columns_of(futr)
    return nf, sidecar
