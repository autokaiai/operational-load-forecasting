"""Tests for the inference contract.

Two layers:
- dependency-free unit tests of predict.py's pure helpers (forecast-index loading, positional
  prediction assignment) — always run in CI without the heavy forecasting stack.
- an end-to-end CLI test that trains a tiny real model, saves a checkpoint bundle, runs
  ``predict.py`` exactly as the harness would, and asserts the output schema + row coverage.
  Gated behind ``importorskip("neuralforecast")`` so the fast CI job stays green.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import predict

REPO_ROOT = Path(__file__).resolve().parent.parent
SERIES = ["unit_000", "unit_001"]
STEPS = 5


def _write_forecast_index(
    input_dir: Path, start: str | pd.Timestamp = "2023-06-30 00:00:00", steps: int = STEPS
) -> pd.DataFrame:
    """Write a forecast index of ``steps`` hours per series beginning at ``start``.

    ``start`` is a parameter because it is the whole variable in play. Before the #32 fix this
    fixture hardcoded ``2023-06-30`` while the CLI test trained on ``2023-01-01 + 120h`` — a
    ~4200h gap — and the test PASSED, because positional matching zipped January predictions onto
    June labels and the schema, the row count and the finiteness checks were all still satisfied.
    **The end-to-end test was demonstrating the defect rather than catching it.**
    """
    ts = pd.date_range(pd.Timestamp(start), periods=steps, freq="h")
    rows = [
        {"series_id": sid, "timestamp": t.strftime("%Y-%m-%d %H:%M:%S")}
        for sid in SERIES
        for t in ts
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(input_dir / "forecast_index_validation.csv", index=False)
    return frame


def test_load_forecast_index(tmp_path: Path) -> None:
    """Loads forecast_index_validation.csv with the expected columns."""
    fi = _write_forecast_index(tmp_path)
    loaded = predict.load_forecast_index(tmp_path)
    assert list(loaded.columns) == ["series_id", "timestamp"]
    assert len(loaded) == len(fi)


def test_assign_predictions_maps_by_series_in_order() -> None:
    """Positional assignment matches each series' chronological predictions, row order kept."""
    forecast_index = pd.DataFrame(
        {
            "series_id": ["unit_000", "unit_001", "unit_000", "unit_001"],
            "timestamp": [
                "2023-06-30 00:00:00",
                "2023-06-30 00:00:00",
                "2023-06-30 01:00:00",
                "2023-06-30 01:00:00",
            ],
        }
    )
    # neuralforecast-style output (shuffled), distinct values per (series, time).
    preds = pd.DataFrame(
        {
            "unique_id": ["unit_001", "unit_000", "unit_001", "unit_000"],
            "ds": pd.to_datetime(
                [
                    "2023-06-30 01:00:00",
                    "2023-06-30 00:00:00",
                    "2023-06-30 00:00:00",
                    "2023-06-30 01:00:00",
                ]
            ),
            "MyModel": [11.0, 0.0, 10.0, 1.0],
        }
    )
    out = predict._assign_predictions(forecast_index, preds, "MyModel")
    # Row order preserved; values correspond to the (series, timestamp) of each index row.
    assert list(out) == [0.0, 10.0, 1.0, 11.0]


def test_predict_cli_contract(tmp_path: Path) -> None:
    """Full CLI: tiny real bundle -> predict.py -> exact schema, one row per index row, finite."""
    pytest.importorskip("neuralforecast")

    import numpy as np
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import DLinear

    from src import bundle

    rng = np.random.default_rng(0)
    rows = []
    for sid in SERIES:
        ds = pd.date_range("2023-01-01", periods=120, freq="h")
        y = 10 + 3 * np.sin(np.arange(120) * 2 * np.pi / 24) + rng.normal(0, 0.3, 120)
        rows.append(pd.DataFrame({"unique_id": sid, "ds": ds, "y": y}))
    long_df = pd.concat(rows, ignore_index=True)

    nf = NeuralForecast(
        models=[
            DLinear(
                h=STEPS,
                input_size=24,
                max_steps=1,
                loss=MAE(),
                enable_progress_bar=False,
                logger=False,
                accelerator="cpu",
                devices=1,
            )
        ],
        freq="h",
    )
    nf.fit(long_df)

    checkpoint = tmp_path / "checkpoint.pt"
    bundle.save(nf, fill_stats={}, cfg={"model": "DLinear", "seed": 42}, out_path=checkpoint)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Contiguous with the training data: the honest validation case, gap = 0. Anchoring the index
    # to the data (rather than to a hardcoded date) is what makes this test able to fail.
    forecast_index = _write_forecast_index(
        input_dir, start=long_df["ds"].max() + pd.Timedelta(hours=1)
    )
    output_file = tmp_path / "out" / "predictions.csv"

    result = subprocess.run(
        [
            sys.executable,
            "predict.py",
            "--input_dir",
            str(input_dir),
            "--output_file",
            str(output_file),
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    predictions = pd.read_csv(output_file)
    assert list(predictions.columns) == ["series_id", "timestamp", "prediction"]
    assert len(predictions) == len(forecast_index)
    assert not predictions.isna().any().any()
    assert np.isfinite(predictions["prediction"]).all()


def _bag_frame() -> pd.DataFrame:
    """A two-series, two-hour forecast from a THREE-seed bag, one column per seed."""
    idx = [
        (sid, t)
        for sid in SERIES
        for t in pd.date_range("2023-06-30 00:00:00", periods=2, freq="h")
    ]
    return pd.DataFrame(
        {
            "unique_id": [s for s, _ in idx],
            "ds": [t for _, t in idx],
            "TFT": [1.0, 1.0, 1.0, 1.0],
            "TFT1": [4.0, 4.0, 4.0, 4.0],
            "TFT2": [7.0, 7.0, 7.0, 7.0],
        }
    )


def test_bag_columns_finds_every_seed_in_order() -> None:
    """The bag is `TFT, TFT1, ... TFTN`; reading only `TFT` ships one seed out of five."""
    assert predict.bag_columns(_bag_frame(), "TFT") == ["TFT", "TFT1", "TFT2"]
    # single-model bundles still work, and a foreign model's columns are not swept up
    one = pd.DataFrame({"unique_id": ["a"], "ds": [1], "TFT": [1.0], "LSTM": [2.0]})
    assert predict.bag_columns(one, "TFT") == ["TFT"]
    assert predict.bag_columns(one, "LSTM") == ["LSTM"]


def test_bag_columns_excludes_quantile_columns() -> None:
    """A distributional head emits `TFT-lo-90` / `TFT-median`; averaging a level into a point
    forecast would be silent rather than loud, so the `\\d*$` anchor has to keep them out."""
    frame = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": [1],
            "TFT": [1.0],
            "TFT1": [2.0],
            "TFT-lo-90": [0.5],
            "TFT-median": [1.5],
            "TFTX": [9.0],
        }
    )
    assert predict.bag_columns(frame, "TFT") == ["TFT", "TFT1"]


def test_the_bag_is_averaged_not_sampled() -> None:
    """THE REGRESSION GUARD (the 13.483 -> 15.794 defect).

    `predict.py` selected `sidecar["model"]` and discarded the other four seeds. On the real
    artifact that is a member mean of 10.0345 against the bag's 10.6458 — a plausible-looking
    forecast of the right hours in the right schema, and 2.3 WAPE points worse. This asserts the
    mean, so taking any single column fails it.
    """
    preds = _bag_frame()
    seeds = predict.bag_columns(preds, "TFT")
    averaged = preds.assign(**{"TFT": preds[seeds].mean(axis=1)})

    forecast_index = preds.rename(columns={"unique_id": "series_id", "ds": "timestamp"})[
        ["series_id", "timestamp"]
    ]
    values = predict.assign_predictions(forecast_index, averaged, "TFT")

    assert values.tolist() == [4.0, 4.0, 4.0, 4.0]  # mean(1, 4, 7), NOT TFT's 1.0


def test_cli_ships_the_whole_bag_not_its_first_seed(tmp_path: Path) -> None:
    """END-TO-END REGRESSION GUARD for the 13.483 -> 15.794 defect.

    The unit tests above pin the arithmetic; this pins the WIRING, which is where the defect
    actually lived — `main()` passed `sidecar["model"]` straight to `assign_predictions`, so a
    5-seed bundle shipped seed 0 and threw away the other four. Nothing in the output could show
    it: right schema, right hours, right row count, finite values, plausible magnitudes.

    Runs the real CLI against a real 3-seed bundle and asserts the CSV equals the MEAN of the
    three columns. Taking any single column fails it.
    """
    pytest.importorskip("neuralforecast")

    import numpy as np
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import DLinear

    from src import bundle

    rng = np.random.default_rng(0)
    rows = []
    for sid in SERIES:
        ds = pd.date_range("2023-01-01", periods=120, freq="h")
        y = 10 + 3 * np.sin(np.arange(120) * 2 * np.pi / 24) + rng.normal(0, 0.3, 120)
        rows.append(pd.DataFrame({"unique_id": sid, "ds": ds, "y": y}))
    long_df = pd.concat(rows, ignore_index=True)

    nf = NeuralForecast(
        models=[
            DLinear(
                h=STEPS,
                input_size=24,
                max_steps=1,
                loss=MAE(),
                random_seed=seed,
                enable_progress_bar=False,
                logger=False,
                accelerator="cpu",
                devices=1,
            )
            for seed in (892, 7739, 6545)
        ],
        freq="h",
    )
    nf.fit(long_df)

    checkpoint = tmp_path / "checkpoint.pt"
    bundle.save(nf, fill_stats={}, cfg={"model": "DLinear", "seed": 892}, out_path=checkpoint)

    expected = nf.predict()
    if "unique_id" not in expected.columns:
        expected = expected.reset_index()
    seeds = predict.bag_columns(expected, "DLinear")
    assert seeds == ["DLinear", "DLinear1", "DLinear2"]
    # The fixture must be ABLE to fail: three identical models would make the mean equal the
    # first column and the assertion below vacuous.
    assert not np.allclose(expected["DLinear"], expected["DLinear1"]), "seeds did not diverge"

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_forecast_index(input_dir, start=long_df["ds"].max() + pd.Timedelta(hours=1))
    output_file = tmp_path / "out" / "predictions.csv"

    result = subprocess.run(
        [
            sys.executable,
            "predict.py",
            "--input_dir",
            str(input_dir),
            "--output_file",
            str(output_file),
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "bag of 3" in result.stdout, result.stdout

    out = pd.read_csv(output_file)
    got = out["prediction"].to_numpy()

    # The CLI de-smooths its output at every rung (`predict.DESMOOTH_GAMMA`), so the CSV is no
    # longer the bare bag mean. Mirror the transform rather than relax the check — what this test
    # exists to catch is shipping ONE SEED instead of the bag, and that property is untouched: the
    # recalibration is a per-unit affine map, so it cannot turn one seed's forecast into three.
    # `.clip(lower=0)` because predict.py clamps: the target is strictly positive and the scorer
    # hard-rejects negatives. A 1-step DLinear does emit some, so the clamp is live here.
    import predict as _p

    def _desmooth(v):
        s_ = pd.Series(v.to_numpy() if hasattr(v, "to_numpy") else v)
        anchor = s_.groupby(out["series_id"].to_numpy()).transform("mean")
        return (anchor + _p.DESMOOTH_GAMMA * (s_ - anchor)).clip(lower=0.0).to_numpy()

    want = _desmooth(expected[seeds].mean(axis=1))
    first_only = _desmooth(expected["DLinear"])

    np.testing.assert_allclose(got, want, rtol=1e-5)
    assert not np.allclose(got, first_only), "the CLI shipped one seed instead of the bag"
