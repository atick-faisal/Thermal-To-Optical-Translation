"""`scripts/modtr_calibration.py` -- ModTr's inner split and the calibration run's verdicts.

The verdict reads a hand-built run directory: a `config.yaml` snapshot and a `monitor.jsonl`.
The rules themselves are tested in `test_epoch_budget.py`; this tests which run the script
agrees to read, and what it logs and returns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from scripts.modtr_calibration import main

from t2o.config import CONFIG_FILENAME, Config
from t2o.data.inner_split import MARKER_KEY
from t2o.data.manifest import DatasetManifest
from t2o.engine.trainer import MONITOR_FILENAME, EpochStats

EPOCHS = 20


@pytest.fixture(scope="module")
def inner_yaml(data_yaml: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("calibration") / "inner"
    assert main(["split", "--data", str(data_yaml), "--out", str(out), "--fraction", "0.4"]) == 0
    return out / "data.yaml"


def _run(
    root: Path,
    manifest: Path,
    map50: Sequence[float | None],
    loss_det: list[float],
    epochs: int = EPOCHS,
) -> Path:
    run = root / "run"
    Config.load(
        overrides={
            "data": {"manifest": str(manifest)},
            "train": {"epochs_per_stage": epochs},
            "coupling": {"task_weights": [1.0]},
        }
    ).snapshot(run)
    rows = [
        EpochStats(epoch=i, train_losses={"loss_det": loss}, val_loss=0.0, val_map50=value)
        for i, (value, loss) in enumerate(zip(map50, loss_det, strict=True))
    ]
    (run / "stage0").mkdir()
    (run / "stage0" / MONITOR_FILENAME).write_text(
        "".join(json.dumps(asdict(row)) + "\n" for row in rows)
    )
    return run


def _messages(caplog: pytest.LogCaptureFixture, prefix: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


FALLING = [5.0] * 10 + [4.0] * 10


def test_split_writes_a_loadable_inner_manifest(inner_yaml: Path) -> None:
    inner = DatasetManifest.load(inner_yaml)

    assert len(list(inner.val_images.iterdir())) == 2
    assert len(list(inner.train_images.iterdir())) == 3


def test_split_takes_the_cap_from_the_config(data_yaml: Path, tmp_path: Path) -> None:
    config = Config.load(overrides={"data": {"max_train_images": 4, "subset_seed": 2}})
    config.snapshot(tmp_path / "experiment")
    out = tmp_path / "inner"

    argv = ["split", "--data", str(data_yaml), "--out", str(out), "--fraction", "0.5"]
    assert main([*argv, "--config", str(tmp_path / "experiment" / CONFIG_FILENAME)]) == 0

    declared = yaml.safe_load((out / "data.yaml").read_text())[MARKER_KEY]
    assert (declared["max_images"], declared["subset_seed"]) == (4, 2)
    inner = DatasetManifest.load(out / "data.yaml")
    assert len(list(inner.train_images.iterdir())) + len(list(inner.val_images.iterdir())) == 4


def test_a_rising_curve_keeps_the_budget_and_a_falling_loss_passes(
    inner_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    run = _run(tmp_path, inner_yaml, [i / 100 for i in range(EPOCHS)], FALLING)

    assert main(["verdict", "--run", str(run)]) == 0
    (budget,) = _messages(caplog, "budget: ")
    assert budget.startswith(f"budget: {EPOCHS} epochs (trigger did not fire)")
    (loss,) = _messages(caplog, "loss: ")
    assert loss.startswith("loss: PASS")


def test_a_fall_after_a_peak_cuts_the_budget(
    inner_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Fractions in the file, points in the rule: 0.50 -> 0.40 is a 10-point drop.
    caplog.set_level(logging.INFO)
    run = _run(tmp_path, inner_yaml, [0.5] * 10 + [0.4] * 10, FALLING)

    assert main(["verdict", "--run", str(run)]) == 0
    (budget,) = _messages(caplog, "budget: ")
    assert budget.startswith("budget: 10 epochs (trigger fired)")


def test_a_loss_that_does_not_fall_fails_the_run(
    inner_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    run = _run(tmp_path, inner_yaml, [0.5] * EPOCHS, FALLING[::-1])

    assert main(["verdict", "--run", str(run)]) == 1
    (loss,) = _messages(caplog, "loss: ")
    assert loss.startswith("loss: FAIL")


def test_a_run_on_the_real_split_is_refused(data_yaml: Path, tmp_path: Path) -> None:
    run = _run(tmp_path, data_yaml, [0.5] * EPOCHS, FALLING)

    with pytest.raises(SystemExit, match="not an inner split"):
        main(["verdict", "--run", str(run)])


def test_a_run_still_in_flight_is_refused(inner_yaml: Path, tmp_path: Path) -> None:
    run = _run(tmp_path, inner_yaml, [0.5] * EPOCHS, FALLING, epochs=400)

    with pytest.raises(SystemExit, match="20 of 400 epochs"):
        main(["verdict", "--run", str(run)])


def test_a_run_without_the_monitor_is_refused(inner_yaml: Path, tmp_path: Path) -> None:
    run = _run(tmp_path, inner_yaml, [None] * EPOCHS, FALLING)

    with pytest.raises(SystemExit, match="--monitor-map"):
        main(["verdict", "--run", str(run)])
