"""`engine/loop.py`'s staged alternation of translator training and detector fine-tune.

Built on the synthetic `data_yaml`/`detector_weights` fixtures (`tests/conftest.py`). The
fast test exercises the translator-only path (`train_detector_stages=False`) but still
wires a real `FrozenDetector` through the coupling loss on the stage with a positive
`task_weight` -- constructing that detector is cheap (one model load, no training), unlike
`train_detector`'s full ultralytics training loop. The slow test is the first end-to-end
exercise of `train_detector` itself, deferred from M0.4 to this milestone (TASKS.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from t2o.config import Config
from t2o.engine.loop import run_loop
from t2o.translators import StubTranslator


def _config(data_yaml: Path, detector_weights: Path) -> Config:
    return Config.load(
        overrides={
            "data": {"manifest": str(data_yaml)},
            "train": {"batch_size": 2, "workers": 0, "epochs_per_stage": 1},
            "coupling": {"task_weights": [0.0, 1.0]},
            "detector": {
                "in_loop": {"weights": str(detector_weights)},
                "evaluation": {"init_weights": str(detector_weights), "epochs": 1, "batch": 2},
            },
            "runtime": {"device": "cpu"},
        }
    )


def test_translator_only_loop_runs_every_stage_and_wires_the_coupling_loss(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    config = _config(data_yaml, detector_weights)
    translator = StubTranslator(hidden_channels=4)
    before = next(translator.parameters()).clone()

    results = run_loop(config, translator, run_dir=tmp_path / "run", train_detector_stages=False)

    assert [r.stage for r in results] == [0, 1]
    assert [r.task_weight for r in results] == [0.0, 1.0]
    assert all(r.detector is None for r in results)
    assert not torch.equal(before, next(translator.parameters()))

    # task_weights[0] == 0 must stay a clean no-op (M0.7's contract, exercised end-to-end
    # here): no coupling term at all, so no "loss_det" key. Stage 1's positive weight must
    # actually reach StubTranslator.fit() and add one.
    assert "loss_det" not in results[0].epochs[-1].train_losses
    assert "loss_det" in results[1].epochs[-1].train_losses

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(metrics) == 2
    assert [m["task_weight"] for m in metrics] == [0.0, 1.0]
    assert all(m["detector"] is None for m in metrics)


@pytest.mark.slow
def test_full_loop_fine_tunes_a_detector_every_stage(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    config = _config(data_yaml, detector_weights)
    translator = StubTranslator(hidden_channels=4)

    results = run_loop(config, translator, run_dir=tmp_path / "run")

    assert len(results) == 2
    for result in results:
        assert result.detector is not None
        assert result.detector.weights.is_file()
        assert 0.0 <= result.detector.map50 <= 1.0

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(metrics) == 2
    assert metrics[0]["detector"]["weights"]
    assert metrics[1]["detector"]["weights"]
