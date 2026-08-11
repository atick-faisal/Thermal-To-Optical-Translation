"""M0.7: the staged lambda_det schedule and its "no detector at weight 0" guarantee."""

from __future__ import annotations

from pathlib import Path

import pytest

from t2o.config.schema import CouplingConfig
from t2o.coupling.detection_loss import DetectionTaskLoss
from t2o.coupling.schedule import build_detection_loss, weight_for_stage


def test_weight_for_stage_indexes_the_ramp() -> None:
    config = CouplingConfig(task_weights=(0.0, 1.0, 2.0, 3.0))
    assert [weight_for_stage(config, stage) for stage in range(4)] == [0.0, 1.0, 2.0, 3.0]


def test_zero_weight_constructs_no_detector(
    detector_weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("FrozenDetector must not be constructed at weight 0")

    monkeypatch.setattr("t2o.coupling.schedule.FrozenDetector", _fail)

    config = CouplingConfig(task_weights=(0.0,))
    result = build_detection_loss(config, stage=0, detector_weights=detector_weights, nc=4)

    assert result is None


def test_missing_detector_weights_constructs_no_detector() -> None:
    config = CouplingConfig(task_weights=(1.0,))
    result = build_detection_loss(config, stage=0, detector_weights=None, nc=4)

    assert result is None


def test_positive_weight_builds_detection_loss(detector_weights: Path) -> None:
    config = CouplingConfig(task_weights=(1.0,))
    result = build_detection_loss(config, stage=0, detector_weights=detector_weights, nc=4)

    assert isinstance(result, DetectionTaskLoss)
