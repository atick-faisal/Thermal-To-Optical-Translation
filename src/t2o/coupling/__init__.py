"""Detection-consistency coupling: the loss and its staged lambda_det schedule."""

from __future__ import annotations

from t2o.coupling.detection_loss import DetectionTaskLoss, TaskLossOutput
from t2o.coupling.schedule import build_detection_loss, weight_for_stage

__all__ = [
    "DetectionTaskLoss",
    "TaskLossOutput",
    "build_detection_loss",
    "weight_for_stage",
]
