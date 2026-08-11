"""Training orchestration: stage loop, trainer, export, and per-stage detector fine-tuning."""

from __future__ import annotations

from t2o.engine.detector_stage import DetectorResult, train_detector, wandb_integration_disabled

__all__ = [
    "DetectorResult",
    "train_detector",
    "wandb_integration_disabled",
]
