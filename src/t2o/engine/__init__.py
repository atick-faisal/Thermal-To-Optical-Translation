"""Training orchestration: stage loop, trainer, export, and per-stage detector fine-tuning."""

from __future__ import annotations

from t2o.engine.detector_stage import DetectorResult, train_detector, wandb_integration_disabled
from t2o.engine.export import export_split, export_translated
from t2o.engine.trainer import EpochStats, Trainer

__all__ = [
    "DetectorResult",
    "EpochStats",
    "Trainer",
    "export_split",
    "export_translated",
    "train_detector",
    "wandb_integration_disabled",
]
