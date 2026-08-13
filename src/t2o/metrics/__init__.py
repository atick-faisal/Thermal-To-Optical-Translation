"""Fidelity, task, and faithfulness metrics -- the one evaluation path (PLAN.md invariant 1)."""

from __future__ import annotations

from t2o.metrics.faithfulness import (
    Detections,
    FaithfulnessEvaluator,
    FaithfulnessMetrics,
    detections_from_labels,
    detections_from_result,
)
from t2o.metrics.fidelity import (
    FidelityError,
    FidelityEvaluator,
    FidelityMetrics,
    evaluate_fidelity,
)
from t2o.metrics.task import TaskMetrics, evaluate_detector

__all__ = [
    "Detections",
    "FaithfulnessEvaluator",
    "FaithfulnessMetrics",
    "FidelityError",
    "FidelityEvaluator",
    "FidelityMetrics",
    "TaskMetrics",
    "detections_from_labels",
    "detections_from_result",
    "evaluate_detector",
    "evaluate_fidelity",
]
