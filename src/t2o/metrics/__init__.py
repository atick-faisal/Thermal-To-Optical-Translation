"""Fidelity, task, and faithfulness metrics -- the one evaluation path (PLAN.md invariant 1).

Fidelity and task exist so far (M0.5 steps 1-2). Faithfulness (C2) lands in a later step.
"""

from __future__ import annotations

from t2o.metrics.fidelity import FidelityEvaluator, FidelityMetrics
from t2o.metrics.task import TaskMetrics, evaluate_detector

__all__ = [
    "FidelityEvaluator",
    "FidelityMetrics",
    "TaskMetrics",
    "evaluate_detector",
]
