"""Fidelity, task, and faithfulness metrics -- the one evaluation path (PLAN.md invariant 1).

Only fidelity exists so far (M0.5 step 1). Task (mAP) and faithfulness (C2) land in
later steps.
"""

from __future__ import annotations

from t2o.metrics.fidelity import FidelityEvaluator, FidelityMetrics

__all__ = [
    "FidelityEvaluator",
    "FidelityMetrics",
]
