"""The two detectors: frozen in-loop, and independently-trained evaluation. Never conflated."""

from __future__ import annotations

from t2o.detection.evaluation import EvaluationDetector
from t2o.detection.frozen import FrozenDetector

__all__ = [
    "EvaluationDetector",
    "FrozenDetector",
]
