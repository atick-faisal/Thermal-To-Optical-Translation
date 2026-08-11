"""Staged lambda_det ramp: which stage gets what weight, and whether it needs a detector.

Pulled out of Clean-SeAFusion's inline stage loop (``engine/loop.py``) into standalone
functions so the construct-or-not decision is testable without a trainer -- t2o's
``engine/loop.py`` (M0.8) doesn't exist yet and shouldn't have to own this by itself.
"""

from __future__ import annotations

from pathlib import Path

from t2o.config.schema import CouplingConfig
from t2o.coupling.detection_loss import DetectionTaskLoss
from t2o.detection.frozen import FrozenDetector


def weight_for_stage(config: CouplingConfig, stage: int) -> float:
    return config.task_weights[stage]


def build_detection_loss(
    config: CouplingConfig,
    stage: int,
    detector_weights: Path | None,
    nc: int,
) -> DetectionTaskLoss | None:
    """None when the stage's weight is <= 0 or no detector weights are configured yet.

    In either case the frozen detector is never constructed at all -- the clean no-op path
    that makes weight-0 (E3's ablation) a genuine control rather than merely a zeroed term.
    """
    weight = weight_for_stage(config, stage)
    if weight <= 0.0 or detector_weights is None:
        return None

    detector = FrozenDetector(detector_weights, nc=nc)
    return DetectionTaskLoss(
        detector, grad_scale=config.grad_scale, reward_target=config.reward_target
    )
