"""A typed handle onto the independently-trained evaluation detector.

No equivalent exists in ``../Clean-SeAFusion`` -- that repo alternates a single detector
between frozen and trainable roles across stages. PLAN.md invariant 7 ("two detectors,
never conflated") splits the roles structurally here instead: :class:`FrozenDetector`
(``detection/frozen.py``) always stays frozen and supplies the in-loop task gradient;
this class only ever names a checkpoint produced by
:func:`t2o.engine.detector_stage.train_detector`, which trains via ultralytics' own
``model.train()`` and therefore never touches the translator's autograd graph.

Deliberately has no base class or code in common with ``FrozenDetector``, so nothing here
can be handed to the coupling loss by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationDetector:
    """A trained checkpoint used only to score translated images, never to train on them."""

    weights: Path

    @classmethod
    def from_checkpoint(cls, weights: Path | str) -> EvaluationDetector:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"evaluation detector checkpoint not found: {path}")
        return cls(weights=path)

    def load(self) -> Any:
        """Load the checkpoint as an ``ultralytics.YOLO`` for validation/inference."""
        from ultralytics import YOLO

        return YOLO(str(self.weights))
