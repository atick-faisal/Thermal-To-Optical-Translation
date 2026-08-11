from __future__ import annotations

from pathlib import Path

import pytest

from t2o.detection.evaluation import EvaluationDetector
from t2o.detection.frozen import FrozenDetector


def test_raises_on_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        EvaluationDetector.from_checkpoint(tmp_path / "missing.pt")


def test_returns_a_handle_on_an_existing_checkpoint(detector_weights: Path) -> None:
    handle = EvaluationDetector.from_checkpoint(detector_weights)
    assert handle.weights == detector_weights


def test_shares_no_base_class_with_frozen_detector() -> None:
    # Invariant 7 ("two detectors, never conflated") holds structurally: nothing that is
    # an EvaluationDetector can be handed anywhere an nn.Module (e.g. FrozenDetector) is
    # expected, and vice versa.
    assert not issubclass(EvaluationDetector, FrozenDetector)
    assert not issubclass(FrozenDetector, EvaluationDetector)
    assert set(EvaluationDetector.__mro__) & set(FrozenDetector.__mro__) == {object}
