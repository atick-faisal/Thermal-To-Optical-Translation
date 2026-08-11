from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from t2o.detection.frozen import FrozenDetector


@pytest.fixture
def detector(detector_weights: Path) -> FrozenDetector:
    return FrozenDetector(detector_weights, nc=4)


def test_detector_stays_in_eval_mode(detector: FrozenDetector) -> None:
    detector.train()  # what an enclosing trainer's model.train() would do
    assert not detector.training
    assert not detector.model.training


def test_non_divisible_input_is_rejected(detector: FrozenDetector) -> None:
    with pytest.raises(ValueError, match="divisible by stride 32"):
        detector(torch.rand(1, 3, 50, 64))


def test_class_count_mismatch_warns(
    detector_weights: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        FrozenDetector(detector_weights, nc=99)
    assert "mismatched class semantics" in caplog.text


def test_canary_detect_forward_returns_a_tuple_with_feats(detector: FrozenDetector) -> None:
    """Pins the ultralytics 8.4.x eval-mode Detect contract.

    If a future ultralytics upgrade changes this, this test is what fails -- loudly and
    with a pointer -- rather than the coupling loss (M0.7) silently computing garbage.
    """
    images = torch.rand(1, 3, 640, 640)
    output = detector(images)

    assert isinstance(output, tuple)
    assert len(output) == 2
    _, preds = output
    assert isinstance(preds, dict)
    assert set(preds) >= {"boxes", "scores", "feats"}
    for feat, stride in zip(preds["feats"], detector.strides, strict=True):
        h, w = feat.shape[-2:]
        assert (h * stride, w * stride) == images.shape[-2:]


def test_no_gradient_accumulates_on_detector_parameters(detector: FrozenDetector) -> None:
    images = torch.rand(1, 3, 640, 640, requires_grad=True)
    _, preds = detector(images)
    loss = preds["scores"].float().sum() + sum(f.float().sum() for f in preds["feats"])
    loss.backward()

    assert not any(p.requires_grad for p in detector.parameters())
    assert all(p.grad is None for p in detector.parameters())


def test_gradient_reaches_the_input_image(detector: FrozenDetector) -> None:
    images = torch.rand(1, 3, 640, 640, requires_grad=True)
    _, preds = detector(images)
    loss = preds["scores"].float().sum() + sum(f.float().sum() for f in preds["feats"])
    loss.backward()

    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
