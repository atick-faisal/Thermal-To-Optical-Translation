"""M0.7: the detection-consistency coupling loss.

Uses the same offline `detector_weights` fixture as `test_frozen_detector.py` -- no
network access. Batches are hand-built (same style as `test_stub_translator.py`'s
`_batch`), not routed through the real dataset/collate path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from t2o.coupling.detection_loss import DetectionTaskLoss
from t2o.data.dataset import TranslationBatch
from t2o.detection.frozen import FrozenDetector


def _rgb(n: int, size: int = 640, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((n, 3, size, size), generator=generator, requires_grad=True)


def _targets(n: int) -> TranslationBatch:
    """One box per image, repeated identically -- what batch-size-invariance needs."""
    return TranslationBatch(
        visible=torch.zeros(0),
        infrared=torch.zeros(0),
        batch_idx=torch.arange(n, dtype=torch.float32),
        cls=torch.zeros((n, 1)),
        bboxes=torch.tensor([[0.5, 0.5, 0.4, 0.4]] * n),
        names=[f"sample_{i}" for i in range(n)],
    )


@pytest.fixture
def detector(detector_weights: Path) -> FrozenDetector:
    return FrozenDetector(detector_weights, nc=4)


def test_loss_is_differentiable_wrt_generated_image(detector: FrozenDetector) -> None:
    """Pins that DetectionTaskLoss never severs the graph back to the translated image.

    If a future ultralytics upgrade changes v8DetectionLoss's targets contract or its
    batch-size premultiplication in a way that breaks this, this test is what fails.
    """
    rgb = _rgb(n=1)
    task_loss = DetectionTaskLoss(detector)

    loss = task_loss(rgb, _targets(1))
    loss.backward()

    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()
    assert rgb.grad.abs().sum() > 0


def test_batch_size_invariance(detector: FrozenDetector) -> None:
    task_loss = DetectionTaskLoss(detector)

    rgb1 = _rgb(n=1, seed=0)
    total1 = task_loss.compute(rgb1, _targets(1)).total

    rgb4 = rgb1.detach().repeat(4, 1, 1, 1).requires_grad_(True)
    total4 = task_loss.compute(rgb4, _targets(4)).total

    assert torch.allclose(total1, total4, atol=1e-4)


def test_reward_target_saturates(detector: FrozenDetector) -> None:
    rgb = _rgb(n=1)
    unbounded = DetectionTaskLoss(detector).compute(rgb, _targets(1)).total

    saturated_loss = DetectionTaskLoss(detector, reward_target=unbounded.detach().item() + 10.0)
    loss = saturated_loss(rgb, _targets(1))
    loss.backward()

    assert loss.detach().item() == 0.0
    assert rgb.grad is not None
    assert torch.all(rgb.grad == 0.0)


def test_to_rebuilds_the_criterion_so_its_internal_buffers_follow_the_move(
    detector: FrozenDetector, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: `v8DetectionLoss` is a plain object, not an `nn.Module`, so its own
    device-pinned buffers (`self.proj`, `self.bbox_loss`, `self.class_weights`) are never
    touched by `nn.Module.to()`'s normal submodule recursion -- they stay wherever the
    detector was when `DetectionTaskLoss` was constructed. This is what let a real CUDA
    server run crash inside `v8DetectionLoss.bbox_decode` even after `Trainer` correctly
    moved the detector's own parameters: `self.criterion` must be rebuilt on `.to()` so it
    re-reads the detector's now-current device, not just moved wholesale.
    """
    from ultralytics.utils import loss as ultra_loss

    calls: list[torch.nn.Module] = []
    original_cls = ultra_loss.v8DetectionLoss

    class RecordingLoss(original_cls):  # type: ignore[misc]
        def __init__(self, model: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
            calls.append(model)
            super().__init__(model, *args, **kwargs)

    monkeypatch.setattr(ultra_loss, "v8DetectionLoss", RecordingLoss)

    task_loss = DetectionTaskLoss(detector)
    assert len(calls) == 1

    task_loss.to(torch.device("cpu"))
    assert len(calls) == 2


def test_grad_scale_scales_the_gradient(detector: FrozenDetector) -> None:
    rgb_full = _rgb(n=1)
    DetectionTaskLoss(detector, grad_scale=1.0)(rgb_full, _targets(1)).backward()
    full_grad = rgb_full.grad
    assert full_grad is not None

    rgb_half = rgb_full.detach().clone().requires_grad_(True)
    DetectionTaskLoss(detector, grad_scale=0.5)(rgb_half, _targets(1)).backward()
    half_grad = rgb_half.grad
    assert half_grad is not None

    assert torch.allclose(half_grad, full_grad * 0.5, atol=1e-6)
