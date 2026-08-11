"""A frozen ultralytics detector, used only to supply a task gradient to the translator.

Ported verbatim from ``../Clean-SeAFusion/src/seafusion/models/detector.py``. No behaviour
changes -- ``weights``/``nc`` are passed in directly rather than read off a config object,
so this file has nothing to adapt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _load_detection_model(weights: Path) -> nn.Module:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import unwrap_model

    model = YOLO(str(weights)).model
    if not isinstance(model, nn.Module):
        raise TypeError(f"{weights}: did not load a detection model (got {type(model).__name__})")
    # Unwrapped because v8DetectionLoss reaches into `model.model[-1]` for the Detect head
    # and would otherwise find a DataParallel/DDP wrapper. Named `de_parallel` up to
    # ~ultralytics 8.4.108 and `unwrap_model` from 8.4.112; this project's floor is 8.4.108
    # so the rename was verified against the installed 8.4.117 rather than assumed.
    return unwrap_model(model)


def _normalize_args(args: Any) -> Any:
    """Ensure ``model.args`` supports attribute access to the box/cls/dfl gains.

    ``v8DetectionLoss`` reads ``model.args.box`` and friends. Depending on how the
    checkpoint was produced, ``args`` arrives as ultralytics' own namespace, as a plain
    dict of saved ``train_args``, or missing entirely; only the first works as-is.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import IterableSimpleNamespace

    defaults = get_cfg()
    if args is None:
        return defaults
    if isinstance(args, dict):
        # Saved train_args win where present, so a detector trained with custom loss
        # gains keeps them; the defaults only fill the gaps.
        return IterableSimpleNamespace(**(vars(defaults) | args))
    return args


class FrozenDetector(nn.Module):
    """Wraps a pretrained ``DetectionModel`` with its parameters and BN statistics frozen.

    Kept in ``eval()`` mode deliberately, for two reasons:

    * BatchNorm uses its running statistics rather than the fused batch's, so the task
      gradient does not shift with whatever images happen to share a batch -- and the
      detector's own statistics are not silently rewritten by images it never trained on.
    * The loss still works: ``Detect.forward`` returns ``(y, preds)`` outside training,
      and ``v8DetectionLoss.parse_output`` unwraps exactly that tuple
      (``utils/loss.py:470``). Training mode would give a marginally cheaper forward pass
      but at the cost of corrupting the frozen statistics.

    Gradients flow *through* the network to the input image; nothing accumulates on the
    detector's own parameters.
    """

    def __init__(self, weights: Path | str, nc: int | None = None) -> None:
        super().__init__()
        self.weights = Path(weights)
        self.model = _load_detection_model(self.weights)
        self.model.eval()
        self.model.requires_grad_(False)

        head: Any = self.model.model[-1]  # type: ignore[index]
        self.nc = int(head.nc)
        self.strides = [int(s) for s in self.model.stride.tolist()]  # type: ignore[operator]
        # Inputs must be divisible by the coarsest level; feature maps come back at each.
        self.stride = max(self.strides)

        if nc is not None and nc != self.nc:
            # Not fatal: class ids below the detector's nc still index a valid logit, so
            # the loss runs. But the semantics are wrong, and mAP against these labels is
            # meaningless until the head is retrained.
            logger.warning(
                "%s has %d classes but the dataset declares %d; the task gradient will be "
                "computed against mismatched class semantics until the detector is "
                "retrained on this data",
                self.weights.name,
                self.nc,
                nc,
            )

        self.model.args = _normalize_args(getattr(self.model, "args", None))  # type: ignore[assignment]

    def train(self, mode: bool = True) -> FrozenDetector:
        """Ignore ``.train()``: this module is frozen by construction.

        Without this, the enclosing trainer's ``model.train()`` would quietly put the
        detector back into batch-statistics mode.
        """
        super().train(False)
        return self

    def forward(self, images: Tensor) -> Any:
        if images.shape[-1] % self.stride or images.shape[-2] % self.stride:
            raise ValueError(
                f"detector input must be divisible by stride {self.stride}, got "
                f"{tuple(images.shape[-2:])}"
            )
        return self.model(images)

    @torch.no_grad()
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
