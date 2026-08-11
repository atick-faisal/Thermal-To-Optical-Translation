"""Detection-consistency coupling loss: scores a translated image with a frozen detector.

Ported from ``../Clean-SeAFusion/src/seafusion/losses/task.py``. That version recombines a
fused luma plane with the visible image's chroma (``ycbcr_to_rgb``) before scoring, because
its fusion network only predicts luma. t2o's translator emits RGB directly
(``Translator.translate() -> Tensor`` in ``[0,1]``), so that recombination step -- and the
``cb``/``cr`` parameters that existed only to feed it -- is dropped entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from t2o.data.dataset import TranslationBatch
from t2o.detection.frozen import FrozenDetector


@dataclass(frozen=True, slots=True)
class TaskLossOutput:
    """Unweighted, unscaled component breakdown, for logging."""

    total: Tensor
    box: Tensor
    cls: Tensor
    dfl: Tensor


class DetectionTaskLoss(nn.Module):
    """Scores a translated RGB image with a frozen detector and backprops into the translator.

    ``grad_scale`` and ``reward_target`` are the anti-reward-hacking guardrails
    ``CouplingConfig`` promises (PLAN.md §8): an aggressive constant downscale on the
    detection gradient, and a saturating-reward hinge that stops rewarding a sample once
    its detection loss is already below the target -- rather than driving it toward zero
    unboundedly, which is what invites the detector to find adversarial textures.
    """

    def __init__(
        self,
        detector: FrozenDetector,
        grad_scale: float = 1.0,
        reward_target: float | None = None,
    ) -> None:
        super().__init__()
        from ultralytics.utils.loss import v8DetectionLoss

        self.detector = detector
        self.criterion = v8DetectionLoss(detector.model)
        self.grad_scale = grad_scale
        self.reward_target = reward_target

    def compute(self, rgb: Tensor, batch: TranslationBatch) -> TaskLossOutput:
        # rgb is fed straight to the detector as float32 [0,1] -- never routed through
        # ultralytics' own preprocess_batch (a .float()/255 meant for a uint8 dataloader
        # tensor), which would create a fresh graph root and sever gradients back to the
        # translator.
        predictions = self.detector(rgb)

        # v8DetectionLoss reads only these three keys and derives image size from the
        # predictions themselves, so the translated image never has to be threaded into
        # the targets dict.
        targets = {
            "batch_idx": batch["batch_idx"],
            "cls": batch["cls"],
            "bboxes": batch["bboxes"],
        }
        components, _ = self.criterion(predictions, targets)

        # ultralytics returns the loss pre-multiplied by batch size. Dividing it back out
        # keeps a stage's lambda_det invariant to batch size -- otherwise doubling the
        # batch would silently double the task term's influence.
        batch_size = rgb.shape[0]
        box, cls, dfl = (components / batch_size).unbind()

        return TaskLossOutput(total=box + cls + dfl, box=box, cls=cls, dfl=dfl)

    def forward(self, rgb: Tensor, batch: TranslationBatch) -> Tensor:
        total = self.compute(rgb, batch).total
        if self.reward_target is not None:
            # ReFL-style hinge: zero gradient once the sample is already at or below the
            # target, rather than AlignProp's symmetric |r - target| that would also
            # penalise a detection loss that is already better than the target.
            total = F.relu(total - self.reward_target)
        return self.grad_scale * total
