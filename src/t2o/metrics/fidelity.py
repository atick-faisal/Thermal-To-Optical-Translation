"""Image-fidelity metrics: PSNR, SSIM, LPIPS, FID, KID.

Nothing here is a port -- no PSNR/SSIM/LPIPS/FID/KID exists in any sibling repo
(PLAN.md §6). Reported but explicitly argued as insufficient on their own (PLAN.md §12):
PSNR/SSIM reward blur, and FID/KID use ImageNet backbones insensitive to domain-specific
structure and unreliable on small sets. The task (mAP) and faithfulness (C2) metrics are
what the paper's claim actually rests on; this module exists so fidelity is reported
honestly alongside them, not so it can carry the argument alone.

All five are computed through ``torchmetrics.image``, which accepts float ``[0, 1]``
input directly (``data_range=1.0`` for PSNR/SSIM, ``normalize=True`` for LPIPS/FID/KID) --
exactly ``TranslationSample``'s native convention. No manual uint8 conversion happens
here; ``t2o.imaging.to_uint8`` stays scoped to the export stage, which is a different
concern (what the evaluation detector reads off disk).

PSNR/SSIM/LPIPS score image pairs; FID/KID are distributional and need features
accumulated over an entire real-image pool and fake-image pool before one ``compute()``.
:class:`FidelityEvaluator` wraps all five uniformly behind ``update``/``compute``/``reset``
so every translator feeds its exported batches through the same accumulator (PLAN.md
invariant 1: one evaluation path).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, cast

from torch import Tensor
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

logger = logging.getLogger(__name__)

# Images arrive as TranslationSample's native convention throughout t2o.
_DATA_RANGE = 1.0


@dataclass(frozen=True, slots=True)
class FidelityMetrics:
    """One evaluation's worth of fidelity numbers, all on the same image pool."""

    psnr: float
    ssim: float
    lpips: float
    fid: float
    kid_mean: float
    kid_std: float


def _clamp_kid_subset_size(configured: int, available: int) -> int:
    """KID's own default (1000) raises if fewer images are available than that.

    Dataset size varies by more than two orders of magnitude between the synthetic smoke
    fixture (a handful of images) and the ~850-pair custom dataset, so the configured
    value must be clamped to what is actually on hand rather than trusting the caller (or
    the library default) to have sized it correctly.
    """
    if configured <= available:
        return configured
    logger.warning(
        "kid_subset_size=%d exceeds the %d images accumulated; clamping to %d",
        configured,
        available,
        available,
    )
    return available


class FidelityEvaluator:
    """Accumulates PSNR/SSIM/LPIPS/FID/KID over a translated-image pool.

    Construct once per evaluation run -- building the LPIPS and Inception backbones is the
    expensive part. ``update`` is called once per batch; ``compute`` reports the pooled
    result. Both LPIPS and FID/KID need pretrained ImageNet backbones (AlexNet /
    InceptionV3) fetched over the network the first time they run, then cached under
    ``~/.cache/torch/`` -- there is no way around that cost for these particular metrics.
    """

    def __init__(
        self,
        lpips_net: Literal["alex", "vgg", "squeeze"] = "alex",
        kid_subset_size: int = 50,
        device: str | None = None,
    ) -> None:
        self._kid_subset_size = kid_subset_size
        self._n_images = 0

        self._psnr = PeakSignalNoiseRatio(data_range=_DATA_RANGE)
        self._ssim = StructuralSimilarityIndexMeasure(data_range=_DATA_RANGE)
        self._lpips = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net, normalize=True)
        self._fid = FrechetInceptionDistance(normalize=True)
        # subset_size is finalised in compute(), once the real image count is known;
        # torchmetrics reads it at compute time, not construction time, so reassigning the
        # attribute here is enough to apply a later clamp.
        self._kid = KernelInceptionDistance(subset_size=kid_subset_size, normalize=True)

        if device is not None:
            self._psnr = self._psnr.to(device)
            self._ssim = self._ssim.to(device)
            self._lpips = self._lpips.to(device)
            self._fid = self._fid.to(device)
            self._kid = self._kid.to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Accumulate one batch. Both tensors are ``(B, 3, H, W)`` float in ``[0, 1]``."""
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
        if pred.ndim != 4:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(pred.shape)}")

        self._psnr.update(pred, target)
        self._ssim.update(pred, target)
        self._lpips.update(pred, target)
        self._fid.update(target, real=True)
        self._fid.update(pred, real=False)
        self._kid.update(target, real=True)
        self._kid.update(pred, real=False)
        self._n_images += pred.shape[0]

    def compute(self) -> FidelityMetrics:
        self._kid.subset_size = _clamp_kid_subset_size(self._kid_subset_size, self._n_images)

        kid_mean, kid_std = self._kid.compute()
        # SSIM.compute()'s static return type includes the return_full_image /
        # return_contrast_sensitivity tuple overloads; neither flag is set here, so this is
        # always the plain-Tensor case.
        ssim = cast(Tensor, self._ssim.compute())
        return FidelityMetrics(
            psnr=float(self._psnr.compute()),
            ssim=float(ssim),
            lpips=float(self._lpips.compute()),
            fid=float(self._fid.compute()),
            kid_mean=float(kid_mean),
            kid_std=float(kid_std),
        )

    def reset(self) -> None:
        self._n_images = 0
        self._psnr.reset()
        self._ssim.reset()
        self._lpips.reset()
        self._fid.reset()
        self._kid.reset()
