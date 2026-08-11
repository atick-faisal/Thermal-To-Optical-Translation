"""Translated-tensor to uint8 image conversion.

Ported unchanged from ``../Clean-SeAFusion/src/seafusion/imaging.py``. Shared by the
export stage, which writes translated images to disk for the evaluation detector, and by
:mod:`t2o.config`, which exposes ``Normalize`` as a config choice.

The invariant worth keeping in sight: normalisation is **per-image, never batch-wise**.
The original SeAFusion normalised across a whole batch (``train.py:369``), which made every
exported pixel depend on which other images happened to share its batch -- and therefore
made the detector's training data depend on batch size and shuffle order.
"""

from __future__ import annotations

from enum import StrEnum

import torch
from torch import Tensor


class Normalize(StrEnum):
    """How a translated float image is mapped onto ``[0, 255]``.

    ``CLAMP``
        Clamp to ``[0, 1]``. Preserves absolute intensity, so exported images are
        comparable across a dataset. The sane default for feeding a detector.
    ``PER_IMAGE``
        Min-max stretch each image independently. Raises apparent contrast on flat
        frames, at the cost of making intensities incomparable between images.
    """

    CLAMP = "clamp"
    PER_IMAGE = "per_image"


def to_uint8(image: Tensor, normalize: Normalize = Normalize.CLAMP) -> Tensor:
    """Convert a single ``(C, H, W)`` float image to ``(C, H, W)`` uint8."""
    if image.ndim != 3:
        raise ValueError(f"expected a (C, H, W) image, got {tuple(image.shape)}")

    image = image.detach().float().cpu()
    if normalize is Normalize.PER_IMAGE:
        low, high = image.min(), image.max()
        span = high - low
        # A constant image has no range to stretch; the original divided by zero here.
        image = (image - low) / span if span > 0 else torch.zeros_like(image)
    else:
        image = image.clamp(0.0, 1.0)

    return (image * 255.0).clamp(0.0, 255.0).to(torch.uint8)
