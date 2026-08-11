"""Paired visible/infrared dataset with an ultralytics-compatible collate.

Augmentation is geometric only. Photometric augmentation is deliberately unavailable:
the translator is scored against the real visible frame by L2 and LPIPS
(``config.LossConfig``), so any brightness/contrast jitter would move the target those
losses score against and turn a supervision signal into noise.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from t2o.data.labels import load_yolo_labels
from t2o.data.pairing import Pairing

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

# Boxes surviving a crop with less than this normalised side are dropped: a sliver of a
# box is a worse training target than no box at all.
_MIN_BOX_SIDE = 1e-3


@dataclass(frozen=True, slots=True)
class TranslationSample:
    """One paired sample. ``cls``/``bboxes`` carry zero rows for unlabelled images."""

    visible: Tensor  # (3, H, W) float in [0, 1]
    infrared: Tensor  # (1, H, W) float in [0, 1]
    cls: Tensor  # (M, 1) float
    bboxes: Tensor  # (M, 4) normalised xywh
    name: str


class TranslationBatch(TypedDict):
    """Collated batch. The label keys are exactly what ``v8DetectionLoss`` consumes."""

    visible: Tensor  # (B, 3, H, W)
    infrared: Tensor  # (B, 1, H, W)
    batch_idx: Tensor  # (M,)
    cls: Tensor  # (M, 1)
    bboxes: Tensor  # (M, 4)
    names: list[str]


def _to_tensor(path: Path, mode: str) -> Tensor:
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _flip_bboxes(bboxes: Tensor) -> Tensor:
    if bboxes.numel() == 0:
        return bboxes
    flipped = bboxes.clone()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    return flipped


def _crop_bboxes(
    bboxes: Tensor,
    source: tuple[int, int],
    window: tuple[int, int, int, int],
) -> tuple[Tensor, Tensor]:
    """Clip normalised xywh boxes to a crop window and renormalise onto it.

    ``source`` is ``(height, width)`` of the uncropped image; ``window`` is
    ``(top, left, height, width)`` in source pixels. Returns the surviving boxes and the
    boolean keep-mask, so the caller can filter ``cls`` identically.
    """
    if bboxes.numel() == 0:
        return bboxes, torch.zeros(0, dtype=torch.bool)

    source_h, source_w = source
    top, left, crop_h, crop_w = window

    cx, cy, w, h = bboxes.unbind(dim=1)
    x1 = (cx - w / 2) * source_w - left
    y1 = (cy - h / 2) * source_h - top
    x2 = (cx + w / 2) * source_w - left
    y2 = (cy + h / 2) * source_h - top

    x1, x2 = x1.clamp(0, crop_w), x2.clamp(0, crop_w)
    y1, y2 = y1.clamp(0, crop_h), y2.clamp(0, crop_h)

    new_w, new_h = (x2 - x1) / crop_w, (y2 - y1) / crop_h
    keep = (new_w > _MIN_BOX_SIDE) & (new_h > _MIN_BOX_SIDE)

    cropped = torch.stack(
        (
            (x1 + x2) / 2 / crop_w,
            (y1 + y2) / 2 / crop_h,
            new_w,
            new_h,
        ),
        dim=1,
    )
    return cropped[keep], keep


def _annotated_subset(paths: list[Path], fraction: float, seed: int) -> frozenset[Path]:
    """Deterministically choose which paths keep their annotations (E8 sweep).

    Every image still trains the translator's reconstruction losses -- the visible target
    is always present -- so this only decides which images additionally supply detection
    supervision. Shuffling a copy of the (already sorted) path list under a seeded RNG
    keeps the choice reproducible across processes without depending on filesystem order.
    """
    if fraction >= 1.0:
        return frozenset(paths)
    ordered = list(paths)
    random.Random(seed).shuffle(ordered)
    keep = round(fraction * len(ordered))
    return frozenset(ordered[:keep])


class TranslationPairDataset(Dataset[TranslationSample]):
    """Visible/infrared pairs from a directory of visible images.

    Takes the images directory itself, which is what a ``data.yaml`` declares for each
    split. Pairing is by filename via :class:`Pairing`, never by sorted index.
    """

    def __init__(
        self,
        images_dir: Path | str,
        pairing: Pairing | None = None,
        hflip: float = 0.0,
        crop: tuple[int, int] | None = None,
        num_classes: int | None = None,
        annotation_fraction: float = 1.0,
        annotation_seed: int = 0,
    ) -> None:
        self.pairing = pairing or Pairing()
        self.hflip = hflip
        self.crop = crop

        self.images_dir = Path(images_dir)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"visible image directory not found: {self.images_dir}")

        self.visible_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.visible_paths:
            raise FileNotFoundError(f"no images found in {self.images_dir}")

        self.pairing.validate_pairs(self.visible_paths)
        if num_classes is not None:
            self._validate_class_ids(num_classes)

        if not 0.0 < annotation_fraction <= 1.0:
            raise ValueError(f"annotation_fraction must be in (0, 1], got {annotation_fraction}")
        self._annotated = _annotated_subset(
            self.visible_paths, annotation_fraction, annotation_seed
        )

        logger.info(
            "%s: %d paired samples, %d annotated",
            self.images_dir,
            len(self.visible_paths),
            len(self._annotated),
        )

    def _validate_class_ids(self, num_classes: int) -> None:
        """Refuse to start if any label names a class the manifest does not declare.

        Left unchecked an out-of-range id reaches ``v8DetectionLoss``, which indexes a
        class logit that does not exist -- surfacing on CUDA as an opaque device-side
        assert far into a run rather than as anything about classes.
        """
        offenders: dict[int, str] = {}
        for visible_path in self.visible_paths:
            cls, _ = load_yolo_labels(self.pairing.label_path(visible_path))
            for value in cls.reshape(-1).tolist():
                if not 0 <= int(value) < num_classes:
                    offenders.setdefault(int(value), visible_path.name)

        if offenders:
            detail = ", ".join(
                f"class {cid} (e.g. {name})" for cid, name in sorted(offenders.items())
            )
            raise ValueError(
                f"{self.images_dir}: labels use class ids outside the manifest's "
                f"nc={num_classes}: {detail}. Fix data.yaml or the labels -- the two "
                f"must agree."
            )

    def __len__(self) -> int:
        return len(self.visible_paths)

    def __getitem__(self, index: int) -> TranslationSample:
        visible_path = self.visible_paths[index]
        visible = _to_tensor(visible_path, "RGB")
        infrared = _to_tensor(self.pairing.infrared_path(visible_path), "L")

        if visible.shape[1:] != infrared.shape[1:]:
            raise ValueError(
                f"{visible_path.name}: visible {tuple(visible.shape[1:])} and infrared "
                f"{tuple(infrared.shape[1:])} differ in size"
            )

        if visible_path in self._annotated:
            cls, bboxes = load_yolo_labels(self.pairing.label_path(visible_path))
        else:
            cls, bboxes = torch.zeros(0, 1), torch.zeros(0, 4)

        if self.crop is not None:
            visible, infrared, cls, bboxes = self._apply_crop(visible, infrared, cls, bboxes)
        if self.hflip > 0 and torch.rand(()).item() < self.hflip:
            visible = torch.flip(visible, dims=(2,))
            infrared = torch.flip(infrared, dims=(2,))
            bboxes = _flip_bboxes(bboxes)

        return TranslationSample(
            visible=visible,
            infrared=infrared,
            cls=cls,
            bboxes=bboxes,
            name=visible_path.name,
        )

    def _apply_crop(
        self, visible: Tensor, infrared: Tensor, cls: Tensor, bboxes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        assert self.crop is not None
        source_h, source_w = visible.shape[1], visible.shape[2]
        crop_h, crop_w = min(self.crop[0], source_h), min(self.crop[1], source_w)

        top = int(torch.randint(0, source_h - crop_h + 1, ()).item())
        left = int(torch.randint(0, source_w - crop_w + 1, ()).item())

        bboxes, keep = _crop_bboxes(bboxes, (source_h, source_w), (top, left, crop_h, crop_w))
        cls = cls[keep] if keep.numel() else cls

        return (
            visible[:, top : top + crop_h, left : left + crop_w],
            infrared[:, top : top + crop_h, left : left + crop_w],
            cls,
            bboxes,
        )


def collate_translation_batch(samples: list[TranslationSample]) -> TranslationBatch:
    """Stack samples into the layout ``v8DetectionLoss`` reads.

    It consumes only ``batch_idx``, ``cls`` and ``bboxes``, deriving image size from the
    predictions rather than from any image tensor -- so the translated image never needs
    to be threaded into the batch dict.
    """
    batch_idx = torch.cat(
        [torch.full((s.cls.shape[0],), i, dtype=torch.float32) for i, s in enumerate(samples)]
    )
    return TranslationBatch(
        visible=torch.stack([s.visible for s in samples]),
        infrared=torch.stack([s.infrared for s in samples]),
        batch_idx=batch_idx,
        cls=torch.cat([s.cls for s in samples]),
        bboxes=torch.cat([s.bboxes for s in samples]),
        names=[s.name for s in samples],
    )
