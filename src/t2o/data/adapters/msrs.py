"""MSRS -> the internal ``{split}/{visible,infrared}/{images,labels}`` representation.

Raw layout (``github.com/Linfeng-Tang/MSRS``, verified against the real clone -- TASKS.md's
M0.9 note that no ``detection/`` folder exists was written from browsing GitHub without
cloning and was wrong)::

    train/vi/*.png, train/ir/*.png          # 1083 pairs, matching filenames
    train/Segmentation_labels/*.png         # one 480x640 uint8 class-index mask per pair
    test/vi/*.png,  test/ir/*.png           # 361 pairs, same shape
    test/Segmentation_labels/*.png
    detection/                              # 80 pairs + boxes -- NOT read, see below

``detection/`` is deliberately not read. Its filenames are disjoint from ``train/`` and
``test/``, but its pixels are not: 71 of the 80 pairs are pixel-identical (mean absolute
error 0.0) to MSRS frames, and 23 of those 71 are ``test`` frames. Merging it into ``train``
therefore put labelled copies of val frames in the train split. The 9 pairs with no twin carry
only ``person, bicycle, car`` labels, so merged in they would teach a detector that unlabelled
cones and car stops are background (docs/features/msrs-passive-gate/plan.md, Q4).

Train is therefore the 1083 ``train/`` frames and val is the 361 ``test`` frames (MSRS has no
dedicated val split). Every frame is labelled from its segmentation mask by
:func:`mask_to_yolo_lines`, so the class names are ``KEPT_CLASSES``, not ``detection/``'s
``classes.txt``. A mask with no kept-class blob writes no label file: the frame is a negative
(44 train / 13 val real frames, plan.md Tech Stack / Approach).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from t2o.data.adapters.common import (
    AdapterError,
    copy_image_pair,
    dest_already_populated,
    write_label_lines,
    write_manifest_yaml,
)

logger = logging.getLogger(__name__)

MASKS_DIRNAME = "Segmentation_labels"

# The classes kept as detection targets, in output-id order (plan.md Q2), written to data.yaml
# as the names. A reordering silently relabels every box already on disk.
KEPT_CLASSES = ("car", "person", "bike", "car_stop", "color_cone")

# Segmentation_labels pixel value -> output class id. The mask order is car, person, bike,
# curve, car_stop, guardrail, color_cone, bump (dataset/raw/msrs/visualize.py:10-33). Values
# 4, 6 and 8 are absent on purpose: too rare to score, or long strips whose boxes mostly cover
# background (plan.md Q2). A value missing here is dropped, never passed through as `value - 1`.
MASK_TO_CLASS = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4}

# Minimum blob area in pixels (plan.md Q3's measured table). Blobs of 1-9 px are annotation
# crumbs, almost all warm-class fragments; 10-29 px car stops and cones are real distant
# objects. Raising this drops the passive classes under test, so it fails silently.
MIN_BLOB_AREA = 10


def adapt_msrs(raw_root: Path, dest_root: Path) -> Path:
    """Convert a ``dataset/raw/msrs``-shaped tree into ``dest_root``. Returns the data.yaml path."""
    raw_root = Path(raw_root)
    dest_root = Path(dest_root)

    if dest_already_populated(dest_root):
        logger.info("%s already populated, skipping", dest_root)
        return dest_root / "data.yaml"

    # MSRS has no val split; its test split is ours (module docstring).
    splits = (("train", "train"), ("test", "val"))
    # Checked before any copy: a half-written tree would be skipped as populated on the rerun.
    for raw_split, _ in splits:
        _require_masks(raw_root / raw_split)

    for raw_split, split in splits:
        frames, labelled, boxes = _adapt_split(raw_root / raw_split, dest_root / split)
        logger.info(
            "msrs %s: %d pairs, %d labelled, %d boxes -> %s",
            split,
            frames,
            labelled,
            boxes,
            dest_root / split,
        )
    return write_manifest_yaml(dest_root, list(KEPT_CLASSES))


def mask_to_yolo_lines(mask: np.ndarray) -> list[str]:
    """Turn one ``Segmentation_labels`` mask into normalised YOLO ``cls cx cy w h`` lines.

    Each kept class contributes one box per 8-connected blob of at least ``MIN_BLOB_AREA``
    pixels. Touching objects of one class merge into one box; that is accepted (plan.md Q3).
    """
    height, width = mask.shape
    lines: list[str] = []
    for mask_value, class_id in MASK_TO_CLASS.items():
        binary = (mask == mask_value).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for index in range(1, count):  # label 0 is the background
            x, y, w, h, area = (int(v) for v in stats[index])
            if area < MIN_BLOB_AREA:
                continue
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    return lines


def _adapt_split(raw_split: Path, split_root: Path) -> tuple[int, int, int]:
    """Copy one raw split's pairs and write their mask-derived labels.

    Returns the frame, labelled-frame and box counts for the log line.
    """
    stems = sorted(_stems(raw_split / "vi"))
    labelled = boxes = 0
    for stem in stems:
        copy_image_pair(
            raw_split / "vi" / f"{stem}.png",
            raw_split / "ir" / f"{stem}.png",
            stem,
            split_root,
            ".png",
        )
        mask = cv2.imread(str(raw_split / MASKS_DIRNAME / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        lines = mask_to_yolo_lines(np.asarray(mask))
        write_label_lines(split_root, stem, lines)
        labelled += bool(lines)
        boxes += len(lines)
    return len(stems), labelled, boxes


def _require_masks(raw_split: Path) -> None:
    """Raise unless every visible frame in ``raw_split`` has its segmentation mask.

    A frame without one would otherwise pass as a negative and silently erase its objects.
    """
    missing = sorted(
        stem
        for stem in _stems(raw_split / "vi")
        if not (raw_split / MASKS_DIRNAME / f"{stem}.png").is_file()
    )
    if missing:
        raise AdapterError(
            f"{len(missing)} frame(s) in {raw_split} have no {MASKS_DIRNAME} mask: {missing[:5]}"
        )


def _stems(images_dir: Path) -> set[str]:
    if not images_dir.is_dir():
        raise AdapterError(f"expected a directory at {images_dir}")
    return {p.stem for p in images_dir.iterdir() if p.is_file()}
