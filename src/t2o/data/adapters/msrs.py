"""MSRS -> the internal ``{split}/{visible,infrared}/{images,labels}`` representation.

Raw layout (``github.com/Linfeng-Tang/MSRS``, verified against the real clone -- TASKS.md's
M0.9 note that no ``detection/`` folder exists was written from browsing GitHub without
cloning and was wrong)::

    train/vi/*.png, train/ir/*.png          # 1083 pairs, matching filenames, unlabelled
    test/vi/*.png,  test/ir/*.png           # 361 pairs, same shape, unlabelled
    detection/vi/*.png, detection/ir/*.png  # 80 pairs, disjoint filenames from train/test
    detection/labels/*.txt                  # YOLO cls/cx/cy/w/h
    detection/labels/classes.txt            # canonical class order

``detection/`` is its own small labelled pool, not a subset of ``train``/``test`` -- verified
disjoint by filename stem. It merges into ``train`` rather than becoming a third split, since
the internal representation only has two. MSRS has no dedicated val split; ``test`` stands in.
"""

from __future__ import annotations

import logging
from pathlib import Path

from t2o.data.adapters.common import (
    AdapterError,
    copy_image_pair,
    dest_already_populated,
    write_label,
    write_manifest_yaml,
)

logger = logging.getLogger(__name__)

CLASSES_FILENAME = "classes.txt"


def adapt_msrs(raw_root: Path, dest_root: Path) -> Path:
    """Convert a ``dataset/raw/msrs``-shaped tree into ``dest_root``. Returns the data.yaml path."""
    raw_root = Path(raw_root)
    dest_root = Path(dest_root)

    if dest_already_populated(dest_root):
        logger.info("%s already populated, skipping", dest_root)
        return dest_root / "data.yaml"

    names = _read_classes(raw_root / "detection" / "labels" / CLASSES_FILENAME)

    train_stems = _stems(raw_root / "train" / "vi")
    detection_stems = _stems(raw_root / "detection" / "vi")
    collisions = train_stems & detection_stems
    if collisions:
        preview = ", ".join(sorted(collisions)[:5])
        raise AdapterError(
            f"{len(collisions)} filename(s) appear in both train/ and detection/: {preview}"
            f"{'...' if len(collisions) > 5 else ''}. Merging them into one train split would "
            f"silently overwrite images."
        )

    train_root = dest_root / "train"
    for stem in sorted(train_stems):
        copy_image_pair(
            raw_root / "train" / "vi" / f"{stem}.png",
            raw_root / "train" / "ir" / f"{stem}.png",
            stem,
            train_root,
            ".png",
        )
    for stem in sorted(detection_stems):
        copy_image_pair(
            raw_root / "detection" / "vi" / f"{stem}.png",
            raw_root / "detection" / "ir" / f"{stem}.png",
            stem,
            train_root,
            ".png",
        )
        write_label(train_root, stem, raw_root / "detection" / "labels" / f"{stem}.txt")

    val_root = dest_root / "val"
    for stem in sorted(_stems(raw_root / "test" / "vi")):
        copy_image_pair(
            raw_root / "test" / "vi" / f"{stem}.png",
            raw_root / "test" / "ir" / f"{stem}.png",
            stem,
            val_root,
            ".png",
        )

    logger.info(
        "msrs: %d train pairs (%d labelled), %d val pairs -> %s",
        len(train_stems) + len(detection_stems),
        len(detection_stems),
        len(_stems(raw_root / "test" / "vi")),
        dest_root,
    )
    return write_manifest_yaml(dest_root, names)


def _stems(images_dir: Path) -> set[str]:
    if not images_dir.is_dir():
        raise AdapterError(f"expected a directory at {images_dir}")
    return {p.stem for p in images_dir.iterdir() if p.is_file()}


def _read_classes(classes_path: Path) -> list[str]:
    if not classes_path.is_file():
        raise AdapterError(f"{classes_path} not found -- cannot determine class names")
    names = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    if not names:
        raise AdapterError(f"{classes_path} is empty")
    return names
