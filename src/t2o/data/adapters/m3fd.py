"""M3FD -> the internal ``{split}/{visible,infrared}/{images,labels}`` representation.

``dataset/raw/m3fd/`` is TarDAL's shared Drive folder (fetched by ``fetch_datasets.py``),
which bundles four zips. Only one is adapted::

    M3FD_Detection.zip   # adapted -- 4200 pairs, VOC-XML boxes
      Vis/NNNNN.png        # 4200, RGB
      Ir/NNNNN.png         # 4200, same stems and sizes as Vis/
      Annotation/NNNNN.xml # 4200, every image has at least one box
    M3FD_Fusion.zip      # skipped -- 300 unlabelled pairs whose stems are a subset of Detection's
    tno.zip              # skipped -- 37 unlabelled pairs, grayscale visible (no colour target)
    roadscene.zip        # skipped -- 42 unlabelled pairs, inconsistent ``meta/`` splits

Verified against the real archive: stems match across all three folders, and the XML
``<size>`` matches the images. The six classes (``Bus``, ``Car``, ``Lamp``, ``Motorcycle``,
``People``, ``Truck``) are collected from the data and sorted alphabetically, as FLIR's are.

M3FD ships **no train/val split** -- the XML ``<folder>``/``<path>`` fields record the
annotators' working directories, not a split. The frames are video sequences in stem order, so
neighbouring stems are near-duplicates and a per-frame shuffle would leak them into val.
:func:`assign_splits` instead cuts the sorted stems into contiguous blocks and sends whole
blocks to val. The result is frozen and hashed in ``splits/m3fd.json`` like every other split.
"""

from __future__ import annotations

import logging
import random
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from t2o.data.adapters.common import (
    AdapterError,
    VocBox,
    dest_already_populated,
    read_voc_box,
    voc_to_yolo_lines,
    write_image_pair_bytes,
    write_label_lines,
    write_manifest_yaml,
)

logger = logging.getLogger(__name__)

ARCHIVE_NAME = "M3FD_Detection.zip"
VISIBLE_PREFIX = "Vis"
INFRARED_PREFIX = "Ir"
ANNOTATIONS_PREFIX = "Annotation"
IMAGE_SUFFIX = ".png"

# 50 frames per block gives 84 blocks from 4200 frames -- enough blocks for a stable 80/20
# ratio, and long enough that a block boundary rarely cuts a sequence into both splits.
SPLIT_BLOCK_SIZE = 50
SPLIT_VAL_FRACTION = 0.2
SPLIT_SEED = 0


@dataclass(frozen=True, slots=True)
class M3fdAnnotation:
    stem: str
    width: int
    height: int
    boxes: tuple[VocBox, ...]


def adapt_m3fd(raw_root: Path, dest_root: Path) -> Path:
    """Convert ``raw_root/M3FD_Detection.zip`` into ``dest_root``. Returns the data.yaml path."""
    raw_root = Path(raw_root)
    dest_root = Path(dest_root)

    if dest_already_populated(dest_root):
        logger.info("%s already populated, skipping", dest_root)
        return dest_root / "data.yaml"

    archive_path = raw_root / ARCHIVE_NAME
    if not archive_path.is_file():
        raise AdapterError(f"{archive_path} not found -- fetch it first via fetch_datasets.py")

    with zipfile.ZipFile(archive_path) as archive:
        annotations = read_annotations(archive)
        names = sorted({box.name for a in annotations for box in a.boxes})
        if not names:
            raise AdapterError(f"{archive_path}: no <object> boxes found in any annotation")
        splits = assign_splits(
            [a.stem for a in annotations], SPLIT_BLOCK_SIZE, SPLIT_VAL_FRACTION, SPLIT_SEED
        )

        dropped_boxes = 0
        for annotation in annotations:
            split_root = dest_root / splits[annotation.stem]
            filename = f"{annotation.stem}{IMAGE_SUFFIX}"
            write_image_pair_bytes(
                archive.read(f"{VISIBLE_PREFIX}/{filename}"),
                archive.read(f"{INFRARED_PREFIX}/{filename}"),
                annotation.stem,
                split_root,
                IMAGE_SUFFIX,
            )
            lines, box_dropped = voc_to_yolo_lines(
                annotation.boxes, annotation.width, annotation.height, names
            )
            dropped_boxes += box_dropped
            write_label_lines(split_root, annotation.stem, lines)

    if dropped_boxes:
        logger.warning(
            "m3fd: dropped %d degenerate box(es) (zero-area after clamping to image bounds)",
            dropped_boxes,
        )
    val = sum(1 for split in splits.values() if split == "val")
    logger.info("m3fd: %d train pairs, %d val pairs -> %s", len(splits) - val, val, dest_root)
    return write_manifest_yaml(dest_root, names)


def assign_splits(
    stems: list[str], block_size: int, val_fraction: float, seed: int
) -> dict[str, str]:
    """Map each stem to ``train`` or ``val``, keeping contiguous blocks of sorted stems together.

    Deterministic for a given input and seed. At least one block always goes to val.
    """
    ordered = sorted(stems)
    blocks = [ordered[i : i + block_size] for i in range(0, len(ordered), block_size)]
    val_count = max(1, round(len(blocks) * val_fraction))
    val_blocks = set(random.Random(seed).sample(range(len(blocks)), val_count))
    return {
        stem: "val" if index in val_blocks else "train"
        for index, block in enumerate(blocks)
        for stem in block
    }


def read_annotations(archive: zipfile.ZipFile) -> list[M3fdAnnotation]:
    annotations: list[M3fdAnnotation] = []
    for name in archive.namelist():
        if not name.startswith(f"{ANNOTATIONS_PREFIX}/") or not name.endswith(".xml"):
            continue
        root = ET.fromstring(archive.read(name))

        size = root.find("size")
        if size is None:
            raise AdapterError(f"{name}: missing <size>")
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            raise AdapterError(f"{name}: non-positive <size> {width}x{height}")

        boxes = tuple(read_voc_box(name, obj) for obj in root.findall("object"))
        annotations.append(M3fdAnnotation(Path(name).stem, width, height, boxes))

    if not annotations:
        raise AdapterError("no annotation XML files found in the archive")
    return annotations
