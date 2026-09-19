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

M3FD ships **no official train/val split** -- the XML ``<folder>``/``<path>`` fields record the
annotators' working directories, not a split, and papers use at least four different ones. We
use DAMSDet's (3368 train / 831 val), the most-reused published list: its val set is contiguous
scene runs, so near-duplicate neighbouring video frames don't leak across splits, and M3FD
numbers stay comparable with DAMSDet/MM-DETR. ``m3fd_damsdet_val.txt`` is their ``val.txt``
copied byte-for-byte (Apache-2.0) from
https://github.com/gjj45/DAMSDet/blob/b3b31ed75a1f50187799816dcc33469e760f9177/dataset/coco_m3fd/val.txt
Every stem not in it goes to train. That includes ``01300``, the one frame DAMSDet's lists omit.
The result is frozen and hashed in ``splits/m3fd.json`` like every other split.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from t2o.data.adapters.common import (
    AdapterError,
    VocBox,
    dest_already_populated,
    read_voc_box,
    read_voc_size,
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
VAL_STEMS_FILE = Path(__file__).with_name("m3fd_damsdet_val.txt")


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
        splits = assign_splits([a.stem for a in annotations], read_val_stems(VAL_STEMS_FILE))

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


def read_val_stems(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def assign_splits(stems: list[str], val_stems: set[str]) -> dict[str, str]:
    """Map each stem to ``val`` if it's in ``val_stems``, otherwise ``train``.

    A val stem missing from the archive means the list and the data have drifted apart --
    fail loudly rather than silently shrink val.
    """
    unknown = val_stems - set(stems)
    if unknown:
        preview = ", ".join(sorted(unknown)[:5])
        raise AdapterError(f"{len(unknown)} val stem(s) not in the archive: {preview}")
    return {stem: "val" if stem in val_stems else "train" for stem in stems}


def read_annotations(archive: zipfile.ZipFile) -> list[M3fdAnnotation]:
    annotations: list[M3fdAnnotation] = []
    for name in archive.namelist():
        if not name.startswith(f"{ANNOTATIONS_PREFIX}/") or not name.endswith(".xml"):
            continue
        root = ET.fromstring(archive.read(name))
        width, height = read_voc_size(name, root)
        boxes = tuple(read_voc_box(name, obj) for obj in root.findall("object"))
        annotations.append(M3fdAnnotation(Path(name).stem, width, height, boxes))

    if not annotations:
        raise AdapterError("no annotation XML files found in the archive")
    return annotations
