"""FLIR-aligned -> the internal ``{split}/{visible,infrared}/{images,labels}`` representation.

The raw layout is a zip archive (``dataset/raw/flir/aligned.zip``, fetched via
``huggingface_hub.snapshot_download`` -- never extracted to disk by ``fetch_datasets.py``),
verified against the real archive (30858 entries)::

    align/JPEGImages/FLIR_XXXXX_RGB.jpg           # 10284 -- every pair, annotated or not
    align/JPEGImages/FLIR_XXXXX_PreviewData.jpeg  # 10284, same count
    align/Annotations/FLIR_XXXXX_PreviewData.xml  # 5142 -- only half the pairs are annotated

Scope is those 5142 annotated pairs. Each XML's own ``<folder>`` field is ``training`` or
``validation`` -- the *only* source of train/val split information (verified against an
800-file sample) -- and the unannotated half has neither a split nor a label, so adapting it
would only add unlabelled bulk MSRS's much larger unlabelled pool already covers more
compactly.

Visible and infrared share no filename stem (``FLIR_00002_RGB.jpg`` vs
``FLIR_00002_PreviewData.jpeg``) -- both are renamed to ``FLIR_00002.jpg`` on copy, since
``data.pairing.Pairing`` derives one modality from the other by substituting only the
``visible``/``infrared`` path segment and expects the filename itself to match exactly. Boxes
are VOC-XML ``bndbox`` (absolute ``xmin``/``ymin``/``xmax``/``ymax``), converted to YOLO's
normalised ``cx cy w h``; class names are collected from the data itself (sorted
alphabetically) rather than hardcoded, since nothing in the archive declares a canonical order
the way MSRS's ``classes.txt`` does.
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

ARCHIVE_NAME = "aligned.zip"
IMAGES_PREFIX = "align/JPEGImages"
ANNOTATIONS_PREFIX = "align/Annotations"
RGB_SUFFIX = "_RGB.jpg"
THERMAL_SUFFIX = "_PreviewData.jpeg"
XML_SUFFIX = "_PreviewData.xml"
DEST_IMAGE_SUFFIX = ".jpg"

# The only two values FLIR's own <folder> tag takes (verified against an 800-file sample of
# the real archive). Anything else is a raw-layout surprise worth failing loudly on rather
# than silently dropping into a default split.
_SPLIT_BY_FOLDER = {"training": "train", "validation": "val"}


@dataclass(frozen=True, slots=True)
class _Annotation:
    stem: str
    split: str
    width: int
    height: int
    boxes: tuple[VocBox, ...]


def adapt_flir(raw_root: Path, dest_root: Path) -> Path:
    """Convert ``raw_root/aligned.zip`` into ``dest_root``. Returns the written data.yaml path."""
    raw_root = Path(raw_root)
    dest_root = Path(dest_root)

    if dest_already_populated(dest_root):
        logger.info("%s already populated, skipping", dest_root)
        return dest_root / "data.yaml"

    archive_path = raw_root / ARCHIVE_NAME
    if not archive_path.is_file():
        raise AdapterError(f"{archive_path} not found -- fetch it first via fetch_datasets.py")

    with zipfile.ZipFile(archive_path) as archive:
        annotations = _read_annotations(archive)
        names = sorted({box.name for a in annotations for box in a.boxes})
        if not names:
            raise AdapterError(f"{archive_path}: no <object> boxes found in any annotation")

        dropped_boxes = 0
        for annotation in annotations:
            split_root = dest_root / annotation.split
            visible_data = archive.read(f"{IMAGES_PREFIX}/{annotation.stem}{RGB_SUFFIX}")
            infrared_data = archive.read(f"{IMAGES_PREFIX}/{annotation.stem}{THERMAL_SUFFIX}")
            write_image_pair_bytes(
                visible_data, infrared_data, annotation.stem, split_root, DEST_IMAGE_SUFFIX
            )
            lines, box_dropped = voc_to_yolo_lines(
                annotation.boxes, annotation.width, annotation.height, names
            )
            dropped_boxes += box_dropped
            write_label_lines(split_root, annotation.stem, lines)

    if dropped_boxes:
        logger.warning(
            "flir: dropped %d degenerate box(es) (zero-area after clamping to image bounds)",
            dropped_boxes,
        )
    train = sum(1 for a in annotations if a.split == "train")
    val = sum(1 for a in annotations if a.split == "val")
    logger.info("flir: %d train pairs, %d val pairs -> %s", train, val, dest_root)
    return write_manifest_yaml(dest_root, names)


def _read_annotations(archive: zipfile.ZipFile) -> list[_Annotation]:
    annotations: list[_Annotation] = []
    for name in archive.namelist():
        if not name.startswith(ANNOTATIONS_PREFIX) or not name.endswith(XML_SUFFIX):
            continue
        stem = Path(name).name.removesuffix(XML_SUFFIX)
        root = ET.fromstring(archive.read(name))

        folder = root.findtext("folder")
        split = _SPLIT_BY_FOLDER.get(folder or "")
        if split is None:
            raise AdapterError(f"{name}: unrecognised <folder>{folder}</folder>")

        width, height = read_voc_size(name, root)
        boxes = tuple(read_voc_box(name, obj) for obj in root.findall("object"))
        annotations.append(_Annotation(stem, split, width, height, boxes))

    if not annotations:
        raise AdapterError("no annotation XML files found in the archive")
    return annotations
