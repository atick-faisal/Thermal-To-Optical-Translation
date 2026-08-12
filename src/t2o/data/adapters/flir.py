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
    dest_already_populated,
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
class _Box:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True, slots=True)
class _Annotation:
    stem: str
    split: str
    width: int
    height: int
    boxes: tuple[_Box, ...]


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
            lines, box_dropped = _yolo_lines(annotation, names)
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

        size = root.find("size")
        if size is None:
            raise AdapterError(f"{name}: missing <size>")
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            raise AdapterError(f"{name}: non-positive <size> {width}x{height}")

        boxes = tuple(_read_box(name, obj) for obj in root.findall("object"))
        annotations.append(_Annotation(stem, split, width, height, boxes))

    if not annotations:
        raise AdapterError("no annotation XML files found in the archive")
    return annotations


def _read_box(xml_name: str, obj: ET.Element) -> _Box:
    box_name = obj.findtext("name")
    bndbox = obj.find("bndbox")
    if box_name is None or bndbox is None:
        raise AdapterError(f"{xml_name}: <object> missing <name> or <bndbox>")
    return _Box(
        name=box_name,
        xmin=float(bndbox.findtext("xmin", "0")),
        ymin=float(bndbox.findtext("ymin", "0")),
        xmax=float(bndbox.findtext("xmax", "0")),
        ymax=float(bndbox.findtext("ymax", "0")),
    )


def _yolo_lines(annotation: _Annotation, names: list[str]) -> tuple[list[str], int]:
    """Convert absolute VOC-XML boxes to normalised YOLO ``cls cx cy w h`` lines.

    Coordinates are clamped to the image before normalising -- a handful of FLIR's boxes run
    slightly past the edge, and a box that's zero-area after clamping is dropped rather than
    written as a degenerate detection target.
    """
    lines: list[str] = []
    dropped = 0
    for box in annotation.boxes:
        xmin = max(0.0, min(box.xmin, annotation.width))
        xmax = max(0.0, min(box.xmax, annotation.width))
        ymin = max(0.0, min(box.ymin, annotation.height))
        ymax = max(0.0, min(box.ymax, annotation.height))
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            dropped += 1
            continue
        cx = (xmin + xmax) / 2 / annotation.width
        cy = (ymin + ymax) / 2 / annotation.height
        nw = w / annotation.width
        nh = h / annotation.height
        lines.append(f"{names.index(box.name)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines, dropped
