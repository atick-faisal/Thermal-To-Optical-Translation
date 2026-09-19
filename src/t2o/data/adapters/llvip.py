"""LLVIP -> the internal ``{split}/{visible,infrared}/{images,labels}`` representation.

``dataset/raw/llvip/LLVIP.zip`` is the registered set from ``bupt-ai-cz/LLVIP``'s
``download_dataset.md`` (fetched by ``fetch_datasets.py``). Verified against the real archive::

    LLVIP/visible/{train,test}/NNNNNN.jpg    # 12025 train / 3463 test, 1280x1024
    LLVIP/infrared/{train,test}/NNNNNN.jpg   # same stems as visible/
    LLVIP/Annotations/NNNNNN.xml             # 15488, VOC-XML, one class: ``person``

Unlike M3FD, LLVIP ships an **official split** in its folder layout: ``train`` stays ``train``
and ``test`` becomes ``val``, as MSRS's does. Two annotations have no boxes (negatives, so no
label file) and five boxes are zero-area (dropped by ``voc_to_yolo_lines``).

**The images may already be here.** ``../Thermal-Image-Registration``'s ``cmreg ingest`` writes
LLVIP's images, byte-for-byte and with the same split, into this very tree -- but no labels and
no ``data.yaml``. So this adapter doesn't use ``dest_already_populated`` (it would skip and
leave the dataset unlabelled). It skips each image pair that already exists and always writes
the labels and ``data.yaml``. Images are never renamed, moved or re-encoded: the sister project
reads them too.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from t2o.data.adapters.common import (
    IMAGES_SEGMENT,
    AdapterError,
    VocBox,
    read_voc_box,
    read_voc_size,
    voc_to_yolo_lines,
    write_image_pair_bytes,
    write_label_lines,
    write_manifest_yaml,
)

logger = logging.getLogger(__name__)

ARCHIVE_NAME = "LLVIP.zip"
ANNOTATIONS_PREFIX = "LLVIP/Annotations/"
IMAGE_SUFFIX = ".jpg"
# Our split -> LLVIP's upstream directory.
SPLIT_SOURCES = {"train": "train", "val": "test"}


@dataclass(frozen=True, slots=True)
class LlvipAnnotation:
    width: int
    height: int
    boxes: tuple[VocBox, ...]


def adapt_llvip(raw_root: Path, dest_root: Path) -> Path:
    """Convert ``raw_root/LLVIP.zip`` into ``dest_root``. Returns the data.yaml path."""
    archive_path = Path(raw_root) / ARCHIVE_NAME
    dest_root = Path(dest_root)
    if not archive_path.is_file():
        raise AdapterError(f"{archive_path} not found -- fetch it first via fetch_datasets.py")

    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        annotations = read_annotations(archive)
        names = sorted({box.name for a in annotations.values() for box in a.boxes})
        if not names:
            raise AdapterError(f"{archive_path}: no <object> boxes found in any annotation")

        counts: dict[str, int] = {}
        dropped_boxes = 0
        for split, upstream in SPLIT_SOURCES.items():
            split_root = dest_root / split
            stems = split_stems(members, upstream, set(annotations))
            for stem in stems:
                _write_images_unless_present(archive, upstream, stem, split_root)
                a = annotations[stem]
                lines, box_dropped = voc_to_yolo_lines(a.boxes, a.width, a.height, names)
                dropped_boxes += box_dropped
                write_label_lines(split_root, stem, lines)
            counts[split] = len(stems)

    if dropped_boxes:
        logger.warning(
            "llvip: dropped %d degenerate box(es) (zero-area after clamping to image bounds)",
            dropped_boxes,
        )
    logger.info(
        "llvip: %d train pairs, %d val pairs -> %s", counts["train"], counts["val"], dest_root
    )
    return write_manifest_yaml(dest_root, names)


def split_stems(members: set[str], upstream: str, annotated: set[str]) -> list[str]:
    """Sorted stems of one upstream split, checked to be complete pairs with annotations.

    A stem missing its infrared twin or its XML would otherwise surface much later as a
    pairing error or a silent "no boxes" negative -- fail here, naming the stems.
    """
    visible = _stems(members, f"LLVIP/visible/{upstream}/")
    if not visible:
        raise AdapterError(f"no images under LLVIP/visible/{upstream}/")
    infrared = _stems(members, f"LLVIP/infrared/{upstream}/")
    if visible != infrared:
        raise AdapterError(
            f"{upstream}: visible and infrared stems differ, e.g. {sorted(visible ^ infrared)[:5]}"
        )
    unannotated = visible - annotated
    if unannotated:
        raise AdapterError(
            f"{upstream}: {len(unannotated)} image(s) have no annotation, "
            f"e.g. {sorted(unannotated)[:5]}"
        )
    return sorted(visible)


def read_annotations(archive: zipfile.ZipFile) -> dict[str, LlvipAnnotation]:
    annotations: dict[str, LlvipAnnotation] = {}
    for name in archive.namelist():
        if not name.startswith(ANNOTATIONS_PREFIX) or not name.endswith(".xml"):
            continue
        root = ET.fromstring(archive.read(name))
        width, height = read_voc_size(name, root)
        boxes = tuple(read_voc_box(name, obj) for obj in root.findall("object"))
        annotations[Path(name).stem] = LlvipAnnotation(width, height, boxes)
    return annotations


def _stems(members: set[str], prefix: str) -> set[str]:
    return {Path(m).stem for m in members if m.startswith(prefix) and m.endswith(IMAGE_SUFFIX)}


def _write_images_unless_present(
    archive: zipfile.ZipFile, upstream: str, stem: str, split_root: Path
) -> None:
    filename = f"{stem}{IMAGE_SUFFIX}"
    modalities = ("visible", "infrared")
    if all((split_root / m / IMAGES_SEGMENT / filename).is_file() for m in modalities):
        return
    write_image_pair_bytes(
        archive.read(f"LLVIP/visible/{upstream}/{filename}"),
        archive.read(f"LLVIP/infrared/{upstream}/{filename}"),
        stem,
        split_root,
        IMAGE_SUFFIX,
    )
