"""Shared helpers for converting a public dataset's raw layout into the internal
``{split}/{visible,infrared}/{images,labels}`` contract (PLAN.md §9).

One place to write the ``rgbt:``-block ``data.yaml``, copy image/label pairs and convert
VOC-XML boxes, so every per-dataset adapter (``msrs.py``, ``flir.py``, ``m3fd.py``) produces
byte-identical shapes rather than each inventing its own. Images are always copied
byte-for-byte, never re-encoded -- either straight off disk (``copy_image_pair``, MSRS's
git-cloned tree) or straight out of an archive (``write_image_pair_bytes``, FLIR-aligned's and
M3FD's zips) -- which is what keeps this step cheap on thousand-plus-image datasets.
"""

from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

from t2o.data.manifest import RGBT_BLOCK

logger = logging.getLogger(__name__)

DATA_FILENAME = "data.yaml"
IMAGES_SEGMENT = "images"
LABELS_SEGMENT = "labels"


class AdapterError(ValueError):
    """Raised when a raw dataset's layout doesn't match what its adapter expects."""


@dataclass(frozen=True, slots=True)
class VocBox:
    """One VOC-XML ``<object>``: class name plus absolute ``bndbox`` pixel coordinates."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def read_voc_size(xml_name: str, root: ET.Element) -> tuple[int, int]:
    """Read a VOC-XML ``<size>`` as ``(width, height)``, rejecting a missing or empty one."""
    size = root.find("size")
    if size is None:
        raise AdapterError(f"{xml_name}: missing <size>")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise AdapterError(f"{xml_name}: non-positive <size> {width}x{height}")
    return width, height


def read_voc_box(xml_name: str, obj: ET.Element) -> VocBox:
    box_name = obj.findtext("name")
    bndbox = obj.find("bndbox")
    if box_name is None or bndbox is None:
        raise AdapterError(f"{xml_name}: <object> missing <name> or <bndbox>")
    return VocBox(
        name=box_name,
        xmin=float(bndbox.findtext("xmin", "0")),
        ymin=float(bndbox.findtext("ymin", "0")),
        xmax=float(bndbox.findtext("xmax", "0")),
        ymax=float(bndbox.findtext("ymax", "0")),
    )


def voc_to_yolo_lines(
    boxes: tuple[VocBox, ...], width: int, height: int, names: list[str]
) -> tuple[list[str], int]:
    """Convert absolute VOC-XML boxes to normalised YOLO ``cls cx cy w h`` lines.

    Coordinates are clamped to the image before normalising -- some datasets' boxes run
    slightly past the edge, and a box that's zero-area after clamping is dropped rather than
    written as a degenerate detection target. Returns the lines and the dropped-box count.
    """
    lines: list[str] = []
    dropped = 0
    for box in boxes:
        xmin = max(0.0, min(box.xmin, width))
        xmax = max(0.0, min(box.xmax, width))
        ymin = max(0.0, min(box.ymin, height))
        ymax = max(0.0, min(box.ymax, height))
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            dropped += 1
            continue
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        lines.append(f"{names.index(box.name)} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    return lines, dropped


def _images_dir(split_root: Path, modality: str) -> Path:
    images_dir = split_root / modality / IMAGES_SEGMENT
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def copy_image_pair(
    visible_src: Path, infrared_src: Path, stem: str, split_root: Path, suffix: str
) -> None:
    """Copy one visible/infrared pair into ``split_root/{visible,infrared}/images/``.

    Both destinations share ``stem`` and ``suffix`` -- required by
    ``data.pairing.Pairing.infrared_path``, which derives the infrared counterpart by
    substituting only the ``visible``/``infrared`` path segment and expects the filename
    itself to match exactly.
    """
    for modality, src in (("visible", visible_src), ("infrared", infrared_src)):
        shutil.copyfile(src, _images_dir(split_root, modality) / f"{stem}{suffix}")


def write_image_pair_bytes(
    visible_data: bytes, infrared_data: bytes, stem: str, split_root: Path, suffix: str
) -> None:
    """Byte-for-byte equivalent of :func:`copy_image_pair` for a source that isn't a plain
    file on disk -- FLIR-aligned's raw layout is a zip archive, read via
    ``zipfile.ZipFile.read()`` rather than a path ``shutil.copyfile`` can take.
    """
    for modality, data in (("visible", visible_data), ("infrared", infrared_data)):
        (_images_dir(split_root, modality) / f"{stem}{suffix}").write_bytes(data)


def write_label(split_root: Path, stem: str, source_label: Path | None) -> None:
    """Copy a label file verbatim into ``split_root/visible/labels/{stem}.txt``.

    ``source_label=None`` (or a path that doesn't exist) writes nothing --
    ``data.labels.load_yolo_labels`` already treats a missing label file as a legitimate
    zero-instance negative, so an empty placeholder would add a file without adding
    information.
    """
    if source_label is None or not source_label.exists():
        return
    labels_dir = split_root / "visible" / LABELS_SEGMENT
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_label, labels_dir / f"{stem}.txt")


def write_label_lines(split_root: Path, stem: str, lines: list[str]) -> None:
    """Write YOLO label lines computed in memory (e.g. converted from VOC-XML boxes).

    ``lines=[]`` writes nothing, for the same "missing = negative" reason
    :func:`write_label` skips a nonexistent source.
    """
    if not lines:
        return
    labels_dir = split_root / "visible" / LABELS_SEGMENT
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def write_manifest_yaml(
    root: Path,
    names: list[str],
    visible_token: str = "visible",
    infrared_token: str = "infrared",
) -> Path:
    """Write the ``data.yaml`` every adapter's output is read through.

    Matches ``tests/conftest.py``'s synthetic-fixture shape exactly, since both are read by
    the same ``t2o.data.manifest.DatasetManifest.load``.
    """
    destination = Path(root) / DATA_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            {
                "path": str(Path(root).resolve()),
                "train": "train/visible/images",
                "val": "val/visible/images",
                "nc": len(names),
                "names": list(names),
                RGBT_BLOCK: {"visible_token": visible_token, "infrared_token": infrared_token},
            },
            sort_keys=False,
        )
    )
    return destination


def dest_already_populated(dest_root: Path) -> bool:
    """True if ``dest_root`` exists and has anything in it.

    Mirrors ``scripts/fetch_datasets.py::fetch_dataset``'s own idempotent-skip convention --
    a dataset already adapted doesn't need its (potentially large) image set copied again.
    """
    return dest_root.exists() and any(dest_root.iterdir())
