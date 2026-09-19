"""`data/adapters/llvip.py` -- LLVIP's zip -> the internal representation.

Everything here builds a tiny LLVIP-*shaped* zip at test time (PLAN.md §9's synthetic-fixture
discipline): the same `LLVIP/{visible,infrared}/{train,test}/` and `LLVIP/Annotations/` entry
names and VOC-XML shape as the real `LLVIP.zip`, at fixture scale. One test additionally reads
the real local archive's annotations when present -- `skipif`-guarded and `slow`.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from t2o.data.adapters import AdapterError, adapt_llvip
from t2o.data.adapters.llvip import read_annotations, split_stems
from t2o.data.manifest import DatasetManifest

XML_TEMPLATE = """<annotation>
  <folder>JPEGImages</folder>
  <filename>{stem}.jpg</filename>
  <size><width>{width}</width><height>{height}</height><depth>1</depth></size>
  {objects}
</annotation>"""

OBJECT_TEMPLATE = """<object>
    <name>person</name>
    <bndbox><xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax><ymax>{}</ymax></bndbox>
  </object>"""

WIDTH, HEIGHT = 50, 50
# stem -> (upstream split, person boxes). 010003 has no boxes: a negative, like two real ones.
PAIRS = {
    "010001": ("train", [(5, 5, 25, 25)]),
    "010002": ("train", [(0, 0, 20, 20), (30, 30, 40, 45)]),
    "010003": ("train", []),
    "190001": ("test", [(10, 10, 40, 30)]),
}


def _jpg_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8), mode="RGB").save(
        buf, format="JPEG"
    )
    return buf.getvalue()


def _build_llvip_zip(raw_root: Path, skip: set[str] | None = None) -> Path:
    """``skip`` holds entry names to leave out, for building a deliberately broken archive."""
    archive_path = raw_root / "LLVIP.zip"
    with zipfile.ZipFile(archive_path, "w") as z:

        def put(name: str, data: bytes | str) -> None:
            if name not in (skip or set()):
                z.writestr(name, data)

        for seed, (stem, (upstream, boxes)) in enumerate(PAIRS.items()):
            put(f"LLVIP/visible/{upstream}/{stem}.jpg", _jpg_bytes(seed))
            put(f"LLVIP/infrared/{upstream}/{stem}.jpg", _jpg_bytes(seed + 100))
            objects = "\n  ".join(OBJECT_TEMPLATE.format(*box) for box in boxes)
            put(
                f"LLVIP/Annotations/{stem}.xml",
                XML_TEMPLATE.format(stem=stem, width=WIDTH, height=HEIGHT, objects=objects),
            )
    return archive_path


@pytest.fixture
def llvip_raw_root(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _build_llvip_zip(raw_root)
    return raw_root


def _names(images_dir: Path) -> set[str]:
    return {p.name for p in images_dir.iterdir()} if images_dir.is_dir() else set()


def test_adapt_llvip_maps_train_to_train_and_test_to_val(
    llvip_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_llvip(llvip_raw_root, dest)

    train_visible = _names(dest / "train" / "visible" / "images")
    val_visible = _names(dest / "val" / "visible" / "images")
    assert train_visible == _names(dest / "train" / "infrared" / "images")
    assert val_visible == _names(dest / "val" / "infrared" / "images")
    assert train_visible == {"010001.jpg", "010002.jpg", "010003.jpg"}
    assert val_visible == {"190001.jpg"}


def test_images_are_copied_byte_for_byte(llvip_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    adapt_llvip(llvip_raw_root, dest)

    with zipfile.ZipFile(llvip_raw_root / "LLVIP.zip") as z:
        expected = z.read("LLVIP/infrared/test/190001.jpg")
    assert (dest / "val" / "infrared" / "images" / "190001.jpg").read_bytes() == expected


def test_written_manifest_loads_with_the_single_person_class(
    llvip_raw_root: Path, tmp_path: Path
) -> None:
    manifest = DatasetManifest.load(adapt_llvip(llvip_raw_root, tmp_path / "processed"))

    assert manifest.class_names == ["person"]
    assert manifest.nc == 1


def test_label_conversion_matches_hand_computed_normalised_boxes(
    llvip_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_llvip(llvip_raw_root, dest)

    labels = dest / "train" / "visible" / "labels"
    # xmin=5,ymin=5,xmax=25,ymax=25 on a 50x50 frame -> cx=cy=0.3, w=h=0.4
    cls, cx, cy, w, h = (labels / "010001.txt").read_text().split()
    assert cls == "0"
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.3, 0.3, 0.4, 0.4))
    assert len((labels / "010002.txt").read_text().splitlines()) == 2
    # A box-less annotation is a negative: no label file, per `write_label_lines`.
    assert not (labels / "010003.txt").exists()


def test_images_already_present_are_kept_and_labels_are_added(
    llvip_raw_root: Path, tmp_path: Path
) -> None:
    # The sister project's `cmreg ingest` leaves exactly this: images, no labels, no data.yaml.
    dest = tmp_path / "processed"
    existing = dest / "train" / "visible" / "images" / "010001.jpg"
    existing_ir = dest / "train" / "infrared" / "images" / "010001.jpg"
    for path in (existing, existing_ir):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"already here")

    data_yaml = adapt_llvip(llvip_raw_root, dest)

    assert existing.read_bytes() == b"already here"
    assert existing_ir.read_bytes() == b"already here"
    assert data_yaml.is_file()
    assert (dest / "train" / "visible" / "labels" / "010001.txt").is_file()
    assert (dest / "val" / "visible" / "images" / "190001.jpg").is_file()


def test_adapt_llvip_is_idempotent(llvip_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    adapt_llvip(llvip_raw_root, dest)
    first = sorted(p.relative_to(dest) for p in dest.rglob("*"))
    adapt_llvip(llvip_raw_root, dest)

    assert sorted(p.relative_to(dest) for p in dest.rglob("*")) == first


def test_missing_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="not found"):
        adapt_llvip(tmp_path, tmp_path / "processed")


def test_missing_infrared_twin_raises(tmp_path: Path) -> None:
    _build_llvip_zip(tmp_path, skip={"LLVIP/infrared/train/010002.jpg"})

    with pytest.raises(AdapterError, match="stems differ"):
        adapt_llvip(tmp_path, tmp_path / "processed")


def test_missing_annotation_raises(tmp_path: Path) -> None:
    _build_llvip_zip(tmp_path, skip={"LLVIP/Annotations/190001.xml"})

    with pytest.raises(AdapterError, match="no annotation"):
        adapt_llvip(tmp_path, tmp_path / "processed")


REAL_LLVIP_ARCHIVE = Path("dataset/raw/llvip/LLVIP.zip")


@pytest.mark.slow
@pytest.mark.skipif(not REAL_LLVIP_ARCHIVE.is_file(), reason="real dataset/raw/llvip not present")
def test_real_llvip_annotations_parse_and_split() -> None:
    # Reads annotations and entry names only -- copying ~4GB of images is the real run's job.
    with zipfile.ZipFile(REAL_LLVIP_ARCHIVE) as archive:
        annotations = read_annotations(archive)
        members = set(archive.namelist())

    assert len(annotations) == 15488
    assert {box.name for a in annotations.values() for box in a.boxes} == {"person"}
    assert len(split_stems(members, "train", set(annotations))) == 12025
    assert len(split_stems(members, "test", set(annotations))) == 3463
