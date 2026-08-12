"""`data/adapters/flir.py` -- FLIR-aligned's zip archive -> the internal representation.

Everything here builds a tiny FLIR-*shaped* zip at test time (PLAN.md §9's synthetic-fixture
discipline): the same `align/{JPEGImages,Annotations}` entry names and VOC-XML shape as the
real `aligned.zip`, at fixture scale. One test additionally exercises the real local
`dataset/raw/flir/aligned.zip` when present -- `skipif`-guarded and `slow` (it reads and
copies real image bytes out of a ~1.4GB archive), and never the sole coverage of any code path.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from t2o.data.adapters import AdapterError, adapt_flir
from t2o.data.manifest import DatasetManifest

XML_TEMPLATE = """<Annotation>
  <folder>{folder}</folder>
  <filename>{stem}.jpeg</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>3</depth>
  </size>
  {objects}
</Annotation>"""

OBJECT_TEMPLATE = """<object>
    <name>{name}</name>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>"""

WIDTH, HEIGHT = 50, 50


def _jpeg_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8), mode="RGB").save(
        buf, format="JPEG"
    )
    return buf.getvalue()


def _build_flir_zip(path: Path, extra_annotation_xml: str | None = None) -> Path:
    """A 3-pair FLIR-shaped archive: two clean training boxes, one validation box, one
    training pair whose only box falls entirely outside the frame after clamping."""
    archive_path = path / "aligned.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        pairs = {
            "FLIR_00001": ("training", [("person", 5, 5, 25, 25), ("car", 10, 10, 40, 30)]),
            "FLIR_00002": ("validation", [("car", 0, 0, 20, 20)]),
            "FLIR_00003": ("training", [("person", 60, 60, 70, 70)]),  # entirely out of frame
        }
        for stem, (folder, boxes) in pairs.items():
            z.writestr(f"align/JPEGImages/{stem}_RGB.jpg", _jpeg_bytes(hash(stem) % 1000))
            z.writestr(
                f"align/JPEGImages/{stem}_PreviewData.jpeg", _jpeg_bytes(hash(stem) % 999 + 1)
            )
            objects = "\n  ".join(
                OBJECT_TEMPLATE.format(name=n, xmin=x0, ymin=y0, xmax=x1, ymax=y1)
                for n, x0, y0, x1, y1 in boxes
            )
            z.writestr(
                f"align/Annotations/{stem}_PreviewData.xml",
                XML_TEMPLATE.format(
                    folder=folder, stem=stem, width=WIDTH, height=HEIGHT, objects=objects
                ),
            )
        if extra_annotation_xml is not None:
            z.writestr("align/Annotations/FLIR_09999_PreviewData.xml", extra_annotation_xml)
    return archive_path


@pytest.fixture
def flir_raw_root(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _build_flir_zip(raw_root)
    return raw_root


def test_adapt_flir_produces_expected_layout_and_counts(
    flir_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_flir(flir_raw_root, dest)

    train_visible = {p.name for p in (dest / "train" / "visible" / "images").iterdir()}
    train_infrared = {p.name for p in (dest / "train" / "infrared" / "images").iterdir()}
    val_visible = {p.name for p in (dest / "val" / "visible" / "images").iterdir()}
    assert train_visible == train_infrared == {"FLIR_00001.jpg", "FLIR_00003.jpg"}
    assert val_visible == {"FLIR_00002.jpg"}


def test_written_manifest_loads_with_alphabetically_sorted_classes(
    flir_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    data_yaml = adapt_flir(flir_raw_root, dest)
    manifest = DatasetManifest.load(data_yaml)

    assert manifest.class_names == ["car", "person"]
    assert manifest.nc == 2


def test_label_conversion_matches_hand_computed_normalised_boxes(
    flir_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_flir(flir_raw_root, dest)

    lines = (dest / "train" / "visible" / "labels" / "FLIR_00001.txt").read_text().splitlines()
    assert len(lines) == 2
    # person: xmin=5,ymin=5,xmax=25,ymax=25 on a 50x50 frame -> cx=cy=0.3, w=h=0.4, cls=1 ("person")
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "1"
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.3, 0.3, 0.4, 0.4))


def test_box_entirely_outside_the_frame_is_dropped_not_written(
    flir_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_flir(flir_raw_root, dest)

    # FLIR_00003's only box (60,60,70,70) clamps to zero area on a 50x50 frame and is
    # dropped -- the image pair still exists, but with no label file (a legitimate negative).
    assert (dest / "train" / "visible" / "images" / "FLIR_00003.jpg").exists()
    assert not (dest / "train" / "visible" / "labels" / "FLIR_00003.txt").exists()


def test_missing_archive_raises(tmp_path: Path) -> None:
    empty_raw_root = tmp_path / "raw"
    empty_raw_root.mkdir()

    with pytest.raises(AdapterError, match="not found"):
        adapt_flir(empty_raw_root, tmp_path / "processed")


def test_unrecognised_folder_value_raises(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    bad_xml = XML_TEMPLATE.format(
        folder="testing",  # not "training"/"validation"
        stem="FLIR_09999",
        width=WIDTH,
        height=HEIGHT,
        objects="",
    )
    _build_flir_zip(raw_root, extra_annotation_xml=bad_xml)

    with pytest.raises(AdapterError, match="unrecognised <folder>"):
        adapt_flir(raw_root, tmp_path / "processed")


def test_adapt_flir_is_idempotent_against_a_populated_dest(
    flir_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"
    dest.mkdir()
    (dest / "sentinel.txt").write_text("already adapted")

    result = adapt_flir(flir_raw_root, dest)

    assert result == dest / "data.yaml"
    assert not (dest / "train").exists()
    assert (dest / "sentinel.txt").read_text() == "already adapted"


REAL_FLIR_ARCHIVE = Path("dataset/raw/flir/aligned.zip")


@pytest.mark.slow
@pytest.mark.skipif(not REAL_FLIR_ARCHIVE.is_file(), reason="real dataset/raw/flir not present")
def test_real_flir_adapts_and_loads(tmp_path: Path) -> None:
    from t2o.data.dataset import TranslationPairDataset

    data_yaml = adapt_flir(REAL_FLIR_ARCHIVE.parent, tmp_path / "processed")
    manifest = DatasetManifest.load(data_yaml)

    train = TranslationPairDataset(
        manifest.train_images, pairing=manifest.pairing, num_classes=manifest.nc
    )
    val = TranslationPairDataset(
        manifest.val_images, pairing=manifest.pairing, num_classes=manifest.nc
    )
    assert len(train) + len(val) >= 5000
