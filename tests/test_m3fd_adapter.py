"""`data/adapters/m3fd.py` -- M3FD's detection zip -> the internal representation.

Everything here builds a tiny M3FD-*shaped* zip at test time (PLAN.md §9's synthetic-fixture
discipline): the same `{Vis,Ir,Annotation}/NNNNN.*` entry names and VOC-XML shape as the real
`M3FD_Detection.zip`, at fixture scale. One test additionally reads the real local archive's
annotations when present -- `skipif`-guarded and `slow`, and never the sole coverage of any
code path.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from t2o.data.adapters import AdapterError, adapt_m3fd, m3fd
from t2o.data.adapters.m3fd import assign_splits, read_annotations, read_val_stems
from t2o.data.manifest import DatasetManifest

XML_TEMPLATE = """<annotation>
  <folder>F</folder>
  <filename>{stem}.png</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>3</depth>
  </size>
  {objects}
</annotation>"""

OBJECT_TEMPLATE = """<object>
    <name>{name}</name>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>"""

WIDTH, HEIGHT = 50, 50
PAIRS = {
    "00000": [("People", 5, 5, 25, 25), ("Car", 10, 10, 40, 30)],
    "00001": [("Car", 0, 0, 20, 20)],
    "00002": [("Lamp", 30, 30, 40, 45)],
}


def _png_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8), mode="RGB").save(
        buf, format="PNG"
    )
    return buf.getvalue()


def _build_m3fd_zip(raw_root: Path) -> Path:
    archive_path = raw_root / "M3FD_Detection.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        for seed, (stem, boxes) in enumerate(PAIRS.items()):
            z.writestr(f"Vis/{stem}.png", _png_bytes(seed))
            z.writestr(f"Ir/{stem}.png", _png_bytes(seed + 100))
            objects = "\n  ".join(
                OBJECT_TEMPLATE.format(name=n, xmin=x0, ymin=y0, xmax=x1, ymax=y1)
                for n, x0, y0, x1, y1 in boxes
            )
            z.writestr(
                f"Annotation/{stem}.xml",
                XML_TEMPLATE.format(stem=stem, width=WIDTH, height=HEIGHT, objects=objects),
            )
    return archive_path


@pytest.fixture
def m3fd_raw_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # A fixture-scale stand-in for the vendored DAMSDet val list.
    val_stems_file = tmp_path / "val.txt"
    val_stems_file.write_text("00001\n")
    monkeypatch.setattr(m3fd, "VAL_STEMS_FILE", val_stems_file)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _build_m3fd_zip(raw_root)
    return raw_root


def _names(images_dir: Path) -> set[str]:
    return {p.name for p in images_dir.iterdir()} if images_dir.is_dir() else set()


def test_adapt_m3fd_writes_every_pair_once_with_matching_filenames(
    m3fd_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_m3fd(m3fd_raw_root, dest)

    train_visible = _names(dest / "train" / "visible" / "images")
    val_visible = _names(dest / "val" / "visible" / "images")
    assert train_visible == _names(dest / "train" / "infrared" / "images")
    assert val_visible == _names(dest / "val" / "infrared" / "images")
    assert train_visible | val_visible == {f"{stem}.png" for stem in PAIRS}
    assert val_visible == {"00001.png"}


def test_written_manifest_loads_with_alphabetically_sorted_classes(
    m3fd_raw_root: Path, tmp_path: Path
) -> None:
    data_yaml = adapt_m3fd(m3fd_raw_root, tmp_path / "processed")
    manifest = DatasetManifest.load(data_yaml)

    assert manifest.class_names == ["Car", "Lamp", "People"]
    assert manifest.nc == 3


def test_label_conversion_matches_hand_computed_normalised_boxes(
    m3fd_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_m3fd(m3fd_raw_root, dest)

    lines = (dest / "train" / "visible" / "labels" / "00000.txt").read_text().splitlines()
    assert len(lines) == 2
    # People: xmin=5,ymin=5,xmax=25,ymax=25 on a 50x50 frame -> cx=cy=0.3, w=h=0.4, cls=2
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "2"
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.3, 0.3, 0.4, 0.4))


def test_missing_archive_raises(tmp_path: Path) -> None:
    empty_raw_root = tmp_path / "raw"
    empty_raw_root.mkdir()

    with pytest.raises(AdapterError, match="not found"):
        adapt_m3fd(empty_raw_root, tmp_path / "processed")


def test_adapt_m3fd_is_idempotent_against_a_populated_dest(
    m3fd_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"
    dest.mkdir()
    (dest / "sentinel.txt").write_text("already adapted")

    result = adapt_m3fd(m3fd_raw_root, dest)

    assert result == dest / "data.yaml"
    assert not (dest / "train").exists()


def test_assign_splits_sends_listed_stems_to_val_and_the_rest_to_train() -> None:
    splits = assign_splits(["00000", "00001", "00002"], {"00001"})

    assert splits == {"00000": "train", "00001": "val", "00002": "train"}


def test_assign_splits_raises_on_a_val_stem_missing_from_the_archive() -> None:
    with pytest.raises(AdapterError, match="not in the archive"):
        assign_splits(["00000"], {"00000", "09999"})


def test_vendored_damsdet_val_list_is_intact() -> None:
    val_stems = read_val_stems(m3fd.VAL_STEMS_FILE)

    assert len(val_stems) == 831
    assert val_stems <= {f"{i:05d}" for i in range(4200)}


REAL_M3FD_ARCHIVE = Path("dataset/raw/m3fd/M3FD_Detection.zip")


@pytest.mark.slow
@pytest.mark.skipif(not REAL_M3FD_ARCHIVE.is_file(), reason="real dataset/raw/m3fd not present")
def test_real_m3fd_annotations_parse_and_split() -> None:
    # Reads annotations only -- copying the ~7GB of images belongs to the real adapter run.
    with zipfile.ZipFile(REAL_M3FD_ARCHIVE) as archive:
        annotations = read_annotations(archive)

    assert len(annotations) == 4200
    assert sorted({box.name for a in annotations for box in a.boxes}) == [
        "Bus",
        "Car",
        "Lamp",
        "Motorcycle",
        "People",
        "Truck",
    ]
    splits = assign_splits([a.stem for a in annotations], read_val_stems(m3fd.VAL_STEMS_FILE))
    assert sum(1 for split in splits.values() if split == "val") == 831
