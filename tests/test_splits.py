"""`data/splits.py` -- freezing, hashing, and drift-detecting a dataset's train/val
membership, against the synthetic `dataset_root` fixture (`tests/conftest.py`), and pinning
the held-out test split from an explicit images directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from t2o.data.manifest import DatasetManifest
from t2o.data.splits import (
    EmptySplitError,
    SplitDriftError,
    SplitManifest,
    freeze_split,
    load_split_manifest,
    load_test_pin,
    pin_test_split,
    verify_split,
    verify_test_split,
    write_split_manifest,
    write_test_pin,
)


def _expected_stems(dataset_root: Path, split: str) -> tuple[str, ...]:
    # Derived from the fixture itself rather than importing tests.conftest's constants -- a
    # `tests` package shipped inside a dependency wheel shadows the local `tests/` directory
    # on pyright's module search path (see tests/test_export.py's own note on this).
    images_dir = dataset_root / split / "visible" / "images"
    return tuple(sorted(p.stem for p in images_dir.iterdir()))


def test_freeze_split_captures_the_current_stems(data_yaml: Path, dataset_root: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)

    record = freeze_split("synthetic", manifest)

    assert record.train_stems == _expected_stems(dataset_root, "train")
    assert record.val_stems == _expected_stems(dataset_root, "val")


def test_freeze_split_is_deterministic(data_yaml: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)

    first = freeze_split("synthetic", manifest)
    second = freeze_split("synthetic", manifest)

    assert first.train_hash == second.train_hash
    assert first.val_hash == second.val_hash
    assert first.combined_hash == second.combined_hash


def test_hash_changes_when_a_stem_is_added(data_yaml: Path, dataset_root: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    before = freeze_split("synthetic", manifest)

    extra = dataset_root / "train" / "visible" / "images" / "99_extra_999.jpg"
    extra.write_bytes(b"not a real image, only the stem matters here")

    after = freeze_split("synthetic", manifest)
    assert after.train_hash != before.train_hash
    assert after.val_hash == before.val_hash
    assert after.combined_hash != before.combined_hash
    extra.unlink()  # session-scoped fixture -- clean up after mutating it


def test_combined_hash_is_a_function_of_both_split_hashes() -> None:
    a = SplitManifest(name="x", train_stems=("a",), val_stems=("b",), train_hash="1", val_hash="2")
    b = SplitManifest(name="x", train_stems=("a",), val_stems=("c",), train_hash="1", val_hash="3")
    assert a.combined_hash != b.combined_hash


def test_write_and_load_round_trip(data_yaml: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    record = freeze_split("synthetic", manifest)

    path = write_split_manifest(record, tmp_path / "splits" / "synthetic.json")
    loaded = load_split_manifest(path)

    assert loaded == record


def test_verify_split_passes_when_nothing_changed(data_yaml: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    record = freeze_split("synthetic", manifest)

    verify_split(record, manifest)  # must not raise


def test_verify_split_raises_on_drift(data_yaml: Path, dataset_root: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    record = freeze_split("synthetic", manifest)

    extra = dataset_root / "val" / "visible" / "images" / "99_extra_999.jpg"
    extra.write_bytes(b"not a real image, only the stem matters here")
    try:
        with pytest.raises(SplitDriftError, match="val: \\+1 -0"):
            verify_split(record, manifest)
    finally:
        extra.unlink()  # session-scoped fixture -- clean up after mutating it


def test_written_manifest_is_readable_json_with_stem_lists(
    data_yaml: Path, dataset_root: Path, tmp_path: Path
) -> None:
    import json

    manifest = DatasetManifest.load(data_yaml)
    record = freeze_split("synthetic", manifest)
    path = write_split_manifest(record, tmp_path / "synthetic.json")

    data = json.loads(path.read_text())
    assert data["name"] == "synthetic"
    assert data["train"]["count"] == len(_expected_stems(dataset_root, "train"))
    assert data["train"]["stems"] == list(_expected_stems(dataset_root, "train"))
    assert data["val"]["count"] == len(_expected_stems(dataset_root, "val"))
    assert "combined_hash" in data


# --- The held-out test split pin ----------------------------------------------------------

_REPO_SPLITS = Path(__file__).resolve().parent.parent / "splits"
_HELD_OUT = ("03_test_002", "01_test_000", "07_test_001")


def _images_dir(root: Path, stems: tuple[str, ...], suffix: str = ".jpg") -> Path:
    # Only the stem matters to a pin, so the files need not be real images.
    root.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (root / f"{stem}{suffix}").write_bytes(b"")
    return root


def test_pin_records_sorted_stems_and_skips_non_images(tmp_path: Path) -> None:
    images = _images_dir(tmp_path / "images", _HELD_OUT)
    (images / "notes.txt").write_text("not an image")

    pin = pin_test_split("synthetic", images)

    assert pin.stems == tuple(sorted(_HELD_OUT))
    assert len(pin.hash) == 16


def test_pin_is_the_same_from_either_modality(tmp_path: Path) -> None:
    visible = _images_dir(tmp_path / "visible", _HELD_OUT, ".png")
    infrared = _images_dir(tmp_path / "infrared", _HELD_OUT, ".jpg")

    assert pin_test_split("synthetic", visible) == pin_test_split("synthetic", infrared)


@pytest.mark.parametrize("make_dir", [False, True], ids=["missing", "empty"])
def test_pin_refuses_a_directory_with_no_images(tmp_path: Path, make_dir: bool) -> None:
    images = tmp_path / "images"
    if make_dir:
        images.mkdir()

    with pytest.raises(EmptySplitError, match=str(images)):
        pin_test_split("synthetic", images)


def test_verify_test_split_passes_when_nothing_changed(tmp_path: Path) -> None:
    images = _images_dir(tmp_path / "images", _HELD_OUT)

    verify_test_split(pin_test_split("synthetic", images), images)  # must not raise


@pytest.mark.parametrize(
    ("add", "remove", "expected"),
    [
        (("09_test_003",), (), "test: \\+1 -0"),
        ((), ("01_test_000",), "test: \\+0 -1"),
        # Same count, one member replaced -- the reshuffle a count check would miss.
        (("09_test_003",), ("01_test_000",), "test: \\+1 -1"),
    ],
    ids=["added", "removed", "swapped"],
)
def test_verify_test_split_raises_on_drift(
    tmp_path: Path, add: tuple[str, ...], remove: tuple[str, ...], expected: str
) -> None:
    images = _images_dir(tmp_path / "images", _HELD_OUT)
    pin = pin_test_split("synthetic", images)

    _images_dir(images, add)
    for stem in remove:
        (images / f"{stem}.jpg").unlink()

    with pytest.raises(SplitDriftError, match=expected):
        verify_test_split(pin, images)


def test_test_pin_write_and_load_round_trip(tmp_path: Path) -> None:
    pin = pin_test_split("synthetic", _images_dir(tmp_path / "images", _HELD_OUT))

    path = write_test_pin(pin, tmp_path / "splits" / "test" / "synthetic.json")

    assert load_test_pin(path) == pin


def test_write_test_pin_refuses_to_overwrite_a_different_pin(tmp_path: Path) -> None:
    images = _images_dir(tmp_path / "images", _HELD_OUT)
    path = write_test_pin(pin_test_split("synthetic", images), tmp_path / "synthetic.json")
    before = path.read_text()

    (images / "01_test_000.jpg").rename(images / "09_test_003.jpg")
    with pytest.raises(SplitDriftError, match="refusing to overwrite"):
        write_test_pin(pin_test_split("synthetic", images), path)

    assert path.read_text() == before


def test_write_test_pin_accepts_an_identical_pin(tmp_path: Path) -> None:
    pin = pin_test_split("synthetic", _images_dir(tmp_path / "images", _HELD_OUT))
    path = write_test_pin(pin, tmp_path / "synthetic.json")

    write_test_pin(pin, path)  # must not raise

    assert load_test_pin(path) == pin


def test_committed_train_val_records_are_unchanged_byte_for_byte() -> None:
    records = sorted(_REPO_SPLITS.glob("*.json"))
    assert records, f"no committed split records under {_REPO_SPLITS}"

    for path in records:
        text = path.read_text()
        data = json.loads(text)
        rebuilt = SplitManifest.from_dict(data)
        assert json.dumps(rebuilt.to_dict(), indent=2) + "\n" == text, path.name
        assert rebuilt.combined_hash == data["combined_hash"], path.name


def test_dataset_manifest_still_ignores_a_test_key(data_yaml: Path, tmp_path: Path) -> None:
    with_test = tmp_path / "data.yaml"
    with_test.write_text(data_yaml.read_text() + "test: val/visible/images\n")

    manifest = DatasetManifest.load(with_test)

    assert not hasattr(manifest, "test_images")
    assert freeze_split("synthetic", manifest) == freeze_split(
        "synthetic", DatasetManifest.load(data_yaml)
    )
