"""`data/splits.py` -- freezing, hashing, and drift-detecting a dataset's train/val
membership, against the synthetic `dataset_root` fixture (`tests/conftest.py`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from t2o.data.manifest import DatasetManifest
from t2o.data.splits import (
    SplitDriftError,
    SplitManifest,
    freeze_split,
    load_split_manifest,
    verify_split,
    write_split_manifest,
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
