"""`scripts/freeze_splits.py` -- dataset discovery, freeze/overwrite, and `--check` mode.

Builds a tiny `<data-root>/synthetic/data.yaml` pointing at the shared synthetic
`dataset_root` fixture (same trick as the other adapter-facing tests: reuse the fixture's
`path:` rather than duplicating image data), so nothing here touches real datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from scripts.freeze_splits import build_parser, discover_datasets, freeze_one, main

from t2o.data.manifest import DatasetManifest
from t2o.data.splits import SplitDriftError, freeze_split, load_split_manifest


@pytest.fixture
def data_root(data_yaml: Path, tmp_path: Path) -> Path:
    root = tmp_path / "data_root"
    target = root / "synthetic" / "data.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(data_yaml.read_text())
    return root


def test_discover_datasets_finds_every_data_yaml(data_root: Path) -> None:
    assert discover_datasets(data_root) == ["synthetic"]


def test_discover_datasets_is_empty_for_a_root_with_no_data_yaml(tmp_path: Path) -> None:
    assert discover_datasets(tmp_path / "nothing_here") == []


def test_freeze_one_writes_a_record_matching_freeze_split(data_root: Path, tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"

    freeze_one("synthetic", data_root, splits_root, check=False)

    manifest = DatasetManifest.load(data_root / "synthetic" / "data.yaml")
    expected = freeze_split("synthetic", manifest)
    assert load_split_manifest(splits_root / "synthetic.json") == expected


def test_check_mode_raises_without_an_existing_record(data_root: Path, tmp_path: Path) -> None:
    with pytest.raises(SplitDriftError, match="no frozen record"):
        freeze_one("synthetic", data_root, tmp_path / "splits", check=True)


def test_check_mode_passes_when_matching(data_root: Path, tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"
    freeze_one("synthetic", data_root, splits_root, check=False)

    freeze_one("synthetic", data_root, splits_root, check=True)  # must not raise


def test_check_mode_raises_on_drift(data_root: Path, tmp_path: Path, dataset_root: Path) -> None:
    splits_root = tmp_path / "splits"
    freeze_one("synthetic", data_root, splits_root, check=False)

    extra = dataset_root / "train" / "visible" / "images" / "99_extra_999.jpg"
    extra.write_bytes(b"only the stem matters")
    try:
        with pytest.raises(SplitDriftError, match="no longer matches"):
            freeze_one("synthetic", data_root, splits_root, check=True)
    finally:
        extra.unlink()


def test_overwriting_a_drifted_record_warns_but_does_not_raise(
    data_root: Path, tmp_path: Path, dataset_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    splits_root = tmp_path / "splits"
    freeze_one("synthetic", data_root, splits_root, check=False)

    extra = dataset_root / "train" / "visible" / "images" / "99_extra_999.jpg"
    extra.write_bytes(b"only the stem matters")
    try:
        with caplog.at_level(logging.WARNING):
            freeze_one("synthetic", data_root, splits_root, check=False)  # must not raise
        assert "overwriting a drifted frozen record" in caplog.text

        manifest = DatasetManifest.load(data_root / "synthetic" / "data.yaml")
        assert load_split_manifest(splits_root / "synthetic.json") == freeze_split(
            "synthetic", manifest
        )
    finally:
        extra.unlink()


def test_main_exits_when_nothing_is_discovered(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no data\\.yaml found"):
        main(["--data-root", str(tmp_path / "empty")])


def test_main_freezes_every_discovered_dataset(data_root: Path, tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"

    exit_code = main(["--data-root", str(data_root), "--splits-root", str(splits_root)])

    assert exit_code == 0
    assert (splits_root / "synthetic.json").is_file()


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.dataset is None
    assert args.data_root == Path("dataset/processed")
    assert args.splits_root == Path("splits")
    assert args.check is False
