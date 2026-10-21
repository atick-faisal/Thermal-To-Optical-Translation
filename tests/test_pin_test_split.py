"""`scripts/pin_test_split.py` -- writing the pin, `--check`, and refusing to re-pin a
reshuffled test split. Stub files in `tmp_path` stand in for the images: only stems matter."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.pin_test_split import build_parser, main, pin_one, pin_path

from t2o.data.splits import SplitDriftError, load_test_pin, pin_test_split

_STEMS = ("03_test_002", "01_test_000", "07_test_001")


@pytest.fixture
def images(tmp_path: Path) -> Path:
    root = tmp_path / "test" / "visible" / "images"
    root.mkdir(parents=True)
    for stem in _STEMS:
        (root / f"{stem}.jpg").write_bytes(b"")
    return root


def test_pin_one_writes_a_pin_matching_pin_test_split(images: Path, tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"

    pin_one("synthetic", images, splits_root, check=False)

    assert load_test_pin(splits_root / "test" / "synthetic.json") == pin_test_split(
        "synthetic", images
    )


def test_the_pin_sits_outside_the_train_val_glob(images: Path, tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"

    pin_one("synthetic", images, splits_root, check=False)

    assert list(splits_root.glob("*.json")) == []
    assert pin_path("synthetic", splits_root).is_file()


def test_check_mode_raises_without_an_existing_pin(images: Path, tmp_path: Path) -> None:
    with pytest.raises(SplitDriftError, match="no test pin"):
        pin_one("synthetic", images, tmp_path / "splits", check=True)


def test_check_mode_passes_when_matching(images: Path, tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"
    pin_one("synthetic", images, splits_root, check=False)

    pin_one("synthetic", images, splits_root, check=True)  # must not raise


def test_check_mode_raises_on_a_swapped_stem(images: Path, tmp_path: Path) -> None:
    splits_root = tmp_path / "splits"
    pin_one("synthetic", images, splits_root, check=False)

    (images / "01_test_000.jpg").rename(images / "09_test_003.jpg")

    with pytest.raises(SplitDriftError, match="test: \\+1 -1"):
        pin_one("synthetic", images, splits_root, check=True)


def test_re_pinning_a_reshuffled_split_raises_and_keeps_the_pin(
    images: Path, tmp_path: Path
) -> None:
    splits_root = tmp_path / "splits"
    pin_one("synthetic", images, splits_root, check=False)
    before = pin_path("synthetic", splits_root).read_text()

    (images / "01_test_000.jpg").rename(images / "09_test_003.jpg")

    with pytest.raises(SplitDriftError, match="refusing to overwrite"):
        pin_one("synthetic", images, splits_root, check=False)
    assert pin_path("synthetic", splits_root).read_text() == before


def test_main_pins_then_checks(images: Path, tmp_path: Path) -> None:
    argv = ["--name", "synthetic", "--images", str(images), "--splits-root", str(tmp_path)]

    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0


def test_build_parser_requires_name_and_images() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])

    args = build_parser().parse_args(["--name", "x", "--images", "d"])
    assert args.splits_root == Path("splits")
    assert args.check is False
