from __future__ import annotations

from pathlib import Path

import pytest

from t2o.data.pairing import Pairing


def test_infrared_path_substitutes_the_token() -> None:
    visible = Path("dataset/yolo_rgbt/train/visible/images/a.jpg")
    expected = Path("dataset/yolo_rgbt/train/infrared/images/a.jpg")
    assert Pairing().infrared_path(visible) == expected


def test_label_path_uses_the_visible_side() -> None:
    visible = Path("dataset/yolo_rgbt/train/visible/images/a.jpg")
    expected = Path("dataset/yolo_rgbt/train/visible/labels/a.txt")
    assert Pairing().label_path(visible) == expected


def test_custom_tokens_are_honoured() -> None:
    pairing = Pairing(visible_token="rgb", infrared_token="thermal")
    assert pairing.infrared_path(Path("d/rgb/images/a.png")) == Path("d/thermal/images/a.png")


def test_raises_when_the_token_is_absent() -> None:
    with pytest.raises(ValueError, match="found 0"):
        Pairing().infrared_path(Path("dataset/train/rgb/images/a.jpg"))


def test_raises_when_the_token_is_ambiguous() -> None:
    with pytest.raises(ValueError, match="found 2"):
        Pairing().infrared_path(Path("visible/train/visible/images/a.jpg"))


def test_label_path_raises_without_an_images_segment() -> None:
    with pytest.raises(ValueError, match="images"):
        Pairing().label_path(Path("dataset/train/visible/a.jpg"))


def test_validate_pairs_accepts_a_complete_dataset(dataset_root: Path) -> None:
    paths = sorted((dataset_root / "train" / "visible" / "images").iterdir())
    Pairing().validate_pairs(paths)


def test_validate_pairs_reports_every_missing_counterpart(tmp_path: Path) -> None:
    for modality in ("visible", "infrared"):
        (tmp_path / modality / "images").mkdir(parents=True)
    visible = []
    for i in range(4):
        path = tmp_path / "visible" / "images" / f"{i}.png"
        path.touch()
        visible.append(path)
    (tmp_path / "infrared" / "images" / "0.png").touch()  # only one of four exists

    with pytest.raises(FileNotFoundError, match=r"3/4 infrared counterparts missing") as exc:
        Pairing().validate_pairs(visible)
    # Every failure named, not just the first -- a half-synced tree should be diagnosable
    # from a single run.
    for i in (1, 2, 3):
        assert f"{i}.png" in str(exc.value)
