"""`data/adapters/msrs.py` -- the pure mask -> YOLO-box function.

Synthetic masks only (the fixture discipline of PLAN.md §9): each test paints a few blobs on a
small canvas, so nothing here needs the real `dataset/raw/msrs` tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from t2o.data.adapters.msrs import MASK_TO_CLASS, MIN_BLOB_AREA, mask_to_yolo_lines

HEIGHT, WIDTH = 40, 80


def _canvas() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def _parse(line: str) -> tuple[int, tuple[float, ...]]:
    cls, *coords = line.split()
    return int(cls), tuple(float(c) for c in coords)


def test_empty_mask_has_no_boxes() -> None:
    assert mask_to_yolo_lines(_canvas()) == []


@pytest.mark.parametrize(("mask_value", "class_id"), sorted(MASK_TO_CLASS.items()))
def test_each_kept_class_is_remapped_not_passed_through(mask_value: int, class_id: int) -> None:
    mask = _canvas()
    mask[10:20, 20:40] = mask_value
    (line,) = mask_to_yolo_lines(mask)
    assert _parse(line)[0] == class_id


def test_box_is_normalised_to_the_mask_shape() -> None:
    mask = _canvas()
    mask[10:20, 20:40] = 1  # rows 10-19, cols 20-39: a 20 x 10 box at (20, 10)
    (line,) = mask_to_yolo_lines(mask)
    cls, coords = _parse(line)
    assert cls == 0
    assert coords == pytest.approx((30 / WIDTH, 15 / HEIGHT, 20 / WIDTH, 10 / HEIGHT))


@pytest.mark.parametrize("mask_value", [4, 6, 8])
def test_dropped_classes_write_nothing(mask_value: int) -> None:
    mask = _canvas()
    mask[10:20, 20:40] = mask_value
    assert mask_to_yolo_lines(mask) == []


def test_area_floor_edge_is_nine_dropped_ten_kept() -> None:
    assert MIN_BLOB_AREA == 10  # the test below hard-codes 9 / 10 around this value
    mask = _canvas()
    mask[5, 5:14] = 5  # 9 px
    mask[20, 5:15] = 5  # 10 px
    (line,) = mask_to_yolo_lines(mask)
    assert _parse(line)[1][1] == pytest.approx(20.5 / HEIGHT)  # the 10 px blob's row


def test_diagonal_neighbours_are_one_blob() -> None:
    mask = _canvas()
    for i in range(10):  # a 10 px diagonal line: one blob under 8-, ten under 4-connectivity
        mask[10 + i, 10 + i] = 7
    assert len(mask_to_yolo_lines(mask)) == 1


def test_separate_blobs_of_one_class_are_separate_boxes() -> None:
    mask = _canvas()
    mask[5:10, 5:10] = 2
    mask[25:30, 60:65] = 2
    assert len(mask_to_yolo_lines(mask)) == 2
