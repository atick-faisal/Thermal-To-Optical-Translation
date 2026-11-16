"""`scripts/registration_box_looseness.py` -- the looseness the registration axis adds per level.

The three properties the task names, plus one CLI run on a hand-built label folder. The diagonal
case has a closed form under a pure roll -- IoU (cos t - sin t) / (cos t + sin t) for a square --
so "falls as the level rises" is a property of the geometry, not of this code.
"""

from __future__ import annotations

import csv
from itertools import pairwise
from pathlib import Path

import pytest
from PIL import Image
from scripts.registration_box_looseness import (
    SYNTHETIC_HEIGHT,
    SYNTHETIC_WIDTH,
    LevelRow,
    Shape,
    main,
    sweep,
    synthetic_boxes,
)

from t2o.data.misalignment import REGISTRATION_LEVELS_PX

# One square-ish box in the middle of the frame: no level pushes it near an edge.
CENTRE_BOX = (0.5, 0.5, 0.2, 0.2)
# Six-decimal label rounding and float32 parsing leave this much slack on an exact IoU of 1.
ROUNDING = 1e-4


def _iou(rows: list[LevelRow], shape: Shape) -> list[float]:
    values = [row.mean_iou for row in rows if row.shape == shape]
    assert all(value is not None for value in values)
    return [value for value in values if value is not None]


def test_level_zero_gives_iou_one() -> None:
    rows = sweep(synthetic_boxes(), SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH, (0.0,))

    assert {row.shape for row in rows} == {Shape.DIAGONAL, Shape.RECTANGLE}
    for row in rows:
        assert row.mean_iou == pytest.approx(1.0, abs=ROUNDING)
        assert (row.clamped, row.dropped) == (0, 0)


def test_interior_rectangle_gives_iou_one_at_every_level() -> None:
    rows = sweep([CENTRE_BOX], SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH, REGISTRATION_LEVELS_PX)

    assert _iou(rows, Shape.RECTANGLE) == pytest.approx(
        [1.0] * len(REGISTRATION_LEVELS_PX), abs=ROUNDING
    )
    assert all(row.clamp_rate == 0 for row in rows)


def test_diagonal_iou_falls_as_the_level_rises() -> None:
    rows = sweep([CENTRE_BOX], SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH, REGISTRATION_LEVELS_PX)

    diagonal = _iou(rows, Shape.DIAGONAL)
    assert all(later < earlier for earlier, later in pairwise(diagonal))
    # Not a rounding-sized fall: at 24 px the roll is ~3.4 degrees, worth ~0.11 IoU.
    assert diagonal[-1] < 0.95


def test_cli_reads_a_label_dir_and_counts_the_clamp(tmp_path: Path) -> None:
    images, labels = tmp_path / "val/infrared/images", tmp_path / "val/infrared/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("L", (64, 48)).save(images / "a.png")
    # One box in the corner, which a 24 px roll on a 64x48 frame pushes past the edge, and one
    # in the middle, which stays inside.
    (labels / "a.txt").write_text("0 0.1 0.1 0.1 0.1\n1 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "looseness.csv"

    assert main(["--labels", str(labels), "--out", str(out), "--levels", "0", "24"]) == 0

    with out.open() as handle:
        written = list(csv.DictReader(handle))
    assert [(row["level_px"], row["shape"]) for row in written] == [
        ("0.0", "diagonal"),
        ("0.0", "rectangle"),
        ("24.0", "diagonal"),
        ("24.0", "rectangle"),
    ]
    assert {row["boxes"] for row in written} == {"2"}
    assert [row["clamped"] for row in written] == ["0", "0", "1", "1"]
