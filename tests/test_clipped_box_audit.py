"""`scripts/clipped_box_audit.py` -- the custom set's warp-and-clamp, recomputed per box.

Every case is a hand-built tree in the sister build's layout: `yolo_original/<split>/{images,
labels}`, `alignment/<stem>.json` and a frozen `<split>/infrared/labels`. The centre-crop case
has answers worked out by hand, so the warp is checked against arithmetic, not against itself.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scripts.clipped_box_audit import (
    Alignment,
    AuditConfig,
    AuditFidelityError,
    Box,
    main,
    warp_box,
)

CONFIG = AuditConfig(
    min_visibility=0.25, min_box_px=2.0, prefix_sep="-", strip_suffix="_embedded", stem_regex=None
)
# A 200x100 optical frame cropped to its 100x100 centre: a pure shift of -50 px in x.
CENTRE_CROP = np.array([[1.0, 0.0, -50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
STEM = "FLIR0001"
ORIGINAL_NAME = f"9a8b7c-{STEM}_embedded"


def _pixels_to_box(cls_id: int, x0: float, y0: float, x1: float, y1: float) -> Box:
    """A box given in 200x100 optical pixels, normalised the YOLO way."""
    return Box(cls_id, (x0 + x1) / 2 / 200, (y0 + y1) / 2 / 100, (x1 - x0) / 200, (y1 - y0) / 100)


# A diagonal pole from (10, 80) to (90, 0): after the crop it runs from (-40, 80) to (40, 0), so
# it enters the frame at (0, 40) and the clamped box keeps half its width but all its height.
POLE = _pixels_to_box(0, 10, 0, 90, 80)
OFF_FRAME = _pixels_to_box(1, 150, 10, 190, 30)  # lands at x 100..140: wholly outside
SLIVER = _pixels_to_box(2, 10, 10, 55, 30)  # lands at x -40..5: keeps 5/45 of its area
INSIDE = _pixels_to_box(1, 70, 10, 90, 30)  # lands at x 20..40: untouched
SOURCE_BOXES = (POLE, OFF_FRAME, SLIVER, INSIDE)
# The two survivors, worked out by hand in 100x100 thermal pixels.
FROZEN = ["0 0.200000 0.400000 0.400000 0.800000", "1 0.300000 0.200000 0.200000 0.200000"]


def _write_tree(root: Path, frozen: list[str]) -> list[str]:
    """Write the three inputs for one val image; return the CLI arguments that read them."""
    original, alignment, frozen_dir = root / "yolo_original", root / "alignment", root / "frozen"
    (original / "val" / "images").mkdir(parents=True)
    (original / "val" / "labels").mkdir(parents=True)
    (original / "val" / "images" / f"{ORIGINAL_NAME}.jpg").write_bytes(b"")
    (original / "val" / "labels" / f"{ORIGINAL_NAME}.txt").write_text(
        "".join(f"{box.to_line()}\n" for box in SOURCE_BOXES)
    )
    alignment.mkdir()
    (alignment / f"{STEM}.json").write_text(
        json.dumps(
            {
                "homography": CENTRE_CROP.tolist(),
                "optical_size": {"width": 200, "height": 100},
                "thermal_size": {"width": 100, "height": 100},
            }
        )
    )
    labels = frozen_dir / "val" / "infrared" / "labels"
    labels.mkdir(parents=True)
    (labels / f"{STEM}.txt").write_text("".join(f"{line}\n" for line in frozen))
    return [
        "--original-dir", str(original),
        "--alignment-dir", str(alignment),
        "--frozen-dir", str(frozen_dir),
        "--out", str(root / "audit.csv"),
        "--splits", "val",
    ]  # fmt: skip


def _crop() -> Alignment:
    return Alignment(CENTRE_CROP, (200, 100), (100, 100))


def test_identity_keeps_the_box_untouched() -> None:
    box = Box(3, 0.5, 0.5, 0.2, 0.4)
    warp = warp_box(box, Alignment(np.eye(3), (100, 100), (100, 100)), CONFIG)

    assert warp.box is not None
    assert warp.box.to_line() == box.to_line()
    assert warp.retained_fraction == pytest.approx(1.0)
    assert warp.quad_fill == pytest.approx(1.0)
    assert warp.cut_edges == ()
    assert warp.drop_reason == ""


def test_clamped_pole_keeps_half_and_its_tight_iou_is_that_half() -> None:
    warp = warp_box(POLE, _crop(), CONFIG)

    assert warp.box is not None
    assert warp.box.to_line() == FROZEN[0]
    assert warp.cut_edges == ("left",)
    assert warp.retained_fraction == pytest.approx(0.5)
    # The pole's surviving segment runs (0, 40)-(40, 0): its tight box is 40x40 inside the
    # clamped 40x80. IoU = 0.5 = the retained fraction -- the plan's central claim, worked out.
    tight_area, clamped_area = 40 * 40, warp.box.w * 100 * warp.box.h * 100
    assert tight_area / clamped_area == pytest.approx(warp.retained_fraction)


def test_drop_reasons_follow_the_build_order() -> None:
    off_frame = warp_box(OFF_FRAME, _crop(), CONFIG)
    sliver = warp_box(SLIVER, _crop(), CONFIG)

    assert (off_frame.box, off_frame.drop_reason) == (None, "min_box_px")
    assert off_frame.retained_fraction == pytest.approx(0.0)
    assert (sliver.box, sliver.drop_reason) == (None, "min_visibility")
    assert sliver.retained_fraction == pytest.approx(5 / 45)


def test_roll_lowers_quad_fill_without_any_clamp() -> None:
    # A centred 20x20 box rolled 30 degrees about the centre of a 100x100 frame. A rotated square
    # fills 1 / (cos + sin)^2 of its axis-aligned box.
    angle = math.radians(30)
    cos, sin = math.cos(angle), math.sin(angle)
    to_origin = np.array([[1, 0, -50], [0, 1, -50], [0, 0, 1]], dtype=np.float64)
    rotate = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)
    roll = np.linalg.inv(to_origin) @ rotate @ to_origin

    warp = warp_box(Box(0, 0.5, 0.5, 0.2, 0.2), Alignment(roll, (100, 100), (100, 100)), CONFIG)

    assert warp.cut_edges == ()
    assert warp.retained_fraction == pytest.approx(1.0)
    assert warp.quad_fill == pytest.approx(1 / (cos + sin) ** 2)


def test_main_writes_one_row_per_source_box(tmp_path: Path) -> None:
    args = _write_tree(tmp_path, FROZEN)

    assert main(args) == 0

    with (tmp_path / "audit.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["stem"] for row in rows] == [STEM] * 4
    assert [row["drop_reason"] for row in rows] == ["", "min_box_px", "min_visibility", ""]
    assert [row["frozen_line"] for row in rows] == ["0", "", "", "1"]
    assert [row["cut_edges"] for row in rows] == ["left", "right", "left", ""]
    assert float(rows[0]["retained_fraction"]) == pytest.approx(0.5)


def test_a_frozen_label_that_disagrees_stops_the_audit(tmp_path: Path) -> None:
    args = _write_tree(tmp_path, [FROZEN[0], "1 0.300000 0.200000 0.200000 0.200001"])

    with pytest.raises(AuditFidelityError, match="1 label files disagree"):
        main(args)
    assert not (tmp_path / "audit.csv").exists()


def test_a_frozen_stem_without_an_original_stops_the_audit(tmp_path: Path) -> None:
    args = _write_tree(tmp_path, FROZEN)
    (tmp_path / "frozen" / "val" / "infrared" / "labels" / "FLIR9999.txt").write_text("")

    with pytest.raises(AuditFidelityError):
        main(args)
