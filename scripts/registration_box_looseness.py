"""How much looseness does the registration axis's own box transform add at each level?

Record 040's degradation curve trains on boxes moved by ``misalign_label_lines``
(``src/t2o/data/misalignment.py``). It passes four corners through the level's roll homography,
takes the axis-aligned box around them, then clamps that box to the frame
(``mirror.deroll_label_lines``). Both steps can leave a box looser than its object:

- re-boxing rotated corners adds slack around anything that does not fill its box;
- the clamp cuts only the axis that crosses the frame edge, as the sister build's clamp does
  (``scripts/clipped_box_audit.py`` sizes that one).

If this looseness grows with the level, part of the curve's slope comes from the labels rather
than from the misalignment (``docs/features/clipped-box-audit/plan.md``, step 2).

The object inside a box is unknown, so each box is read as the hull of two bounding shapes:

- **diagonal** -- a thin segment from corner to corner, the worst case. It is scored by the
  worse of the box's two diagonals;
- **rectangle** -- an object filling its box, the best case. Away from the frame edges it gives
  IoU 1 at every level, by construction.

For each shape the *tight* box is the box around the warped object's part inside the frame. The
CSV holds, per level and shape, the mean IoU between the written box and the tight box, and the
share of boxes the clamp cut. That is a band of label looseness to set beside record 040's curve.

**The written box is the axis's own output.** It comes from ``misalign_label_lines`` itself,
six-decimal rounding included, not from a re-implementation. Each box goes through in a file of
its own, because the function reports how many boxes it dropped but not which.

Standalone script, not part of the ``t2o`` package -- it owns its own ``logging.basicConfig``, as
``scripts/clipped_box_audit.py`` does.
"""

from __future__ import annotations

import argparse
import csv
import logging
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

import numpy as np

from t2o.data.calibration import FloatArray
from t2o.data.labels import load_yolo_labels
from t2o.data.mirror import _map_points, _split_image_shape
from t2o.data.misalignment import REGISTRATION_LEVELS_PX, Misalignment, misalign_label_lines
from t2o.data.pairing import LABEL_SUFFIX

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# The custom set's thermal frame, as record 036's sweep log reads it ("custom at 640x480"). The
# synthetic run only means something at the frame the levels were trained on.
SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH = 480, 640
# Synthetic box centres, as a fraction of the frame on each axis. The outer ring sits close
# enough to the edges for the larger levels to push corners past them.
SYNTHETIC_CENTRES = tuple(float(c) for c in np.linspace(0.1, 0.9, 9))
# Synthetic box sides, as a fraction of the frame: a small and a mid-sized object.
SYNTHETIC_SIDES = (0.05, 0.15)

YoloBox = tuple[float, float, float, float]  # normalised cx, cy, w, h
PixelBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


class LoosenessError(ValueError):
    """Raised for an input with no boxes to measure."""


class Shape(StrEnum):
    """The bounding object shape a box is read as."""

    DIAGONAL = "diagonal"
    RECTANGLE = "rectangle"


@dataclass(frozen=True, slots=True)
class LevelRow:
    """One CSV row: one level, one shape."""

    level_px: float
    misalignment_digest: str
    shape: str
    boxes: int  # boxes in
    dropped: int  # written box dropped by the axis: wholly off the frame
    clamped: int  # boxes whose warped corners crossed a frame edge, dropped ones included
    mean_iou: float | None  # over the boxes the axis kept; None when it kept none
    clamp_rate: float  # clamped / boxes


def synthetic_boxes() -> list[YoloBox]:
    """A grid of square-ish boxes over the frame: every centre at every side length."""
    return [
        (cx, cy, side, side)
        for side in SYNTHETIC_SIDES
        for cy in SYNTHETIC_CENTRES
        for cx in SYNTHETIC_CENTRES
    ]


def read_label_dir(labels_dir: Path) -> list[YoloBox]:
    """Every box in ``labels_dir``'s label files, through the same loader the axis reads with."""
    boxes: list[YoloBox] = []
    for path in sorted(labels_dir.glob(f"*{LABEL_SUFFIX}")):
        _, table = load_yolo_labels(path)
        boxes.extend((cx, cy, w, h) for cx, cy, w, h in table.tolist())
    if not boxes:
        raise LoosenessError(f"{labels_dir} holds no boxes in any *{LABEL_SUFFIX} file")
    return boxes


def _to_pixels(box: YoloBox, height: int, width: int) -> PixelBox:
    cx, cy, w, h = box
    return (
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    )


def written_box(box: YoloBox, misalignment: Misalignment, scratch: Path) -> PixelBox | None:
    """The box ``misalign_label_lines`` writes for ``box``, or None if it drops it."""
    label = scratch / f"box{LABEL_SUFFIX}"
    label.write_text("0 " + " ".join(repr(value) for value in box) + "\n")
    lines, _ = misalign_label_lines(label, misalignment)
    if not lines:
        return None
    _, cx, cy, w, h = (float(value) for value in lines[0].split())
    return _to_pixels((cx, cy, w, h), misalignment.height, misalignment.width)


def _clip_to_frame(points: FloatArray, height: int, width: int) -> FloatArray:
    """Sutherland-Hodgman clip of a closed polygon to the frame; may come back empty.

    A two-point "polygon" is a segment walked there and back, so the same clip serves both
    shapes: what survives is the visible part of the segment.
    """
    clipped = list(points)
    for axis, limit, upper in (
        (0, 0.0, False),
        (0, width, True),
        (1, 0.0, False),
        (1, height, True),
    ):
        source, clipped = clipped, []
        for current, following in zip(source, source[1:] + source[:1], strict=True):
            current_in = current[axis] <= limit if upper else current[axis] >= limit
            following_in = following[axis] <= limit if upper else following[axis] >= limit
            if current_in:
                clipped.append(current)
            if current_in != following_in:
                t = (limit - current[axis]) / (following[axis] - current[axis])
                clipped.append(current + t * (following - current))
    return np.asarray(clipped, dtype=np.float64).reshape(-1, 2)


def tight_box(outline: FloatArray, misalignment: Misalignment) -> PixelBox | None:
    """The box around the warped object's visible part, or None if none of it is visible."""
    moved = _map_points(outline, misalignment.homography())
    visible = _clip_to_frame(moved, misalignment.height, misalignment.width)
    if len(visible) == 0:
        return None
    (x0, y0), (x1, y1) = visible.min(axis=0), visible.max(axis=0)
    return (float(x0), float(y0), float(x1), float(y1))


def outlines(box: PixelBox) -> dict[Shape, list[FloatArray]]:
    """Each shape's candidate objects inside ``box``: two diagonals, or the box itself."""
    x0, y0, x1, y1 = box
    return {
        Shape.DIAGONAL: [np.array([[x0, y0], [x1, y1]]), np.array([[x1, y0], [x0, y1]])],
        Shape.RECTANGLE: [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])],
    }


def iou(written: PixelBox, tight: PixelBox | None) -> float:
    """IoU of two pixel boxes. A written box around nothing visible scores 0.

    ``written`` always has positive area -- ``deroll_label_lines`` drops the rest -- so the
    union is never zero.
    """
    if tight is None:
        return 0.0
    inter_w = max(0.0, min(written[2], tight[2]) - max(written[0], tight[0]))
    inter_h = max(0.0, min(written[3], tight[3]) - max(written[1], tight[1]))
    inter = inter_w * inter_h
    area_written = (written[2] - written[0]) * (written[3] - written[1])
    area_tight = (tight[2] - tight[0]) * (tight[3] - tight[1])
    return inter / (area_written + area_tight - inter)


def _is_clamped(box: PixelBox, misalignment: Misalignment) -> bool:
    """Whether the axis's warped corners cross a frame edge, so its clamp cuts the box."""
    (corners,) = outlines(box)[Shape.RECTANGLE]
    moved = _map_points(corners, misalignment.homography())
    return bool(
        moved.min() < 0
        or moved[:, 0].max() > misalignment.width
        or moved[:, 1].max() > misalignment.height
    )


def measure_level(
    boxes: Sequence[YoloBox], misalignment: Misalignment, scratch: Path
) -> list[LevelRow]:
    """One row per shape for one level."""
    ious: dict[Shape, list[float]] = {shape: [] for shape in Shape}
    dropped = clamped = 0
    for box in boxes:
        pixels = _to_pixels(box, misalignment.height, misalignment.width)
        clamped += _is_clamped(pixels, misalignment)
        written = written_box(box, misalignment, scratch)
        if written is None:
            dropped += 1
            continue
        for shape, candidates in outlines(pixels).items():
            ious[shape].append(min(iou(written, tight_box(c, misalignment)) for c in candidates))
    return [
        LevelRow(
            level_px=misalignment.level_px,
            misalignment_digest=misalignment.digest(),
            shape=shape.value,
            boxes=len(boxes),
            dropped=dropped,
            clamped=clamped,
            mean_iou=float(np.mean(values)) if values else None,
            clamp_rate=clamped / len(boxes),
        )
        for shape, values in ious.items()
    ]


def sweep(
    boxes: Sequence[YoloBox], height: int, width: int, levels: Sequence[float]
) -> list[LevelRow]:
    """Every level's rows, in the order ``levels`` gives."""
    rows: list[LevelRow] = []
    with tempfile.TemporaryDirectory() as scratch:
        for level in levels:
            rows.extend(measure_level(boxes, Misalignment(height, width, level), Path(scratch)))
    return rows


def write_csv(rows: Sequence[LevelRow], path: Path) -> None:
    """One row per level and shape. ``None`` is written as an empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(LevelRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in asdict(row).items()})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """``--labels`` reads real boxes; without it the synthetic grid is measured."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="a split's infrared/labels dir; the frame size is read from its sibling images/",
    )
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    parser.add_argument("--levels", type=float, nargs="+", default=list(REGISTRATION_LEVELS_PX))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure every level and shape, then write the CSV and log each row."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    if args.labels is None:
        boxes, (height, width) = synthetic_boxes(), (SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH)
        logger.info("synthetic grid: %d boxes at %dx%d", len(boxes), width, height)
    else:
        boxes = read_label_dir(args.labels)
        # The images, not a flag: a level is pixels on the frame it was trained at.
        height, width = _split_image_shape(args.labels.parent / "images")
        logger.info("%s: %d boxes at %dx%d", args.labels, len(boxes), width, height)
    rows = sweep(boxes, height, width, args.levels)
    write_csv(rows, args.out)
    for row in rows:
        logger.info(
            "%5.1f px %-9s mean IoU %.4f | clamped %d/%d (%.3f) | dropped %d",
            row.level_px,
            row.shape,
            float("nan") if row.mean_iou is None else row.mean_iou,
            row.clamped,
            row.boxes,
            row.clamp_rate,
            row.dropped,
        )
    logger.info("wrote %d rows to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
