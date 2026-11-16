"""How loose did the custom set's boxes get when they were warped onto the thermal frame?

The custom set's boxes were drawn on the optical image, whose field of view is wider than the
thermal camera's. The sister build script moves them across in
``alignment-scripts/build_rgbt_dataset.py:235-295`` (``warp_box``): four corners through the
per-image RoMA homography, the axis-aligned box around them, a clamp to the thermal frame, then a
drop when less than ``--min-visibility`` of the area survives.

The clamp cuts only the axis that crosses the frame edge. The other axis keeps its full extent,
so a thin diagonal object -- a pole -- no longer reaches the clamped box's corners. For such an
object cut on one side, the **retained-area fraction equals the best IoU any detector can reach**
against a tight box: a box kept at 0.4 cannot be matched at IoU 0.5 by a perfect detector.
Re-boxing the warped corners adds a second, smaller looseness when the homography rolls; the
``quad_fill`` column (warped-quad area over box area, 1 for no roll) measures it.

This script recomputes that warp for every source box and writes one CSV row per box, so the
problem can be sized by class and split before any label changes
(``docs/features/clipped-box-audit/plan.md``).

**Why re-implement ``warp_box`` rather than import it** (plan Q1). The sister repo has no
``just verify`` gate, and this repo's tests could not reach it. Instead the re-implementation is
held to the sister's output: every recomputed kept box must reproduce the frozen tree's label
line **string for string**, and every stem in the frozen tree must be accounted for. Any
difference -- rounding, operation order, a different ``--min-visibility`` at build time, the wrong
``--original-dir`` -- raises :class:`AuditFidelityError` before a CSV is written. The arithmetic
below therefore follows the sister's line by line, including its ``np.clip`` calls on the
normalised output.

Standalone script, not part of the ``t2o`` package -- it owns its own ``logging.basicConfig``, as
``scripts/registration_report.py`` does.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# The sister's `IMAGE_SUFFIXES` (build_rgbt_dataset.py:53). A suffix it scanned and this script
# skipped would leave that image's frozen labels unaccounted for -- a fidelity failure.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})
LABEL_FIELDS = 5  # class cx cy w h
# Where the frozen tree keeps the thermal-side labels. The sister writes the same text to
# `visible/labels` and `infrared/labels` (build_rgbt_dataset.py:383-387), so either would do.
FROZEN_LABELS = Path("infrared") / "labels"
# Retained-fraction cut-offs the summary log counts below. 0.5 is the mAP50 matching limit for
# a thin diagonal object; 0.75 is where mAP50-95's stricter IoU levels start to lose it.
SUMMARY_THRESHOLDS = (0.5, 0.75)
# How many fidelity mismatches to spell out before the count; the rest are only counted.
MISMATCHES_SHOWN = 10


class AuditError(RuntimeError):
    """Raised for inputs the audit cannot read at all."""


class AuditFidelityError(AuditError):
    """Raised when the recomputed boxes do not reproduce the frozen tree's labels."""


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """The sister build's settings. They must match the ones the frozen tree was built with."""

    # A box whose surviving area fraction is below this is dropped. The sister default is 0.25;
    # a build run at another value shows up as a fidelity failure, not as silently wrong rows.
    min_visibility: float
    # A box narrower or shorter than this many thermal pixels after the clamp is dropped.
    min_box_px: float
    # `0284327weq34-FLIR0240_embedded` -> `FLIR0240`: split once on this, keep the right side.
    prefix_sep: str
    # ...then strip this suffix. A wrong value finds no alignment file and skips every image.
    strip_suffix: str
    # Overrides the two rules above; needs a `(?P<stem>...)` group, as in the sister.
    stem_regex: re.Pattern[str] | None


@dataclass(frozen=True, slots=True)
class Box:
    """One YOLO box, normalised to its image."""

    cls_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        """The sister's label line format (build_rgbt_dataset.py ``Box.to_line``)."""
        return f"{self.cls_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass(frozen=True, slots=True)
class Alignment:
    """One image's homography, original-optical pixels to native thermal pixels."""

    matrix: npt.NDArray[np.float64]
    optical_size: tuple[int, int]  # (width, height)
    thermal_size: tuple[int, int]  # (width, height)


@dataclass(frozen=True, slots=True)
class Warp:
    """What happened to one box on its way onto the thermal frame."""

    box: Box | None  # the kept box, or None when dropped
    # Clamped area over pre-clamp area. None when the warp itself was unusable.
    retained_fraction: float | None
    cut_edges: tuple[str, ...]  # frame edges the pre-clamp box crossed
    # Warped-quad area over its axis-aligned box's area: 1 with no roll, lower with more.
    quad_fill: float | None
    # Empty when kept; else nonfinite | degenerate | min_box_px | min_visibility.
    drop_reason: str


@dataclass(frozen=True, slots=True)
class BoxRow:
    """One CSV row: one source box."""

    split: str
    stem: str
    box_index: int  # position in the original label file, counting parsed boxes only
    class_id: int
    retained_fraction: float | None
    cut_edges: str  # "+"-joined, e.g. "left+top"; empty when uncut
    quad_fill: float | None
    drop_reason: str
    in_frozen_tree: bool
    # Line index in the frozen label file; None when dropped or the stem is not in the tree.
    frozen_line: int | None


def parse_stem(filename: str, config: AuditConfig) -> str:
    """The sister's ``parse_stem``: ``0284327weq34-FLIR0240_embedded.jpg`` -> ``FLIR0240``."""
    name = Path(filename).stem
    if config.stem_regex is not None:
        match = config.stem_regex.search(name)
        if match is None:
            raise AuditError(f"--stem-regex did not match {filename!r}")
        return match.group("stem")
    if config.prefix_sep and config.prefix_sep in name:
        name = name.split(config.prefix_sep, 1)[1]
    if config.strip_suffix and name.endswith(config.strip_suffix):
        name = name[: -len(config.strip_suffix)]
    if not name:
        raise AuditError(f"stem parsing produced an empty name for {filename!r}")
    return name


def read_labels(path: Path) -> list[Box]:
    """The sister's ``read_labels``: short or unparseable lines are skipped, long ones truncated.

    Skipping the same lines matters: ``box_index`` counts parsed boxes, as the sister's loop did.
    """
    if not path.is_file():
        return []
    boxes: list[Box] = []
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        values = raw_line.split()
        if not values:
            continue
        if len(values) < LABEL_FIELDS:
            logger.warning("%s:%d: %d fields, skipping as the build did", path, lineno, len(values))
            continue
        try:
            boxes.append(
                Box(
                    cls_id=int(float(values[0])),
                    cx=float(values[1]),
                    cy=float(values[2]),
                    w=float(values[3]),
                    h=float(values[4]),
                )
            )
        except ValueError:
            logger.warning("%s:%d: unparseable numbers, skipping as the build did", path, lineno)
    return boxes


def load_alignment(path: Path) -> Alignment:
    """Read ``alignment/<stem>.json`` as written by ``alignment-scripts/align_flir.py``."""
    record = json.loads(path.read_text())
    matrix = np.asarray(record["homography"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise AuditError(f"{path}: homography is {matrix.shape}, expected (3, 3)")
    optical = record["optical_size"]
    thermal = record["thermal_size"]
    return Alignment(
        matrix=matrix,
        optical_size=(int(optical["width"]), int(optical["height"])),
        thermal_size=(int(thermal["width"]), int(thermal["height"])),
    )


def _quad_area(points: npt.NDArray[np.floating | np.integer]) -> float:
    """Shoelace area of the four warped corners, in order."""
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def warp_box(box: Box, alignment: Alignment, config: AuditConfig) -> Warp:
    """The sister's ``warp_box`` (build_rgbt_dataset.py:235-295), plus what it measured but dropped.

    The arithmetic is kept in the sister's order so the kept box's six-decimal line matches.
    """
    width_optical, height_optical = alignment.optical_size
    width_thermal, height_thermal = alignment.thermal_size

    half_w = box.w * width_optical / 2.0
    half_h = box.h * height_optical / 2.0
    center_x = box.cx * width_optical
    center_y = box.cy * height_optical
    corners = np.array(
        [
            [center_x - half_w, center_y - half_h],
            [center_x + half_w, center_y - half_h],
            [center_x + half_w, center_y + half_h],
            [center_x - half_w, center_y + half_h],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    warped = cv2.perspectiveTransform(corners, alignment.matrix).reshape(-1, 2)
    if not np.all(np.isfinite(warped)):
        return Warp(None, None, (), None, "nonfinite")

    x_min, y_min = warped.min(axis=0)
    x_max, y_max = warped.max(axis=0)
    area_before = (x_max - x_min) * (y_max - y_min)
    if area_before <= 0.0:
        return Warp(None, None, (), None, "degenerate")

    clipped_x_min = float(np.clip(x_min, 0.0, width_thermal))
    clipped_x_max = float(np.clip(x_max, 0.0, width_thermal))
    clipped_y_min = float(np.clip(y_min, 0.0, height_thermal))
    clipped_y_max = float(np.clip(y_max, 0.0, height_thermal))
    clipped_w = clipped_x_max - clipped_x_min
    clipped_h = clipped_y_max - clipped_y_min

    cut_edges = tuple(
        edge
        for edge, crossed in (
            ("left", x_min < 0.0),
            ("right", x_max > width_thermal),
            ("top", y_min < 0.0),
            ("bottom", y_max > height_thermal),
        )
        if crossed
    )
    retained = float((clipped_w * clipped_h) / area_before)
    quad_fill = _quad_area(warped) / float(area_before)

    if clipped_w < config.min_box_px or clipped_h < config.min_box_px:
        return Warp(None, retained, cut_edges, quad_fill, "min_box_px")
    if retained < config.min_visibility:
        return Warp(None, retained, cut_edges, quad_fill, "min_visibility")

    kept = Box(
        cls_id=box.cls_id,
        cx=float(np.clip((clipped_x_min + clipped_x_max) / 2.0 / width_thermal, 0.0, 1.0)),
        cy=float(np.clip((clipped_y_min + clipped_y_max) / 2.0 / height_thermal, 0.0, 1.0)),
        w=float(np.clip(clipped_w / width_thermal, 0.0, 1.0)),
        h=float(np.clip(clipped_h / height_thermal, 0.0, 1.0)),
    )
    return Warp(kept, retained, cut_edges, quad_fill, "")


def _frozen_lines(path: Path) -> list[str] | None:
    """The frozen label file's non-empty lines, or None when the stem is not in the tree."""
    if not path.is_file():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def audit_split(
    original_dir: Path,
    alignment_dir: Path,
    frozen_dir: Path,
    split: str,
    config: AuditConfig,
) -> tuple[list[BoxRow], list[str]]:
    """Every box of one split, plus a description of each fidelity mismatch found."""
    images_dir = original_dir / split / "images"
    labels_dir = original_dir / split / "labels"
    frozen_labels_dir = frozen_dir / split / FROZEN_LABELS
    for directory in (images_dir, frozen_labels_dir):
        if not directory.is_dir():
            raise AuditError(f"missing directory: {directory}")

    rows: list[BoxRow] = []
    mismatches: list[str] = []
    seen_stems: set[str] = set()
    skipped = 0
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = parse_stem(image_path.name, config)
        if stem in seen_stems:
            raise AuditError(f"{split}: two originals map to stem {stem!r}")
        seen_stems.add(stem)

        alignment_path = alignment_dir / f"{stem}.json"
        if not alignment_path.is_file():
            # The build skipped these too (build_rgbt_dataset.py:331-341).
            skipped += 1
            logger.warning("%s (%s): no alignment file, skipping as the build did", stem, split)
            continue
        alignment = load_alignment(alignment_path)
        frozen = _frozen_lines(frozen_labels_dir / f"{stem}.txt")

        kept_lines: list[str] = []
        for index, box in enumerate(read_labels(labels_dir / f"{image_path.stem}.txt")):
            warp = warp_box(box, alignment, config)
            frozen_line = None
            if warp.box is not None:
                kept_lines.append(warp.box.to_line())
                frozen_line = len(kept_lines) - 1 if frozen is not None else None
            rows.append(
                BoxRow(
                    split=split,
                    stem=stem,
                    box_index=index,
                    class_id=box.cls_id,
                    retained_fraction=warp.retained_fraction,
                    cut_edges="+".join(warp.cut_edges),
                    quad_fill=warp.quad_fill,
                    drop_reason=warp.drop_reason,
                    in_frozen_tree=frozen is not None,
                    frozen_line=frozen_line,
                )
            )
        if frozen is not None and frozen != kept_lines:
            mismatches.append(f"{split}/{stem}: frozen {frozen} but recomputed {kept_lines}")

    frozen_stems = {path.stem for path in frozen_labels_dir.glob("*.txt")}
    for stem in sorted(frozen_stems - seen_stems):
        mismatches.append(f"{split}/{stem}: in the frozen tree but has no original image")
    if skipped:
        logger.warning("%s: %d images skipped for a missing alignment file", split, skipped)
    return rows, mismatches


def audit(
    original_dir: Path,
    alignment_dir: Path,
    frozen_dir: Path,
    splits: Sequence[str],
    config: AuditConfig,
) -> list[BoxRow]:
    """Every box of every split, or :class:`AuditFidelityError` if any frozen label disagrees."""
    rows: list[BoxRow] = []
    mismatches: list[str] = []
    for split in splits:
        split_rows, split_mismatches = audit_split(
            original_dir, alignment_dir, frozen_dir, split, config
        )
        rows.extend(split_rows)
        mismatches.extend(split_mismatches)
    if mismatches:
        for mismatch in mismatches[:MISMATCHES_SHOWN]:
            logger.error("fidelity: %s", mismatch)
        raise AuditFidelityError(
            f"{len(mismatches)} label files disagree with the frozen tree at {frozen_dir}. "
            f"Check --original-dir, --alignment-dir and that --min-visibility "
            f"({config.min_visibility}) and --min-box-px ({config.min_box_px}) match the build"
        )
    return rows


def write_csv(rows: Sequence[BoxRow], path: Path) -> None:
    """One row per box. ``None`` is written as an empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(BoxRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in asdict(row).items()})


def log_summary(rows: Sequence[BoxRow]) -> None:
    """Per split and class: boxes in, clipped, dropped, and kept boxes under each threshold."""
    groups: dict[tuple[str, int], list[BoxRow]] = defaultdict(list)
    for row in rows:
        groups[(row.split, row.class_id)].append(row)
    for (split, class_id), group in sorted(groups.items()):
        kept = [row for row in group if not row.drop_reason]
        under = [
            sum(1 for row in kept if (row.retained_fraction or 0.0) < threshold)
            for threshold in SUMMARY_THRESHOLDS
        ]
        logger.info(
            "%s class %d: %d boxes | %d clipped | %d dropped | kept under %s: %s",
            split,
            class_id,
            len(group),
            sum(1 for row in group if row.cut_edges),
            len(group) - len(kept),
            SUMMARY_THRESHOLDS,
            under,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Paths are required: the server's data paths are never guessed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--original-dir", type=Path, required=True, help="yolo_original root")
    parser.add_argument("--alignment-dir", type=Path, required=True, help="alignment/*.json")
    parser.add_argument("--frozen-dir", type=Path, required=True, help="yolo_rgbt_29_jul root")
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    # The sister's defaults (build_rgbt_dataset.py parse_args).
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--min-box-px", type=float, default=2.0)
    parser.add_argument("--prefix-sep", default="-")
    parser.add_argument("--strip-suffix", default="_embedded")
    parser.add_argument("--stem-regex", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Audit, check fidelity, then write the CSV and log a per-class summary."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    config = AuditConfig(
        min_visibility=args.min_visibility,
        min_box_px=args.min_box_px,
        prefix_sep=args.prefix_sep,
        strip_suffix=args.strip_suffix,
        stem_regex=re.compile(args.stem_regex) if args.stem_regex else None,
    )
    rows = audit(args.original_dir, args.alignment_dir, args.frozen_dir, args.splits, config)
    write_csv(rows, args.out)
    log_summary(rows)
    logger.info("wrote %d box rows to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
