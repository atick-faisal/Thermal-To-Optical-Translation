"""A known, injected cross-modal misalignment: one level of C2's registration-error axis.

``data/calibration.py`` *removes* a measured misalignment; this module *adds* a synthetic one,
so label transfer and translation can each be trained at a controlled alignment error and their
degradation measured (``docs/goal.md`` C2, amended 2026-11-03). It is an input to the
experiment, never a registration method (Non-Goal "Not solving registration").

**Why a roll-shaped corner field.** FLIR's real residual is a corner field that reads as a
-1.18 degree roll (``calibration.py``'s module docstring), and plan Q1 picked that shape over a
uniform or per-image shift so the curve speaks to how real rigs fail. Every corner moves along
the tangent of a small rotation about the image centre. All four corners sit at the same radius,
so each moves by exactly ``level_px`` and :meth:`Misalignment.magnitude_px` -- the same
mean-corner-displacement reading ``ResidualCalibration.magnitude_px`` gives -- equals the level
by construction. The sense (clockwise on screen) is arbitrary but fixed: the axis measures
degradation against magnitude, and a fixed sense keeps every level a scaled copy of one field.

**Why not a ``ResidualCalibration``.** That type's required fields are matcher provenance
(``matchers``, ``spread_px``, ``git_sha``), and it refuses a record without matchers on
purpose. A synthetic level has none, so it reuses the corner machinery rather than faking it.

**Why a whole tree per level.** :func:`write_misaligned_tree` copies a dataset into a new
directory and misaligns only that copy's thermal train labels. Rewriting ``infrared/labels`` in
place would hit both traps ``data/mirror.py`` documents: ultralytics' ``labels.cache`` cannot see
a same-length rewrite (``labels.invalidate_label_cache``), and the source tree would no longer be
the zero point every level is measured against. The copy follows ``data/inner_split.py``, which
already writes a tree of its own for the same kind of reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from t2o.data.calibration import (
    _HASH_LENGTH,
    FloatArray,
    _homography_from_corners,
    _image_corners,
)
from t2o.data.dataset import IMAGE_SUFFIXES
from t2o.data.manifest import RGBT_BLOCK, DatasetManifest
from t2o.data.mirror import (
    PROVENANCE_FILENAME,
    BoxTransform,
    _split_image_shape,
    deroll_label_lines,
    read_label_provenance,
)
from t2o.data.pairing import LABEL_SUFFIX, LABELS_SEGMENT

logger = logging.getLogger(__name__)

DATA_FILENAME = "data.yaml"
# The `mirror.py` default: map all four corners and re-bound. Recorded in the provenance so a
# later `centre` tree cannot pass for this one.
_BOX_TRANSFORM: BoxTransform = "corners"

# Plan Q2: brackets FLIR's measured 5.90 px and reaches clear failure. Corner-field px at
# training resolution. Changing a value orphans every row and run stamped with the old digest.
REGISTRATION_LEVELS_PX = (0.0, 3.0, 6.0, 12.0, 24.0)


class MisalignmentError(ValueError):
    """Raised for a misalignment level that cannot describe a real offset."""


@dataclass(frozen=True, slots=True)
class Misalignment:
    """One injected misalignment level: a roll-shaped corner field of ``level_px`` per corner."""

    # The frame the level is measured on. A field built for 640x480 and applied at 1280x960 is
    # half the misalignment it claims, so callers pass the *training* resolution.
    height: int
    width: int
    # Displacement of every corner, in pixels. Negative is refused rather than read as the
    # opposite sense: the axis is a magnitude, and two signs would give one level two digests.
    level_px: float

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise MisalignmentError(f"shape must be positive, got {self.height}x{self.width}")
        if not math.isfinite(self.level_px) or self.level_px < 0:
            raise MisalignmentError(
                f"level must be a finite, non-negative pixel count, got {self.level_px}"
            )

    def corner_shift(self) -> FloatArray:
        """The four ``(dx, dy)`` displacements, clockwise from the top-left like the calibration."""
        corners = _image_corners(self.height, self.width)
        centre = np.array([self.width / 2, self.height / 2])
        offset = corners - centre
        # The rotation tangent (-y, x) in image coordinates (y down): clockwise on screen.
        tangent = np.stack([-offset[:, 1], offset[:, 0]], axis=1)
        unit = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
        return np.asarray(unit * self.level_px, dtype=np.float64)

    def homography(self) -> FloatArray:
        """Map a point on the aligned frame to where it lands once misaligned.

        Level 0 returns the identity exactly rather than a DLT solve that rounds to it: a
        level-0 label tree must be byte-identical to the aligned one, and ``1e-16`` of solver
        noise is enough to change a six-decimal label coordinate.
        """
        if self.level_px == 0:
            return np.eye(3, dtype=np.float64)
        reference = _image_corners(self.height, self.width)
        return _homography_from_corners(reference, reference + self.corner_shift())

    def magnitude_px(self) -> float:
        """Mean corner displacement, measured from the field rather than read from the level."""
        return float(np.linalg.norm(self.corner_shift(), axis=1).mean())

    def digest(self) -> str:
        """A fingerprint of the level, stamped on every sweep row and run it produces.

        The shape kind goes in alongside the numbers so a later, differently shaped field at
        the same magnitude cannot share a digest with this one.
        """
        payload = json.dumps(
            {
                "kind": "roll-corner-field",
                "height": self.height,
                "width": self.width,
                "level_px": round(self.level_px, 6),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:_HASH_LENGTH]


def _copy_split(manifest: DatasetManifest, images_dir: Path, out_split: Path) -> Path:
    """Copy one split's pairs, both cameras' images and labels, into ``out_split``.

    Returns the copy's thermal labels directory. Pair by pair rather than directory by
    directory: only images and ``.txt`` labels travel, so a ``labels.cache`` in the source is
    never carried into a tree whose labels it does not describe.
    """
    pairing = manifest.pairing
    split_root = Path(os.path.commonpath([images_dir, pairing.infrared_path(images_dir)]))
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for image in images:
        infrared = pairing.infrared_path(image)
        for source in (image, infrared, pairing.label_path(image), pairing.label_path(infrared)):
            # A pair with no boxes may have no label file at all; that is a negative, not a gap.
            if source.is_file():
                destination = out_split / source.relative_to(split_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

    source_labels = pairing.infrared_path(images_dir).parent / LABELS_SEGMENT
    copied_labels = out_split / source_labels.relative_to(split_root)
    # A de-rolled source (FLIR) says so beside its labels; the copy keeps saying it.
    if (provenance := source_labels.parent / PROVENANCE_FILENAME).is_file():
        shutil.copyfile(provenance, copied_labels.parent / PROVENANCE_FILENAME)
    return copied_labels


def write_misaligned_tree(
    manifest: DatasetManifest, misalignment: Misalignment, out_dir: Path | str
) -> Path:
    """Copy ``manifest``'s dataset into ``out_dir`` with its thermal train boxes misaligned.

    The thermal train labels are the dataset's *aligned* thermal boxes (the custom set's own,
    FLIR's mirrored and de-rolled ones), pushed through :meth:`Misalignment.homography` by
    ``mirror.deroll_label_lines``. Everything else -- both cameras' images, the visible labels,
    and the whole val split -- is copied byte for byte, so split membership and the scoring
    split are the source's (design.md §5, invariant 2). Level 0 copies the train labels too
    rather than round-tripping them through six-decimal arithmetic, so a level-0 tree is the
    source tree byte for byte. Returns the new ``data.yaml``.

    Refuses a non-empty ``out_dir`` (a second level written over a first would load cleanly as
    a mix of the two), a misalignment built for another frame size, and a source with no
    thermal labels to misalign.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise MisalignmentError(f"{out_dir} is not empty; write each level into a new directory")

    infrared_train = manifest.pairing.infrared_path(manifest.train_images)
    for images_dir in (infrared_train, manifest.pairing.infrared_path(manifest.val_images)):
        labels_dir = images_dir.parent / LABELS_SEGMENT
        if not labels_dir.is_dir() or not any(labels_dir.glob(f"*{LABEL_SUFFIX}")):
            raise MisalignmentError(
                f"{labels_dir} holds no thermal labels to misalign. A public dataset carries "
                "labels on its visible side only: mirror (and de-roll) them first."
            )
    shape = _split_image_shape(infrared_train)
    if shape != (misalignment.height, misalignment.width):
        raise MisalignmentError(
            f"misalignment is built for {misalignment.height}x{misalignment.width} but "
            f"{infrared_train} is {shape[0]}x{shape[1]}; the level would not be the pixels it "
            "claims"
        )

    train_labels = _copy_split(manifest, manifest.train_images, out_dir / "train")
    _copy_split(manifest, manifest.val_images, out_dir / "val")

    source_labels = infrared_train.parent / LABELS_SEGMENT
    files = sorted(train_labels.glob(f"*{LABEL_SUFFIX}"))
    dropped = 0
    if misalignment.level_px > 0:
        matrix = misalignment.homography()
        for label in files:
            lines, lost = deroll_label_lines(label, matrix, shape, _BOX_TRANSFORM)
            dropped += lost
            label.write_text("\n".join(lines) + "\n" if lines else "")
    (train_labels.parent / PROVENANCE_FILENAME).write_text(
        json.dumps(
            {
                "labels": "aligned thermal labels with an injected misalignment",
                "source": str(manifest.path),
                "misalignment_level_px": misalignment.level_px,
                "misalignment_digest": misalignment.digest(),
                "box_transform": _BOX_TRANSFORM,
                "files": len(files),
                "boxes_dropped": dropped,
                # What the source labels already were, e.g. FLIR's de-roll. None on a tree whose
                # thermal labels were drawn in their own right (the custom set).
                "source_provenance": read_label_provenance(source_labels),
            },
            indent=2,
        )
        + "\n"
    )

    split_root = Path(os.path.commonpath([manifest.train_images, infrared_train]))
    visible_images = manifest.train_images.relative_to(split_root).as_posix()
    destination = out_dir / DATA_FILENAME
    destination.write_text(
        yaml.safe_dump(
            {
                # Absolute: ultralytics honours `path:` literally (CLAUDE.md).
                "path": str(out_dir.resolve()),
                "train": f"train/{visible_images}",
                "val": f"val/{visible_images}",
                "nc": manifest.nc,
                "names": manifest.class_names,
                RGBT_BLOCK: {
                    "visible_token": manifest.pairing.visible_token,
                    "infrared_token": manifest.pairing.infrared_token,
                },
            },
            sort_keys=False,
        )
    )
    level = logger.warning if dropped else logger.info
    level(
        "misaligned tree %s: %d train label file(s) at %.2f px (digest %s), %d box(es) dropped",
        destination,
        len(files),
        misalignment.level_px,
        misalignment.digest(),
        dropped,
    )
    return destination
