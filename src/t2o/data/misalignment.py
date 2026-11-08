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
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from t2o.data.calibration import (
    _HASH_LENGTH,
    FloatArray,
    _homography_from_corners,
    _image_corners,
)

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
