"""FLIR's residual cross-modal misalignment, read from a frozen published constant.

FLIR-aligned's two cameras are not actually aligned. Three matchers spanning three decades of
compute cost agree on a ~5.9 px residual against a ~0.2 px pipeline floor, which makes it a
property of *the data* rather than of any matcher (TASKS.md M3 E9, blocker 1; measured in the
sibling ``Thermal-Image-Registration`` repo). ``data/mirror.py`` copies visible-side boxes onto
the thermal side, so every mirrored box inherits that residual -- and at mAP50's IoU 0.5 that
pushes the measured raw-thermal floor *down*, which **flatters** translation. This module is the
constant that removes it.

**Why a corner field and not "-1.18 degrees of roll".** The *total* displacement is well
determined -- the three matchers' corner fields agree to 1.23 px mean, 2.33 px worst case -- but
its decomposition into rotation and scale is not: over a limited field of view a small roll and a
small scale anisotropy generate nearly the same corner displacement, and the matchers disagree
about the split by 38%. So the publishable object is four corner displacements and the degrees are
a reading of them. This file is a byte-for-byte transcription of the sibling's
``calibration/flir.json`` (``git_sha 739796f``); :meth:`ResidualCalibration.digest` is what proves
the transcription, and ``calibration/rejected/`` over there is why "accepted" is a real
distinction worth re-checking.

**The direction was measured, not reasoned, and it is the opposite of what the source's wording
suggests.** The sibling documents ``corner_shift`` as carrying a *reference* corner to where the
*moving* modality lands, with ``moving`` defaulting to thermal -- which reads as visible -> thermal
= ``H``. Warping FLIR val's visible frame by ``H(alpha)`` (the corner field scaled by ``alpha``)
and scoring it against the raw thermal frame over 60 pairs says otherwise:

=========  ====================  =============
``alpha``  mutual information    gradient NCC
=========  ====================  =============
-1.50      0.6680                0.2911
**-1.00**  **0.6833**            **0.3346**
-0.50      0.6644                0.2992
0.00 (raw) 0.6335                0.2558
+1.00      0.6070                0.2196
=========  ====================  =============

Both criteria peak at ``alpha = -1``, smoothly and unimodally, and ``alpha = -1`` improved mutual
information on **59 of 60** pairs where ``alpha = +1`` improved it on 8. So the visible -> thermal
point map is :meth:`visible_to_infrared`, i.e. ``inv(H)``. Applying ``H`` instead would have
*doubled* the misalignment while looking exactly like a correction, so the sign is pinned by a test
(``tests/test_calibration.py``) rather than left to this docstring.

The file layout follows ``data/splits.py``: a small tracked JSON under a top-level directory, a
frozen dataclass with ``from_dict``, and a layer-specific error that names what drifted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_HASH_LENGTH = 16  # matches data/splits.py's digest width
_CORNERS = 4
# Clockwise from the top-left, the order `corner_shift` is stored in. Named so a log line and a
# JSON file cannot disagree about which pair of numbers is which corner.
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


class CalibrationError(ValueError):
    """Raised for a malformed calibration, or one applied to data it does not describe."""


def _image_corners(height: int, width: int) -> FloatArray:
    """The four image corners in ``(x, y)``, clockwise from the top-left.

    ``(width, height)`` and not ``(width - 1, height - 1)``: the corners are the outer edges of
    the sampling grid, which is the convention the corner field was estimated under. Using the
    last pixel index instead would silently rescale every displacement.
    """
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)


def _homography_from_corners(source: FloatArray, destination: FloatArray) -> FloatArray:
    """The homography carrying four ``source`` points onto four ``destination`` points.

    A plain 8x8 DLT solve with ``h33`` fixed to 1. ``cv2.getPerspectiveTransform`` does exactly
    this and is what the sibling repo calls, but **opencv is not a declared dependency here** --
    it only arrives transitively through ultralytics, and adding a direct dependency to save
    twelve lines is not a trade worth making. ``tests/test_calibration.py`` asserts this against
    ``cv2`` when cv2 happens to import, and against analytic cases when it does not.
    """
    rows = []
    targets = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        targets.extend((u, v))
    try:
        solution = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(targets))
    except np.linalg.LinAlgError as error:  # four collinear corners; not reachable from a file
        raise CalibrationError(f"corner field is degenerate: {error}") from error
    return np.append(solution, 1.0).reshape(3, 3)


@dataclass(frozen=True, slots=True)
class ResidualCalibration:
    """The one homography a dataset's two cameras are offset by, as four corner shifts."""

    dataset: str
    height: int
    width: int
    # Four (dx, dy) displacements in pixels, clockwise from the top-left. This *is* the constant;
    # every other field is provenance. See the module docstring for which way it points.
    corner_shift: tuple[tuple[float, float], ...]
    # The matchers the element-wise median was taken over. One leg is not a calibration, it is
    # that matcher's bias -- recorded so a reader can see how many witnesses the constant has.
    matchers: tuple[str, ...]
    # Mean and worst-case distance of a leg from the published median. This is the constant's
    # stated uncertainty, and the number that decides whether applying it is a net win: 1.23 px
    # of matcher-choice bias against the ~4.1 px of per-pair scatter no calibration can remove.
    spread_px: float
    worst_case_px: float
    n_pairs: int
    split: str
    git_sha: str
    note: str

    def __post_init__(self) -> None:
        if len(self.corner_shift) != _CORNERS:
            raise CalibrationError(
                f"{self.dataset}: a corner field has {_CORNERS} corners, got "
                f"{len(self.corner_shift)}"
            )
        if not self.matchers:
            raise CalibrationError(
                f"{self.dataset}: a calibration records the matchers it was estimated from, and "
                "one leg is a matcher's bias rather than a constant; got none"
            )
        if self.height <= 0 or self.width <= 0:
            raise CalibrationError(
                f"{self.dataset}: shape must be positive, got {self.height}x{self.width}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)``, the convention every array in this repo uses."""
        return (self.height, self.width)

    def homography(self) -> FloatArray:
        """The published constant as a matrix: refit from the four displaced corners."""
        reference = _image_corners(self.height, self.width)
        shifted = reference + np.asarray(self.corner_shift, dtype=np.float64)
        return _homography_from_corners(reference, shifted)

    def visible_to_infrared(self) -> FloatArray:
        """Map a point on the visible frame to where its content lands on the thermal frame.

        ``inv(homography())``, and the inverse is the whole point -- see the module docstring's
        alpha sweep. This is the only matrix callers should use: it is named for the direction it
        travels rather than for the source repo's reference/moving convention, because that
        convention is what made the sign ambiguous in the first place.
        """
        return np.asarray(np.linalg.inv(self.homography()), dtype=np.float64)

    def magnitude_px(self) -> float:
        """Mean corner displacement -- how much misalignment applying this removes."""
        return float(np.linalg.norm(np.asarray(self.corner_shift, dtype=np.float64), axis=1).mean())

    def digest(self) -> str:
        """A fingerprint of the *constant*, not of the file that carries it.

        Only the dataset, the shape and the four displacements go in: two files agreeing on those
        describe the same calibration whatever their notes say. A gate row stamped with this is
        traceable to the exact matrix that produced it, which hashing the configured *path*
        cannot do -- two different constants can share a filename across two machines.
        """
        payload = json.dumps(
            {
                "dataset": self.dataset,
                "height": self.height,
                "width": self.width,
                "corner_shift": [[round(dx, 6), round(dy, 6)] for dx, dy in self.corner_shift],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:_HASH_LENGTH]

    def validate_for(self, dataset: str, shape: tuple[int, int]) -> None:
        """Raise unless this constant describes ``dataset`` at ``shape``.

        Both checks are fatal rather than accommodating. A corner field measured at 640x512 and
        applied to a 1280x1024 pair is silently *half* the misalignment it claims to be, and one
        applied to the wrong dataset removes an offset that was never there while adding one that
        is -- neither fails loudly on its own, and both would surface months later as an
        inexplicable table.
        """
        if dataset != self.dataset:
            raise CalibrationError(
                f"calibration is for '{self.dataset}' but this run is on '{dataset}'"
            )
        if shape != self.shape:
            raise CalibrationError(
                f"{self.dataset}: calibration was measured at {self.height}x{self.width} but the "
                f"images are {shape[0]}x{shape[1]}; a corner field does not rescale"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualCalibration:
        """Build from a parsed JSON record.

        ``digest`` and ``magnitude_px`` in the file are ignored: they are written for the reader
        and for ``git diff``, and reading them back would let a hand-edited corner hide behind a
        stale fingerprint.
        """
        try:
            return cls(
                dataset=data["dataset"],
                height=int(data["height"]),
                width=int(data["width"]),
                corner_shift=tuple((float(dx), float(dy)) for dx, dy in data["corner_shift"]),
                matchers=tuple(data["matchers"]),
                spread_px=float(data["spread_px"]),
                worst_case_px=float(data["worst_case_px"]),
                n_pairs=int(data["n_pairs"]),
                split=data["split"],
                git_sha=data["git_sha"],
                note=data["note"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationError(f"malformed calibration record: {error}") from error


def load_calibration(path: Path | str) -> ResidualCalibration:
    """Read a calibration JSON. Raises :class:`CalibrationError` rather than ``OSError``."""
    location = Path(path)
    try:
        data = json.loads(location.read_text())
    except OSError as error:
        raise CalibrationError(f"cannot read calibration {location}: {error}") from error
    except json.JSONDecodeError as error:
        raise CalibrationError(f"{location} is not valid JSON: {error}") from error
    return ResidualCalibration.from_dict(data)
