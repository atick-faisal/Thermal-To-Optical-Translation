"""The injected misalignment levels of C2's registration-error axis (`t2o.data.misalignment`).

The load-bearing claims: each level moves every corner by exactly its stated pixels, level 0 is
the identity *exactly* (a level-0 label tree must be byte-identical), and each level carries its
own stable digest so a sweep row can be traced to the field that produced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from t2o.data.calibration import _image_corners
from t2o.data.misalignment import REGISTRATION_LEVELS_PX, Misalignment, MisalignmentError


def _map(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


@pytest.mark.parametrize("level", REGISTRATION_LEVELS_PX)
def test_the_measured_magnitude_is_the_level(level: float) -> None:
    misalignment = Misalignment(height=480, width=640, level_px=level)

    assert misalignment.magnitude_px() == pytest.approx(level, abs=1e-9)
    per_corner = np.linalg.norm(misalignment.corner_shift(), axis=1)
    np.testing.assert_allclose(per_corner, level, atol=1e-9)


@pytest.mark.parametrize("level", [3.0, 24.0])
def test_the_homography_carries_each_corner_onto_its_shifted_position(level: float) -> None:
    misalignment = Misalignment(height=480, width=640, level_px=level)
    corners = _image_corners(480, 640)

    moved = _map(corners, misalignment.homography())

    np.testing.assert_allclose(moved - corners, misalignment.corner_shift(), atol=1e-6)


def test_the_field_is_a_clockwise_roll_about_the_centre() -> None:
    shift = Misalignment(height=480, width=640, level_px=6.0).corner_shift()
    top_left, top_right, bottom_right, bottom_left = shift

    # Clockwise on screen (y down): the top edge moves right, the bottom edge moves left.
    assert top_left[0] > 0 and top_right[0] > 0
    assert bottom_right[0] < 0 and bottom_left[0] < 0
    # A rotation about the centre: opposite corners move in opposite directions.
    np.testing.assert_allclose(top_left, -bottom_right, atol=1e-12)
    np.testing.assert_allclose(top_right, -bottom_left, atol=1e-12)


def test_level_zero_is_exactly_the_identity() -> None:
    misalignment = Misalignment(height=480, width=640, level_px=0.0)

    assert np.array_equal(misalignment.homography(), np.eye(3))
    assert not misalignment.corner_shift().any()


def test_the_digest_is_stable_and_differs_per_level_and_shape() -> None:
    digests = {Misalignment(480, 640, level).digest() for level in REGISTRATION_LEVELS_PX}

    assert len(digests) == len(REGISTRATION_LEVELS_PX)
    assert Misalignment(480, 640, 6.0).digest() == Misalignment(480, 640, 6.0).digest()
    assert Misalignment(480, 640, 6.0).digest() != Misalignment(512, 640, 6.0).digest()
    assert all(len(digest) == 16 for digest in digests)


@pytest.mark.parametrize(
    ("height", "width", "level"),
    [(480, 640, -3.0), (480, 640, float("nan")), (480, 640, float("inf")), (0, 640, 3.0)],
)
def test_an_impossible_level_or_shape_is_refused(height: int, width: int, level: float) -> None:
    with pytest.raises(MisalignmentError):
        Misalignment(height=height, width=width, level_px=level)
