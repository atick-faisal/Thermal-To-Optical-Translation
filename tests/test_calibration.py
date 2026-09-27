"""The published residual-misalignment constant (`t2o.data.calibration`).

Two things here are load-bearing rather than routine. The **digest** pins a byte-for-byte
transcription of a constant that lives in a sibling repo, so a typo in a corner cannot pass as a
calibration. The **direction** pins `visible_to_infrared` to `inv(homography())`: the source
repo's reference/moving wording reads the other way round, and applying the non-inverted matrix
would have *doubled* FLIR's misalignment while looking exactly like a correction. The evidence for
the sign is an alpha sweep over real pairs (module docstring); this file is what stops a refactor
quietly reverting it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from t2o.data.calibration import (
    CalibrationError,
    ResidualCalibration,
    load_calibration,
)

FLIR_CALIBRATION = Path(__file__).resolve().parents[1] / "calibration" / "flir.json"
# The fingerprint of the sibling repo's `calibration/flir.json` at git_sha 739796f, over the
# dataset, the shape and the four corners only. Hardcoded on purpose: this is the assertion, not
# a value derived from the file under test.
FLIR_DIGEST = "210142740cdba163"


def make_calibration(**overrides: object) -> ResidualCalibration:
    fields: dict[str, object] = {
        "dataset": "synthetic",
        "height": 480,
        "width": 640,
        "corner_shift": ((0.0, 0.0),) * 4,
        "matchers": ("roma",),
        "spread_px": 1.0,
        "worst_case_px": 2.0,
        "n_pairs": 10,
        "split": "val",
        "git_sha": "abc1234",
        "note": "synthetic",
    }
    return ResidualCalibration(**(fields | overrides))  # type: ignore[arg-type]


def translation(dx: float, dy: float) -> ResidualCalibration:
    """A calibration whose four corners move identically, i.e. a pure translation."""
    return make_calibration(corner_shift=((dx, dy),) * 4)


def test_the_real_flir_constant_is_transcribed_intact() -> None:
    calibration = load_calibration(FLIR_CALIBRATION)

    assert calibration.dataset == "flir"
    assert calibration.shape == (512, 640)
    assert calibration.split == "val"
    assert calibration.n_pairs == 1013
    assert len(calibration.matchers) == 3
    assert calibration.digest() == FLIR_DIGEST
    assert calibration.magnitude_px() == pytest.approx(9.3158481, abs=1e-6)


def test_the_stored_digest_is_not_read_back_as_the_answer(tmp_path: Path) -> None:
    """A hand-edited corner must not hide behind the digest the file happens to carry."""
    data = json.loads(FLIR_CALIBRATION.read_text())
    data["corner_shift"][0][0] += 5.0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(data))

    assert load_calibration(tampered).digest() != FLIR_DIGEST


def test_a_zero_shift_is_the_identity() -> None:
    assert make_calibration().homography() == pytest.approx(np.eye(3))


def test_a_uniform_corner_shift_is_exactly_that_translation() -> None:
    matrix = translation(4.0, -3.0).homography()

    expected = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    assert matrix == pytest.approx(expected)


def test_the_solve_agrees_with_opencv() -> None:
    """opencv is not a declared dependency, hence the twelve-line solve -- but when it is
    importable it is the reference implementation the sibling repo calls, so use it."""
    cv2 = pytest.importorskip("cv2", reason="opencv not installed (a transitive dep only)")
    calibration = load_calibration(FLIR_CALIBRATION)
    height, width = calibration.shape
    reference = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    shifted = reference + np.asarray(calibration.corner_shift)

    expected = cv2.getPerspectiveTransform(reference.astype(np.float32), shifted.astype(np.float32))
    assert calibration.homography() == pytest.approx(expected, abs=1e-6)


def test_visible_to_infrared_inverts_the_published_matrix() -> None:
    calibration = load_calibration(FLIR_CALIBRATION)

    product = calibration.homography() @ calibration.visible_to_infrared()
    assert product == pytest.approx(np.eye(3), abs=1e-9)


def test_the_direction_moves_flirs_left_edge_left() -> None:
    """The sign assertion. FLIR's published field carries the bottom-left corner *right* by
    +13.489 px; the visible -> thermal map is the inverse, so a box down there must move
    *left*. Getting this backwards doubles the misalignment instead of removing it, and the
    result looks like a plausible correction either way -- hence a test rather than a comment.
    """
    calibration = load_calibration(FLIR_CALIBRATION)
    height, _ = calibration.shape

    point = np.array([0.0, float(height), 1.0])
    moved = calibration.visible_to_infrared() @ point
    dx = moved[0] / moved[2] - point[0]

    assert dx == pytest.approx(-13.757, abs=0.01)


def test_validate_for_rejects_another_dataset() -> None:
    with pytest.raises(CalibrationError, match="is for 'flir' but this run is on 'llvip'"):
        load_calibration(FLIR_CALIBRATION).validate_for("llvip", (512, 640))


def test_validate_for_rejects_another_shape() -> None:
    """A 640x512 corner field at 1280x1024 is silently half the offset it claims to be."""
    with pytest.raises(CalibrationError, match="does not rescale"):
        load_calibration(FLIR_CALIBRATION).validate_for("flir", (1024, 1280))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"corner_shift": ((0.0, 0.0), (1.0, 1.0))}, "4 corners"),
        ({"matchers": ()}, "got none"),
        ({"height": 0}, "shape must be positive"),
    ],
)
def test_a_malformed_constant_is_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(CalibrationError, match=message):
        make_calibration(**overrides)


def test_a_missing_field_names_itself(tmp_path: Path) -> None:
    incomplete = tmp_path / "partial.json"
    incomplete.write_text(json.dumps({"dataset": "flir", "height": 512, "width": 640}))

    with pytest.raises(CalibrationError, match="malformed calibration record"):
        load_calibration(incomplete)


def test_a_missing_file_raises_a_calibration_error(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="cannot read calibration"):
        load_calibration(tmp_path / "nope.json")


def test_invalid_json_raises_a_calibration_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")

    with pytest.raises(CalibrationError, match="is not valid JSON"):
        load_calibration(broken)
