"""The injected misalignment levels of C2's registration-error axis (`t2o.data.misalignment`).

The load-bearing claims: each level moves every corner by exactly its stated pixels, level 0 is
the identity *exactly* (a level-0 label tree must be byte-identical), and each level carries its
own stable digest so a sweep row can be traced to the field that produced it.

`write_misaligned_tree`'s: only the thermal train boxes move, by the level's homography; val and
every image are the source's byte for byte; level 0 is the source tree; and no ultralytics label
cache travels into a tree whose labels it does not describe.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import IMG_SIZE, N_TRAIN, N_VAL
from t2o.data.calibration import _image_corners
from t2o.data.labels import load_yolo_labels
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import PROVENANCE_FILENAME, mirror_labels_onto_infrared
from t2o.data.misalignment import (
    REGISTRATION_LEVELS_PX,
    Misalignment,
    MisalignmentError,
    misalign_image,
    write_misaligned_tree,
)


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


HEIGHT, WIDTH = IMG_SIZE


@pytest.fixture
def aligned(dataset_root: Path, tmp_path: Path) -> DatasetManifest:
    """A copy of the shared fixture with thermal labels, standing in for an aligned dataset.

    Copied because `dataset_root` is session-scoped; `path:` is rewritten because it is absolute
    and would otherwise resolve back to the original tree (`test_mirror.py::mirrorable`). A stale
    `labels.cache` is planted to prove the writer leaves it behind.
    """
    root = tmp_path / "aligned"
    shutil.copytree(dataset_root, root)
    manifest_path = root / "data.yaml"
    manifest_path.write_text(
        manifest_path.read_text().replace(f"path: {dataset_root}", f"path: {root}")
    )
    manifest = DatasetManifest.load(manifest_path)
    mirror_labels_onto_infrared(manifest)
    (root / "train" / "infrared" / "labels.cache").write_bytes(b"stale")
    return manifest


def _files(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` by relative path, except the manifest and provenance."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"data.yaml", PROVENANCE_FILENAME}
    }


def _without_cache(files: dict[str, bytes]) -> dict[str, bytes]:
    return {name: data for name, data in files.items() if not name.endswith(".cache")}


def test_train_thermal_boxes_move_by_the_level_homography(
    aligned: DatasetManifest, tmp_path: Path
) -> None:
    misalignment = Misalignment(HEIGHT, WIDTH, 24.0)
    out = write_misaligned_tree(aligned, misalignment, tmp_path / "level-24").parent
    matrix = misalignment.homography()

    moved_any = False
    for source in sorted((aligned.root / "train" / "infrared" / "labels").glob("*.txt")):
        _, before = load_yolo_labels(source)
        _, after = load_yolo_labels(out / "train" / "infrared" / "labels" / source.name)
        assert len(after) == len(before)
        for (cx, cy, _, _), (mx, my, _, _) in zip(before.tolist(), after.tolist(), strict=True):
            # The roll field is affine, so a re-bounded box keeps its centre on the mapped centre.
            ((ex, ey),) = _map(np.array([[cx * WIDTH, cy * HEIGHT]]), matrix)
            assert mx * WIDTH == pytest.approx(ex, abs=1e-3)
            assert my * HEIGHT == pytest.approx(ey, abs=1e-3)
            moved_any |= bool(np.hypot(ex - cx * WIDTH, ey - cy * HEIGHT) > 1.0)
    assert moved_any, "no box moved; the fixture would not catch a skipped warp"


def test_everything_but_the_train_thermal_labels_is_the_source_byte_for_byte(
    aligned: DatasetManifest, tmp_path: Path
) -> None:
    out = write_misaligned_tree(aligned, Misalignment(HEIGHT, WIDTH, 12.0), tmp_path / "l12").parent

    source, copied = _without_cache(_files(aligned.root)), _files(out)
    assert copied.keys() == source.keys()
    changed = {name for name in source if source[name] != copied[name]}
    assert changed, "a 12 px level changed no file"
    assert all(name.startswith("train/infrared/labels/") for name in changed)
    assert sum(name.startswith("val/") for name in copied) == 4 * N_VAL  # 2 images, 2 labels each


def test_level_zero_is_the_source_tree_byte_for_byte(
    aligned: DatasetManifest, tmp_path: Path
) -> None:
    out = write_misaligned_tree(aligned, Misalignment(HEIGHT, WIDTH, 0.0), tmp_path / "l0").parent

    assert _files(out) == _without_cache(_files(aligned.root))


def test_no_label_cache_travels_and_the_manifest_points_at_its_own_tree(
    aligned: DatasetManifest, tmp_path: Path
) -> None:
    written = write_misaligned_tree(aligned, Misalignment(HEIGHT, WIDTH, 6.0), tmp_path / "l6")

    assert not list(written.parent.rglob("*.cache"))
    loaded = DatasetManifest.load(written)
    assert loaded.root == written.parent.resolve()
    assert loaded.train_images == written.parent.resolve() / "train" / "visible" / "images"
    assert loaded.names == aligned.names


def test_the_provenance_names_the_level_and_its_digest(
    aligned: DatasetManifest, tmp_path: Path
) -> None:
    misalignment = Misalignment(HEIGHT, WIDTH, 3.0)
    out = write_misaligned_tree(aligned, misalignment, tmp_path / "l3").parent

    provenance = json.loads((out / "train" / "infrared" / PROVENANCE_FILENAME).read_text())
    assert provenance["misalignment_level_px"] == 3.0
    assert provenance["misalignment_digest"] == misalignment.digest()
    assert provenance["files"] == N_TRAIN
    assert provenance["boxes_dropped"] == 0
    assert provenance["source_provenance"] is None  # a plain mirror carries no marker
    assert not (out / "val" / "infrared" / PROVENANCE_FILENAME).exists()


def test_a_misalignment_for_another_frame_size_is_refused(
    aligned: DatasetManifest, tmp_path: Path
) -> None:
    with pytest.raises(MisalignmentError, match="built for"):
        write_misaligned_tree(aligned, Misalignment(HEIGHT * 2, WIDTH * 2, 6.0), tmp_path / "x")


def test_a_source_without_thermal_labels_is_refused(data_yaml: Path, tmp_path: Path) -> None:
    with pytest.raises(MisalignmentError, match="no thermal labels"):
        write_misaligned_tree(
            DatasetManifest.load(data_yaml), Misalignment(HEIGHT, WIDTH, 6.0), tmp_path / "x"
        )


def test_a_non_empty_destination_is_refused(aligned: DatasetManifest, tmp_path: Path) -> None:
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "leftover.txt").write_text("")

    with pytest.raises(MisalignmentError, match="not empty"):
        write_misaligned_tree(aligned, Misalignment(HEIGHT, WIDTH, 6.0), tmp_path / "taken")


def test_the_image_moves_the_way_the_boxes_do() -> None:
    # A bright dot near the top-left corner must land where the homography that moves the boxes
    # sends it -- the image and its labels have to agree, or the loop trains on mismatched boxes.
    height, width = IMG_SIZE
    misalignment = Misalignment(height, width, 24.0)
    image = np.zeros((height, width, 1), dtype=np.float32)
    image[40:43, 50:53] = 1.0
    warped = misalign_image(image, misalignment)

    (target,) = _map(np.array([[51.0, 41.0]]), misalignment.homography())
    ys, xs = np.nonzero(warped[:, :, 0] > 0.5)
    landed = np.array([xs.mean(), ys.mean()])
    assert np.linalg.norm(landed - target) < 1.0
    # Near a corner the field moves a point by most of the level (here ~20 of 24 px).
    assert np.linalg.norm(landed - np.array([51.0, 41.0])) > 15.0


def test_the_image_warp_keeps_shape_and_dtype() -> None:
    height, width = IMG_SIZE
    image = np.random.default_rng(0).random((height, width, 3), dtype=np.float32)
    warped = misalign_image(image, Misalignment(height, width, 6.0))
    assert warped.shape == image.shape
    assert warped.dtype == np.float32


def test_the_image_warp_refuses_another_frame_size() -> None:
    height, width = IMG_SIZE
    with pytest.raises(MisalignmentError, match="built for"):
        misalign_image(
            np.zeros((height // 2, width, 3), np.float32), Misalignment(height, width, 6)
        )
