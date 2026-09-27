"""Mirror a split's visible-side labels onto its infrared side, optionally de-rolled.

Every adapted public dataset writes labels under ``visible/labels`` only
(``data/adapters/common.py``'s ``write_label`` / ``write_label_lines`` both hardcode that
path), because ``data.pairing.Pairing.label_path`` resolves a label from the *visible* path
and translator training never needs the thermal side. The custom pairs do carry both sides.

That difference stays invisible until something points a detector at ``infrared/images``:
ultralytics derives its label path by swapping the ``images`` segment for ``labels``, finds
nothing, and reads the entire split as unlabelled -- it trains, it reports, and the number
means nothing. ``data.budget._require_labels_beside`` raises instead of letting that happen,
and this module is the operation whose absence it is reporting.

**The mirrored boxes are drawn on the visible frame, and the two frames are not perfectly
registered.** On FLIR-aligned the residual is ~5.9 px, structured as a fixed camera roll
(TASKS.md's E9 section; measured in the sibling ``Thermal-Image-Registration`` repo across
three matchers against a ~0.2 px pipeline floor). A plain mirrored box therefore sits a few
pixels off its thermal object. At mAP50's IoU 0.5 threshold that is a small penalty on a
typical car or person box, and it pushes the measured raw-thermal score *down* -- which
**flatters** translation, because that score is the floor any translation gain is measured
against.

``calibration`` is the correction. Passing a :class:`t2o.data.calibration.ResidualCalibration`
transforms each box onto the thermal frame instead of copying it, which turns that caveat into
a measurement: E9 re-runs the gate table against de-rolled labels and reads the headroom off
the less-flattering of the two floors. Without it the copy path is byte-for-byte as before, so
the uncorrected state is always one forced re-run away.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from t2o.data.calibration import FloatArray, ResidualCalibration
from t2o.data.labels import load_yolo_labels
from t2o.data.manifest import DatasetManifest
from t2o.data.pairing import LABEL_SUFFIX, LABELS_SEGMENT

logger = logging.getLogger(__name__)

PROVENANCE_FILENAME = "LABELS_PROVENANCE.json"

# How a box crosses modalities. `corners` maps all four and re-bounds, which is the standard
# and geometrically correct choice but changes width/height by ~1 px under FLIR's roll;
# `centre` maps only the centre and keeps the extent exactly. The two agreeing is the cheapest
# available evidence that the choice does not carry the result, which is why both exist.
BoxTransform = Literal["corners", "centre"]


class MirrorError(Exception):
    """Raised when labels cannot be mirrored onto a dataset's infrared side."""


def _split_image_shape(images_dir: Path) -> tuple[int, int]:
    """``(height, width)`` of the first image in ``images_dir``.

    One image rather than all of them: an adapted tree is written by a single adapter pass, so
    its frames are uniform by construction (FLIR's 4,129 + 1,013 pairs were audited and every
    one is 640x512). The point of reading any image at all is that
    :meth:`ResidualCalibration.validate_for` must compare against something measured rather
    than against a number a caller asserted.
    """
    images = sorted(p for p in images_dir.iterdir() if p.is_file())
    if not images:
        raise MirrorError(f"{images_dir} holds no images; is this an adapted dataset?")
    with Image.open(images[0]) as image:
        width, height = image.size
    return (height, width)


def _map_points(points: FloatArray, matrix: FloatArray) -> FloatArray:
    """Push ``(N, 2)`` pixel coordinates through a 3x3 homography."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ matrix.T
    return np.asarray(projected[:, :2] / projected[:, 2:3], dtype=np.float64)


def deroll_label_lines(
    label_path: Path, matrix: FloatArray, shape: tuple[int, int], mode: BoxTransform
) -> tuple[list[str], int]:
    """Transform one visible-frame label file's boxes onto the thermal frame.

    Returns the rewritten lines and a dropped-box count, matching
    ``adapters/common.py::voc_to_yolo_lines`` -- including its clamp-then-drop rule, because a
    box clamped to zero area is a degenerate detection target rather than a negative. A ~6 px
    shift cannot empty a real box, so a nonzero count here is a signal, not routine.

    Reads through ``data.labels.load_yolo_labels`` rather than splitting the text again: it
    already validates the five-field format and treats a missing or empty file as a legitimate
    zero-instance negative.
    """
    height, width = shape
    classes, boxes = load_yolo_labels(label_path)
    lines: list[str] = []
    dropped = 0
    for klass, (cx, cy, bw, bh) in zip(classes[:, 0].tolist(), boxes.tolist(), strict=True):
        cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
        if mode == "centre":
            ((moved_x, moved_y),) = _map_points(np.array([[cx, cy]]), matrix)
            x0, y0, x1, y1 = moved_x - bw / 2, moved_y - bh / 2, moved_x + bw / 2, moved_y + bh / 2
        else:
            corners = np.array(
                [
                    [cx - bw / 2, cy - bh / 2],
                    [cx + bw / 2, cy - bh / 2],
                    [cx + bw / 2, cy + bh / 2],
                    [cx - bw / 2, cy + bh / 2],
                ]
            )
            moved = _map_points(corners, matrix)
            x0, y0 = moved.min(axis=0)
            x1, y1 = moved.max(axis=0)

        x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
        y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
        new_w, new_h = x1 - x0, y1 - y0
        if new_w <= 0 or new_h <= 0:
            dropped += 1
            continue
        lines.append(
            f"{int(klass)} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{new_w / width:.6f} {new_h / height:.6f}"
        )
    return lines, dropped


def mirror_labels_onto_infrared(
    manifest: DatasetManifest,
    *,
    force: bool = False,
    calibration: ResidualCalibration | None = None,
    box_transform: BoxTransform = "corners",
) -> dict[str, int]:
    """Copy each split's ``visible/labels/*.txt`` into that split's ``infrared/labels/``.

    Returns ``{split: files_written}``. Goes through ``manifest.pairing`` rather than string
    surgery for the same reason :func:`t2o.data.budget.write_budget_manifest` does: a dataset
    that names its modalities differently still works, and a layout with no visible segment
    fails loudly.

    With no ``calibration`` the copy is byte-for-byte, so a re-run rewrites identical files.
    With one, each box is transformed onto the thermal frame by
    :meth:`ResidualCalibration.visible_to_infrared` and a ``LABELS_PROVENANCE.json`` is written
    beside ``labels/`` recording which constant produced them -- the tree is then able to say
    which of its two possible states it is in, which matters because both states are the same
    filenames in the same directory. A plain re-run removes that file along with the
    correction, so the marker can never outlive what it describes.

    Refuses a destination that already holds label files unless ``force``. **The custom dataset
    is exactly that case**, and its thermal labels may be drawn on the thermal frame in their
    own right -- overwriting those with visible-side boxes would quietly replace better
    annotation with worse, on the dataset every headline result rests on.
    """
    written: dict[str, int] = {}
    matrix = calibration.visible_to_infrared() if calibration else None
    if calibration:
        logger.info(
            "de-rolling with %s (digest %s, %s, %.2f px mean corner shift, %s transform)",
            calibration.dataset,
            calibration.digest(),
            "/".join(calibration.matchers),
            calibration.magnitude_px(),
            box_transform,
        )

    for split, images_dir in (("train", manifest.train_images), ("val", manifest.val_images)):
        source = images_dir.parent / LABELS_SEGMENT
        labels = sorted(source.glob(f"*{LABEL_SUFFIX}")) if source.is_dir() else []
        if not labels:
            raise MirrorError(
                f"{source} holds no {LABEL_SUFFIX} files -- there is nothing to mirror. "
                f"Is {manifest.path} pointing at an adapted dataset?"
            )

        destination = manifest.pairing.infrared_path(images_dir).parent / LABELS_SEGMENT
        existing = sorted(destination.glob(f"*{LABEL_SUFFIX}")) if destination.is_dir() else []
        if existing and not force:
            raise MirrorError(
                f"{destination} already holds {len(existing)} label file(s). Re-run with "
                "force=True only if those are known to be copies -- if they are thermal-side "
                "annotations of their own, this would replace better boxes with worse."
            )

        destination.mkdir(parents=True, exist_ok=True)
        provenance = destination.parent / PROVENANCE_FILENAME
        if calibration is None or matrix is None:
            for label in labels:
                shutil.copyfile(label, destination / label.name)
            provenance.unlink(missing_ok=True)
            logger.info("%s: mirrored %d label file(s) -> %s", split, len(labels), destination)
        else:
            shape = _split_image_shape(images_dir)
            calibration.validate_for(manifest.root.name, shape)
            dropped = 0
            for label in labels:
                lines, lost = deroll_label_lines(label, matrix, shape, box_transform)
                dropped += lost
                (destination / label.name).write_text("\n".join(lines) + "\n" if lines else "")
            provenance.write_text(
                json.dumps(
                    {
                        "labels": "de-rolled onto the thermal frame",
                        "source": f"{manifest.root.name}/{LABELS_SEGMENT} (visible side)",
                        "calibration_digest": calibration.digest(),
                        "calibration_git_sha": calibration.git_sha,
                        "direction": "visible_to_infrared = inv(homography)",
                        "box_transform": box_transform,
                        "files": len(labels),
                        "boxes_dropped": dropped,
                    },
                    indent=2,
                )
                + "\n"
            )
            level = logger.warning if dropped else logger.info
            level(
                "%s: de-rolled %d label file(s), %d box(es) dropped -> %s",
                split,
                len(labels),
                dropped,
                destination,
            )
        written[split] = len(labels)

    return written
