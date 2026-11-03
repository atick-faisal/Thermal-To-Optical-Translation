"""Counterfactual erasure: inpaint ground-truth boxes out of a split's thermal images.

The cascade gate's probe (``docs/features/cascade-gate/plan.md``, Q4) asks whether a pipeline
puts back an object that is no longer in its thermal input. This module makes that input: each
labelled box, grown by a few pixels, is filled with OpenCV's Telea inpainting
(``cv2.INPAINT_TELEA``), which propagates the surrounding background inward. Telea was chosen
over a flat ring-median fill, which leaves a uniform patch a diffusion model may "repair" into
something, and over a heavy blur, which keeps the object's heat signature and so tests
robustness rather than erasure.

**The boxes are grown before inpainting** because the custom set's thermal boxes were moved
from the visible frame by registration (``docs/goal.md``, Constraints), so an object can sit a
few pixels outside its own box. A rim of warm pixels left behind would be a trace for stage 1 to
complete from, and the completion rate would then measure the label error, not the translator.

**Everything is written as PNG**, whatever the source suffix. The custom set's frames are
JPEG, and re-encoding an erased JPEG would change pixels *outside* the boxes too. PNG stores
the decoded pixels exactly, so the only difference from the source frame the translator
loads (through PIL, ``data/dataset.py::load_image_tensor``) is the erased region. Both
modalities are re-saved because ``Pairing`` derives the thermal path from the visible one by
segment substitution and keeps the filename, suffix included.

**Only the named classes are erased when ``classes`` is given** (cascade-gate ADR-03: fuses and
switches). Every other object stays in the frame untouched, and its label line is dropped from
the copy. The erased tree's labels are then exactly the erased boxes, so the completion rate
scores only those: a visible pole the judge finds would otherwise count as "put back". A kept
line is copied as text, never re-formatted. With no ``classes`` every box is erased and the
labels are copied byte for byte.

The boxes stay where the objects *were*, which is exactly what the completion rate scores
against. ultralytics' ``labels.cache`` is never copied
(``data/labels.py::invalidate_label_cache`` says why a stale one is silently trusted).
"""

from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from t2o.data.dataset import IMAGE_SUFFIXES
from t2o.data.labels import load_yolo_labels
from t2o.data.manifest import DatasetManifest
from t2o.data.pairing import LABEL_SUFFIX, LABELS_SEGMENT

logger = logging.getLogger(__name__)

DATA_FILENAME = "data.yaml"
ERASED_SUFFIX = ".png"

# Pixels added on every side of a box before it is inpainted. Too small leaves a warm rim
# outside a registration-shifted box for the translator to complete from; too large inpaints
# neighbouring objects and real background along with the target.
DEFAULT_DILATE_PX = 4
# Telea's neighbourhood radius in pixels (OpenCV's documented example value). Larger blends
# from farther away and smears more; it never touches pixels outside the mask.
DEFAULT_INPAINT_RADIUS = 3.0


class ErasureError(Exception):
    """Raised when an erased split cannot be written."""


def box_mask(shape: tuple[int, int], bboxes: NDArray[np.floating], dilate_px: int) -> NDArray:
    """A uint8 mask (255 inside) of every normalised ``cxcywh`` box, grown and clipped.

    Box edges are rounded outward, so a box never shrinks by a fraction of a pixel.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for cx, cy, w, h in bboxes.tolist():
        x0 = max(0, math.floor((cx - w / 2) * width) - dilate_px)
        y0 = max(0, math.floor((cy - h / 2) * height) - dilate_px)
        x1 = min(width, math.ceil((cx + w / 2) * width) + dilate_px)
        y1 = min(height, math.ceil((cy + h / 2) * height) + dilate_px)
        mask[y0:y1, x0:x1] = 255
    return mask


def erase_boxes(
    image: NDArray,
    bboxes: NDArray[np.floating],
    *,
    dilate_px: int = DEFAULT_DILATE_PX,
    inpaint_radius: float = DEFAULT_INPAINT_RADIUS,
) -> NDArray:
    """Inpaint every box out of ``image`` with Telea. Pixels outside the grown boxes are exact.

    ``image`` is ``(H, W)`` or ``(H, W, 3)`` uint8. ``bboxes`` is ``(N, 4)`` normalised
    ``cxcywh``, the YOLO label convention. Deterministic: Telea has no random state.
    """
    if len(bboxes) == 0:
        return image.copy()
    mask = box_mask(image.shape[:2], bboxes, dilate_px)
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


def _class_ids(manifest: DatasetManifest, classes: Sequence[str] | None) -> frozenset[int]:
    """Resolve class names to the manifest's indices; ``None`` means every class."""
    names = manifest.class_names
    if classes is None:
        return frozenset(range(len(names)))
    unknown = sorted(set(classes) - set(names))
    if unknown:
        raise ErasureError(f"unknown class name(s) {unknown}; {manifest.path} declares {names}")
    return frozenset(names.index(name) for name in classes)


def _kept_lines(text: str, keep: frozenset[int]) -> str:
    """The label lines whose class is in ``keep``, each copied as written."""
    lines = [line for line in text.splitlines() if line.strip()]
    return "".join(f"{line}\n" for line in lines if int(float(line.split()[0])) in keep)


def _copy_labels(source: Path, destination: Path, keep: frozenset[int] | None) -> int:
    """Copy ``*.txt`` only -- never ``labels.cache``. Returns the file count.

    ``keep=None`` copies each file byte for byte; otherwise only the lines of those classes.
    """
    destination.mkdir(parents=True, exist_ok=True)
    labels = sorted(source.glob(f"*{LABEL_SUFFIX}")) if source.is_dir() else []
    for label in labels:
        if keep is None:
            shutil.copyfile(label, destination / label.name)
        else:
            (destination / label.name).write_text(_kept_lines(label.read_text(), keep))
    return len(labels)


def write_erased_split(
    manifest: DatasetManifest,
    out_root: Path | str,
    split: str = "val",
    *,
    classes: Sequence[str] | None = None,
    dilate_px: int = DEFAULT_DILATE_PX,
    inpaint_radius: float = DEFAULT_INPAINT_RADIUS,
) -> Path:
    """Write an erased copy of one split, plus a ``data.yaml`` that ``t2o export`` can read.

    ``classes`` names the classes to erase, as ``names:`` spells them; ``None`` erases all.
    Every frame is written either way, so the export's layout and frame count do not depend
    on it.

    The copy keeps the source's own layout relative to its root, so ``manifest.pairing``
    resolves in it unchanged. Returns the written ``data.yaml``.

    **That ``data.yaml`` names the erased split as both ``train`` and ``val``.** ``t2o export``
    (``engine/export.py::export_translated``) reads both keys and ``DatasetManifest`` requires
    them. Pointing ``train`` at the source's unerased train split instead would make export
    translate hundreds of images nobody scores. A tree written here is a probe input, never a
    training set.

    ``path:`` is absolute, because ultralytics honours it literally rather than falling back
    to the manifest's directory as ``DatasetManifest`` does.

    Refuses a non-empty ``out_root``: a re-run into an old tree could leave stale frames or a
    ``labels.cache`` that ultralytics would trust.
    """
    keep = _class_ids(manifest, classes)
    out_root = Path(out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise ErasureError(
            f"{out_root} is not empty. Erase into a fresh directory; a re-run over an old "
            "tree could leave stale frames or a labels.cache behind."
        )

    source_images = manifest.train_images if split == "train" else manifest.val_images
    relative = source_images.relative_to(manifest.root)
    visible_out = out_root / relative
    infrared_out = manifest.pairing.infrared_path(visible_out)
    visible_out.mkdir(parents=True)
    infrared_out.mkdir(parents=True)

    visible_paths = sorted(p for p in source_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not visible_paths:
        raise ErasureError(f"no images found in {source_images}")
    manifest.pairing.validate_pairs(visible_paths)

    erased_boxes = 0
    for visible_path in visible_paths:
        name = f"{visible_path.stem}{ERASED_SUFFIX}"
        with Image.open(visible_path) as visible:
            visible.save(visible_out / name)

        cls, bboxes = load_yolo_labels(manifest.pairing.label_path(visible_path))
        bboxes = bboxes[[int(c) in keep for c in cls[:, 0].tolist()]]
        with Image.open(manifest.pairing.infrared_path(visible_path)) as infrared:
            thermal = np.asarray(infrared)
        erased = erase_boxes(
            thermal, bboxes.numpy(), dilate_px=dilate_px, inpaint_radius=inpaint_radius
        )
        Image.fromarray(erased).save(infrared_out / name)
        erased_boxes += len(bboxes)

    for images_dir, out_images in (
        (source_images, visible_out),
        (manifest.pairing.infrared_path(source_images), infrared_out),
    ):
        _copy_labels(
            images_dir.parent / LABELS_SEGMENT,
            out_images.parent / LABELS_SEGMENT,
            None if classes is None else keep,
        )

    destination = out_root / DATA_FILENAME
    destination.write_text(
        yaml.safe_dump(
            {
                "path": str(out_root),
                "train": relative.as_posix(),
                "val": relative.as_posix(),
                "nc": manifest.nc,
                "names": manifest.class_names,
                "rgbt": {
                    "visible_token": manifest.pairing.visible_token,
                    "infrared_token": manifest.pairing.infrared_token,
                },
                # Ignored by DatasetManifest and ultralytics; read by whoever writes the
                # probe's record, so the dilation used is on disk beside the frames.
                "erasure": {
                    "source": str(manifest.path),
                    "split": split,
                    "method": "opencv-telea",
                    "classes": [name for i, name in enumerate(manifest.class_names) if i in keep],
                    "dilate_px": dilate_px,
                    "inpaint_radius": inpaint_radius,
                },
            },
            sort_keys=False,
        )
    )
    logger.info(
        "erased %d box(es) from %d %s frame(s) (dilate %d px, radius %.1f) -> %s",
        erased_boxes,
        len(visible_paths),
        split,
        dilate_px,
        inpaint_radius,
        destination,
    )
    return destination
