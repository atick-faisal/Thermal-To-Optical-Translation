"""Annotation-budget manifests for E8's low-annotation sweep.

E8 asks at what annotation budget translation beats training a detector directly on
thermal, so every arm needs the *detector* trained on the same ``N`` images. That is a
different cut from ``DataConfig.annotation_fraction``, which gates only the batch's
``cls``/``bboxes`` inside the translator's coupling term -- PLAN.md §16 records why
lowering that knob inside E3 is the wrong experiment: the lambda=0 control never reads
annotations at all, so it makes the two arms *more* alike rather than less.

The budget is expressed as a fraction rather than a count because
:func:`t2o.data.dataset.annotated_subset` -- the same selector the translator side uses --
is fraction-based, and reusing it is what keeps the two sides selecting identical images.
``round(fraction * len(paths))`` recovers an exact count when a caller passes ``n / len``,
so a count-driven sweep needs no separate arithmetic.

What comes out is an **ultralytics manifest, not one of ours**: ``train`` points at a text
file of image paths, which is YOLO's own convention for a subset and which
:meth:`t2o.data.manifest.DatasetManifest.load` deliberately rejects (it requires split
directories). Nothing here should ever be fed back to the translator's data layer, and the
rejection is the guard that says so. A directory of symlinks would have loaded on both
sides, and PLAN.md §3 rules symlinks out on native Windows anyway.

Labels are resolved by ultralytics itself, which substitutes ``/images/`` -> ``/labels/``
in each listed path (``ultralytics/data/utils.py::img2label_paths``). That holds for the
source layout (``{split}/{visible,infrared}/{images,labels}``) and for an export's
(``{split}/{images,labels}``) alike, which is why one function serves E8's thermal arm and
its translated arm with no branch between them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from t2o.data.dataset import IMAGE_SUFFIXES, annotated_subset
from t2o.data.manifest import DatasetManifest
from t2o.data.pairing import LABEL_SUFFIX, LABELS_SEGMENT

logger = logging.getLogger(__name__)

DATA_FILENAME = "data.yaml"
TRAIN_LIST_FILENAME = "train.txt"


class BudgetError(Exception):
    """Raised when an annotation budget cannot produce a trainable manifest."""


def _require_labels_beside(images_dir: Path) -> None:
    """Fail if the labels ultralytics will look for are not there at all.

    The thermal arm is the one that can go silently wrong. Every adapted public dataset in
    this repo writes labels under ``visible/labels`` only (``data/adapters/common.py``), so
    pointing a detector at ``infrared/images`` there yields a dataset ultralytics reads as
    entirely unlabelled -- it trains, it reports, and the number is meaningless. The custom
    pairs do carry both sides, which is exactly why the difference is easy to miss.

    Checked at the directory rather than per file because an individual image with no boxes
    is legitimate data (a true negative), and rejecting those would refuse a real dataset.
    """
    labels_dir = images_dir.parent / LABELS_SEGMENT
    if not labels_dir.is_dir() or not any(labels_dir.glob(f"*{LABEL_SUFFIX}")):
        raise BudgetError(
            f"{images_dir} has no labels beside it ({labels_dir} is missing or empty). "
            "A detector trained here would read every image as a negative and report a "
            "number that means nothing. Mirror the labels onto this modality first."
        )


def write_budget_manifest(
    manifest: DatasetManifest,
    out_dir: Path | str,
    fraction: float,
    *,
    seed: int = 0,
    infrared: bool = False,
) -> Path:
    """Write a data.yaml whose train split is ``fraction`` of ``manifest``'s, val untouched.

    ``infrared`` switches the images to the thermal side of the same pairs -- E8's arm A
    (a detector trained directly on thermal) against arm B (the same detector trained on
    translated frames). Both read the identical scenes and the identical boxes, so the
    only thing that differs between the two arms is the pixels, which is what makes the
    contrast between them paired. It goes through :class:`t2o.data.pairing.Pairing` rather
    than string surgery so a dataset that names its modalities differently still works, and
    so a layout with no visible segment at all -- an export -- fails loudly instead of
    silently handing back the visible path.

    The validation split is never subsampled: a budget is a statement about training data,
    and every point on the curve has to be scored against the same frames to be a curve.
    """
    out_dir = Path(out_dir)
    train_images = sorted(
        p for p in manifest.train_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not train_images:
        raise BudgetError(f"no images found in {manifest.train_images}")

    keep = annotated_subset(train_images, fraction, seed)
    # Filtering the sorted list rather than iterating the frozenset: the file has to be
    # byte-identical across runs for the same (fraction, seed), and set iteration order is
    # not a promise. Two arms diffing their train.txt is how a budget mismatch gets caught.
    selected = [p for p in train_images if p in keep]
    if not selected:
        raise BudgetError(
            f"fraction {fraction} of {len(train_images)} images selects nothing. "
            "A zero budget is not a trainable manifest -- E8's N=0 point is the "
            "untrained detector, which is an evaluation and not a fine-tune."
        )

    val_images = manifest.val_images
    if infrared:
        selected = [manifest.pairing.infrared_path(p) for p in selected]
        val_images = manifest.pairing.infrared_path(val_images)
        _require_labels_beside(manifest.pairing.infrared_path(manifest.train_images))
        _require_labels_beside(val_images)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_list = out_dir / TRAIN_LIST_FILENAME
    train_list.write_text("".join(f"{p.resolve()}\n" for p in selected))

    destination = out_dir / DATA_FILENAME
    destination.write_text(
        yaml.safe_dump(
            {
                # Absolute, and `path` deliberately left out: ultralytics joins `path`
                # onto a relative `train`/`val`, and these two are already resolved.
                "train": str(train_list.resolve()),
                "val": str(val_images.resolve()),
                "nc": manifest.nc,
                "names": manifest.class_names,
            },
            sort_keys=False,
        )
    )
    logger.info(
        "budget manifest %s: %d/%d train images (fraction %.4f, seed %d, %s)",
        destination,
        len(selected),
        len(train_images),
        fraction,
        seed,
        "infrared" if infrared else "visible",
    )
    return destination
