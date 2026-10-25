"""An inner train/val split carved out of the training split, for ModTr's calibration run.

modtr-baseline ADR-04 fixes ModTr's budget at 400 epochs, checked by one calibration run that
trains on 80% of the training images and watches the in-loop detector's mAP@50 on the other 20%
every epoch. Validation is the reporting split (`goal.md`), so the calibration must never read
it.

**A copy, not a cut in config.** The training corpus is read by `engine.trainer.Trainer`,
`engine.export.export_translated` and `translators._count_training_batches` (ModTr's `T_max`),
so a config-level split would have to reach all three. A tree of its own, with its own
``data.yaml``, needs none of them: the calibration is an ordinary run against that manifest,
and the real validation split is not in it to be touched. The manifest loader rejects text-file
splits and native Windows rules out symlinks (`budget.py`), so the pairs are copied.

The inner-val images are drawn through :func:`t2o.data.dataset.annotated_subset`, the seeded
selector every other subset in `t2o` uses.

**A capped corpus is split, not the whole folder (ADR-05).** FLIR's campaign trains on 600 of
its training pairs (`data.max_train_images`), so its calibration must split those 600, not the
folder they were drawn from. The images are listed and cut exactly as
:class:`t2o.data.dataset.TranslationPairDataset` lists and cuts them, so at the same cap and
seed the inner split's two halves are the campaign's corpus. A null cap splits every image.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import yaml

from t2o.data.dataset import IMAGE_SUFFIXES, annotated_subset, capped_subset
from t2o.data.manifest import RGBT_BLOCK, DatasetManifest

logger = logging.getLogger(__name__)

DATA_FILENAME = "data.yaml"
# Ignored by `DatasetManifest.load` and by ultralytics. `scripts/modtr_calibration.py verdict`
# reads it to refuse a run that was not trained on an inner split.
MARKER_KEY = "t2o_inner_split"


class InnerSplitError(ValueError):
    """Raised when an inner split cannot be written as asked."""


def write_inner_split(
    manifest: DatasetManifest,
    out_dir: Path | str,
    fraction: float = 0.2,
    seed: int = 0,
    max_images: int | None = None,
    subset_seed: int = 0,
) -> Path:
    """Copy ``manifest``'s training pairs into ``out_dir`` as an inner train/val split.

    The training images are first cut to ``max_images`` under ``subset_seed``, as a run's
    ``data.max_train_images`` and ``data.subset_seed`` cut them; null keeps them all.
    ``fraction`` of that corpus becomes the inner val split, the rest the inner train split; the
    source's val split is never read. Each pair's thermal image and both modalities' labels
    travel with it, laid out as the source lays them out. Returns the new ``data.yaml``.

    Refuses a non-empty ``out_dir``: a second split written over a first would leave a mix of
    the two that loads cleanly.
    """
    if not 0.0 < fraction < 1.0:
        raise InnerSplitError(f"fraction must be in (0, 1), got {fraction}")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise InnerSplitError(f"{out_dir} is not empty; write the inner split into a new directory")

    # The listing `TranslationPairDataset` makes, then its cut: anything else picks other images.
    images = sorted(
        p for p in manifest.train_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    images = capped_subset(images, max_images, subset_seed)
    held_out = annotated_subset(images, fraction, seed)
    if not held_out or len(held_out) == len(images):
        raise InnerSplitError(
            f"fraction {fraction} of {len(images)} training images leaves "
            f"{len(held_out)} for inner val and {len(images) - len(held_out)} for inner train; "
            "both must be non-empty"
        )

    # The directory holding both modalities of the training split (`train/` in the
    # `{split}/{visible,infrared}/{images,labels}` layout), so a pair is copied with its layout.
    pairing = manifest.pairing
    split_root = Path(
        os.path.commonpath([manifest.train_images, pairing.infrared_path(manifest.train_images)])
    )
    for image in images:
        split = "val" if image in held_out else "train"
        infrared = pairing.infrared_path(image)
        for source in (image, infrared, pairing.label_path(image), pairing.label_path(infrared)):
            # Thermal labels exist on the custom set but not on every adapted dataset.
            if source.is_file():
                destination = out_dir / split / source.relative_to(split_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

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
                    "visible_token": pairing.visible_token,
                    "infrared_token": pairing.infrared_token,
                },
                MARKER_KEY: {
                    "source": str(manifest.path),
                    "fraction": fraction,
                    "seed": seed,
                    "max_images": max_images,
                    "subset_seed": subset_seed,
                },
            },
            sort_keys=False,
        )
    )
    logger.info(
        "inner split %s: %d train / %d val of %s's %d training images "
        "(fraction %.2f, seed %d; cap %s, subset seed %d)",
        destination,
        len(images) - len(held_out),
        len(held_out),
        manifest.path,
        len(images),
        fraction,
        seed,
        max_images,
        subset_seed,
    )
    return destination
