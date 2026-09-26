"""Mirror a split's visible-side labels onto its infrared side.

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
registered.** On FLIR-aligned the residual is ~5.9 px, structured as a fixed -1.18 deg camera
roll (TASKS.md's E9 section; measured in the sibling ``Thermal-Image-Registration`` repo
across three matchers against a ~0.2 px pipeline floor). A mirrored box therefore sits a few
pixels off its thermal object. At mAP50's IoU 0.5 threshold that is a small penalty on a
typical car or person box, and it pushes the measured raw-thermal score *down* -- which
**flatters** translation, because that score is the floor any translation gain is measured
against. Stated here as a pre-registered caveat rather than something to discover afterwards.
"""

from __future__ import annotations

import logging
import shutil

from t2o.data.manifest import DatasetManifest
from t2o.data.pairing import LABEL_SUFFIX, LABELS_SEGMENT

logger = logging.getLogger(__name__)


class MirrorError(Exception):
    """Raised when labels cannot be mirrored onto a dataset's infrared side."""


def mirror_labels_onto_infrared(
    manifest: DatasetManifest, *, force: bool = False
) -> dict[str, int]:
    """Copy each split's ``visible/labels/*.txt`` into that split's ``infrared/labels/``.

    Returns ``{split: files_written}``. Idempotent -- the copy is byte-for-byte, so a re-run
    rewrites identical files. Goes through ``manifest.pairing`` rather than string surgery for
    the same reason :func:`t2o.data.budget.write_budget_manifest` does: a dataset that names
    its modalities differently still works, and a layout with no visible segment fails loudly.

    Refuses a destination that already holds label files unless ``force``. **The custom
    dataset is exactly that case**, and its thermal labels may be drawn on the thermal frame
    in their own right -- overwriting those with visible-side boxes would quietly replace
    better annotation with worse, on the dataset every headline result rests on.
    """
    written: dict[str, int] = {}
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
        for label in labels:
            shutil.copyfile(label, destination / label.name)
        written[split] = len(labels)
        logger.info("%s: mirrored %d label file(s) -> %s", split, len(labels), destination)

    return written
