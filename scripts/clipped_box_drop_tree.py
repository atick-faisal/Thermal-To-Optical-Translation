"""Copy a dataset tree with the val boxes the clamp left too loose to match removed.

``scripts/clipped_box_audit.py`` writes, for every custom box, the fraction of its area that
survived the clamp to the thermal frame. For a thin diagonal object that fraction is the best IoU
any detector can reach against it, so a val box kept at 0.4 is a guaranteed miss at mAP50. This
script builds the copy of the val tree that the sensitivity re-score scores against: every kept
val box whose retained fraction is below ``--threshold`` is removed, and nothing else changes
(``docs/features/clipped-box-audit/plan.md`` Q2).

**Why drop, not ignore.** Ignore regions would need a second scoring path, which ultralytics
``val`` cannot provide and invariant 1 forbids. Dropping biases mAP down through false positives
on the dropped objects, where the loose labels bias it down through missed matches; the re-score
counts the first so both are visible.

**Why a whole copied tree.** It follows ``t2o.data.misalignment.write_misaligned_tree``, and for
the same reasons. A rewrite in place would destroy the reporting split. ultralytics'
``labels.cache`` is keyed on file sizes, never on contents, and the copy is made pair by pair, so
no cache travels. The new ``data.yaml`` carries an absolute ``path:``, because ultralytics honours
it literally.

**Which line is which.** The audit's ``frozen_line`` column indexes the non-empty lines of
``val/infrared/labels/<stem>.txt``. The sister build writes the same text to ``visible/labels``,
so both sides are rewritten, and a visible file that differs from its thermal twin is refused.
A label file whose line count disagrees with the audit's kept boxes is refused too: the CSV
then describes another tree, and dropping by index would remove the wrong boxes.

Standalone script, not part of the ``t2o`` package -- it owns its own ``logging.basicConfig``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from t2o.data.labels import invalidate_label_cache
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import PROVENANCE_FILENAME, read_label_provenance
from t2o.data.misalignment import _copy_split, _write_manifest
from t2o.data.pairing import LABEL_SUFFIX, LABELS_SEGMENT

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
SPLIT = "val"  # the reporting split; train is copied byte for byte


class DropTreeError(RuntimeError):
    """Raised when the audit CSV and the dataset tree do not describe the same boxes."""


def read_retained_fractions(audit_csv: Path) -> dict[str, list[float]]:
    """Per val stem, each frozen label line's retained fraction, in line order.

    Only boxes the build kept carry a ``frozen_line``; dropped boxes are not in the tree.
    """
    indexed: dict[str, dict[int, float]] = defaultdict(dict)
    with audit_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != SPLIT or not row["frozen_line"]:
                continue
            indexed[row["stem"]][int(row["frozen_line"])] = float(row["retained_fraction"])
    fractions: dict[str, list[float]] = {}
    for stem, by_line in indexed.items():
        if sorted(by_line) != list(range(len(by_line))):
            raise DropTreeError(f"{audit_csv}: {stem}'s frozen lines are {sorted(by_line)}")
        fractions[stem] = [by_line[index] for index in range(len(by_line))]
    return fractions


def drop_lines(text: str, fractions: Sequence[float], threshold: float) -> tuple[str, int]:
    """``text`` without the non-empty lines whose fraction is below ``threshold``.

    Every other line keeps its exact bytes, line ending included. Returns the text and the
    number of lines removed.
    """
    lines = text.splitlines(keepends=True)
    boxes = [index for index, line in enumerate(lines) if line.strip()]
    if len(boxes) != len(fractions):
        raise DropTreeError(f"{len(boxes)} label lines but the audit kept {len(fractions)} boxes")
    removed = {
        line_index
        for line_index, fraction in zip(boxes, fractions, strict=True)
        if fraction < threshold
    }
    kept = (line for index, line in enumerate(lines) if index not in removed)
    return "".join(kept), len(removed)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_drop_tree(
    manifest: DatasetManifest, audit_csv: Path, threshold: float, out_dir: Path
) -> Path:
    """Copy ``manifest``'s tree into ``out_dir`` minus the val boxes below ``threshold``.

    Returns the copy's ``data.yaml``. Refuses a non-empty ``out_dir``, so two thresholds never
    land in one tree.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DropTreeError(f"{out_dir} is not empty; write each threshold into a new directory")
    fractions = read_retained_fractions(audit_csv)

    _copy_split(manifest, manifest.train_images, out_dir / "train")
    infrared_labels = _copy_split(manifest, manifest.val_images, out_dir / SPLIT)
    split_root = Path(
        os.path.commonpath(
            [manifest.val_images, manifest.pairing.infrared_path(manifest.val_images)]
        )
    )
    visible_labels = (
        out_dir / SPLIT / manifest.val_images.relative_to(split_root).parent / LABELS_SEGMENT
    )

    stems = {path.stem for path in infrared_labels.glob(f"*{LABEL_SUFFIX}")}
    if missing := sorted(set(fractions) - stems):
        raise DropTreeError(f"the audit has kept boxes for stems with no label file: {missing}")
    dropped = changed = 0
    for stem in sorted(stems):
        infrared = infrared_labels / f"{stem}{LABEL_SUFFIX}"
        visible = visible_labels / f"{stem}{LABEL_SUFFIX}"
        text = infrared.read_text()
        if visible.is_file() and visible.read_text() != text:
            raise DropTreeError(f"{visible} differs from {infrared}; the audit indexes the latter")
        try:
            new_text, lost = drop_lines(text, fractions.get(stem, []), threshold)
        except DropTreeError as error:
            raise DropTreeError(f"{infrared}: {error}. Is --audit-csv for this tree?") from error
        if lost:
            dropped += lost
            changed += 1
            for path in (infrared, visible):
                if path.is_file():
                    path.write_text(new_text)

    source_labels = manifest.pairing.infrared_path(manifest.val_images).parent / LABELS_SEGMENT
    provenance = {
        "labels": "val labels with low-visibility boxes removed (clipped-box audit)",
        "source": str(manifest.path),
        "min_retained_fraction": threshold,
        "audit_csv": str(audit_csv),
        "audit_csv_sha256": _sha256(audit_csv),
        "files": len(stems),
        "files_changed": changed,
        "boxes_dropped": dropped,
        "source_provenance": read_label_provenance(source_labels),
    }
    record = json.dumps(provenance, indent=2) + "\n"
    for labels_dir in (infrared_labels, visible_labels):
        # A fresh copy has no cache; this states the invariant rather than trusting it.
        invalidate_label_cache(labels_dir)
        (labels_dir.parent / PROVENANCE_FILENAME).write_text(record)

    destination = _write_manifest(manifest, out_dir)
    logger.info(
        "drop tree %s: %d val box(es) below %.2f removed from %d of %d file(s)",
        destination,
        dropped,
        threshold,
        changed,
        len(stems),
    )
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Paths and the threshold are required: none of them has a safe default."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--data", type=Path, required=True, help="source data.yaml")
    parser.add_argument("--audit-csv", type=Path, required=True, help="clipped_box_audit.py CSV")
    parser.add_argument("--threshold", type=float, required=True, help="e.g. 0.5 or 0.75")
    parser.add_argument("--out", type=Path, required=True, help="new, empty tree directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build one drop tree."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    write_drop_tree(DatasetManifest.load(args.data), args.audit_csv, args.threshold, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
