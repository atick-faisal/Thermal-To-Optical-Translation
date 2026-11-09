"""C2's registration-error axis, label-transfer half: how fast does it degrade as alignment worsens?

Label transfer trains a thermal detector on boxes moved over from the visible image. Its quality
is bounded by the registration that moved them (``docs/goal.md`` C2, amended 2026-11-03). This
script injects a known misalignment into the thermal **train** boxes at each level of
``t2o.data.misalignment.REGISTRATION_LEVELS_PX``. At each level it fine-tunes a detector and
scores it on the untouched thermal val split, so the x-axis is pixels of registration error and
nothing else moves.

**One tree per level, shared by every seed.** :func:`write_misaligned_tree` copies the dataset
and misaligns only the copy's thermal train labels (plan "Label-transfer half"). Rewriting
``infrared/labels`` in place would hit ultralytics' size-keyed ``labels.cache``. A tree whose
``data.yaml`` exists was finished, so a resumed sweep reuses it. It is reused only when its
provenance digest is the level's, since a tree written for another frame size would otherwise
pass as this one.

**Val is the source's, byte for byte.** Each level is scored on its own tree's val copy, which
``write_misaligned_tree`` guarantees is unperturbed (design.md §5, invariant 2). Scoring every
level against the same boxes is what makes the points a curve.

**Every row is stamped** with the measured ``misalignment_px``, its ``misalignment_digest``, the
``boxes_dropped`` by the warp and the source's ``calibration_digest`` (FLIR's de-roll, or ""),
the way ``scripts/gate_table.py`` stamps ``calibration_digest``. A row whose labels changed state
on disk then cannot pass for one whose labels did not.

**Every trained row is re-scored by ``evaluate_detector``**, as in ``scripts/annotation_sweep.py``.
``DetectorResult`` has no per-class AP, and one evaluation path is design.md §5 invariant 1.

**Cheap cells first means seed-major here.** Every cell is the same ~10-minute fine-tune, so
order cannot save time. What it can save is the shape of a partial result. Running every level at
seed 0 before any level at seed 1 means an interruption leaves a whole curve at fewer seeds,
rather than three seeds of its left end.

**The custom set's 0 px is not truly zero.** Its thermal boxes were moved from visible by the
sister repo's registration, so every level sits on top of that error. The record says so; this
script only stamps what it ran.

Standalone script, not part of the ``t2o`` package. It owns its own ``logging.basicConfig``, like
``scripts/annotation_sweep.py``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import cast

from t2o.data.budget import write_budget_manifest
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import _split_image_shape, read_label_provenance
from t2o.data.misalignment import (
    DATA_FILENAME,
    REGISTRATION_LEVELS_PX,
    Misalignment,
    MisalignmentError,
    write_misaligned_tree,
)
from t2o.data.pairing import LABELS_SEGMENT
from t2o.metrics.task import primary_mean

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

CSV_FILENAME = "registration-tidy.csv"

# E8 arm A's settings (scripts/annotation_sweep.py). The custom set's 0 px cell is meant to
# reproduce E8 arm A at N=600 (0.9300, record 011), and it only can at the same init, epochs,
# image size and batch.
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16


@dataclass(frozen=True, slots=True)
class Row:
    """One cell of the sweep. ``dataset``/``misalignment_digest``/``seed`` identify it for resuming.

    Keyed on the digest rather than the pixel count: the digest also covers the frame size, so a
    sweep re-run at another resolution writes new cells instead of skipping them as done.
    """

    dataset: str
    misalignment_px: float  # measured off the field, not echoed from the CLI
    misalignment_digest: str
    seed: int
    # The recipe, with `epochs`. FLIR's 0 px cells are its C4 thermal-trained reference, which
    # `scripts/gate_table.py` reads only on the judge's recipe (plan Q5, Approach).
    init_weights: str
    epochs: int
    precision: float
    recall: float
    map50: float  # every class ultralytics scored, its own mean
    primary_map50: float | None  # the same pass over `--primary-classes` only; None without it
    map50_95: float
    # Boxes the warp pushed wholly off the frame. Non-zero means the level trained on fewer
    # boxes as well as moved ones, which a reader of a high-px cell needs to know.
    boxes_dropped: int
    # The de-roll the *source* labels carry (FLIR), or "" for labels drawn on thermal's own
    # frame. The FLIR 0 px point is only FLIR's label-transfer reference on de-rolled labels.
    calibration_digest: str
    weights: str
    data: str
    # Written as one `ap50_<class>` column each; a class val has no instances of is blank,
    # never 0.0 -- the distinction `primary_mean` relies on.
    per_class_ap50: dict[str, float]


def _completed(csv_path: Path) -> set[tuple[str, str, int]]:
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline="") as handle:
        return {
            (row["dataset"], row["misalignment_digest"], int(row["seed"]))
            for row in csv.DictReader(handle)
        }


def _append(csv_path: Path, row: Row, class_names: Sequence[str]) -> None:
    """Write one row and close the file, every time, so an interrupted sweep keeps its cells."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.is_file()
    scalars = [f.name for f in fields(Row) if f.name != "per_class_ap50"]
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*scalars, *(f"ap50_{name}" for name in class_names)]
        )
        if new:
            writer.writeheader()
        values = asdict(row)
        del values["per_class_ap50"]
        writer.writerow(
            values | {f"ap50_{name}": row.per_class_ap50.get(name, "") for name in class_names}
        )


def _plan(levels: Sequence[float], seeds: Sequence[int]) -> list[tuple[float, int]]:
    """Every ``(level, seed)`` cell, seed-major so a partial run is a whole curve (module doc)."""
    return [(level, seed) for seed in seeds for level in sorted(set(levels))]


def _level_tree(source: DatasetManifest, misalignment: Misalignment, out: Path) -> Path:
    """The level's misaligned tree, written once and reused across seeds and resumes.

    ``data.yaml`` is the last file ``write_misaligned_tree`` writes, so its presence means the
    tree is finished. A directory without it is a half-written tree from an interrupted copy;
    that is the human's to delete, not this script's.
    """
    tree = out / "trees" / f"level-{misalignment.level_px:g}px"
    data_yaml = tree / DATA_FILENAME
    if not data_yaml.is_file():
        try:
            return write_misaligned_tree(source, misalignment, tree)
        except MisalignmentError as error:
            raise SystemExit(
                f"cannot write the {misalignment.level_px:g} px tree: {error}"
            ) from None

    manifest = DatasetManifest.load(data_yaml)
    labels = manifest.pairing.infrared_path(manifest.train_images).parent / LABELS_SEGMENT
    recorded = read_label_provenance(labels) or {}
    if recorded.get("misalignment_digest") != misalignment.digest():
        raise SystemExit(
            f"{tree} holds digest {recorded.get('misalignment_digest')!r}, not this level's "
            f"{misalignment.digest()!r}; it was written for another level or frame size"
        )
    return data_yaml


def _cell(
    args: argparse.Namespace,
    tree_yaml: Path,
    misalignment: Misalignment,
    seed: int,
    calibration_digest: str,
) -> Row:
    from t2o.engine.detector_stage import train_detector
    from t2o.metrics.task import evaluate_detector

    out = Path(args.out).resolve()
    tree = DatasetManifest.load(tree_yaml)
    level = f"l{misalignment.level_px:g}"
    # Fraction 1.0 keeps every train image. The manifest exists to point both splits at the
    # thermal side, val on the tree's untouched copy. At 1.0 it ignores the seed: one per level.
    data_yaml = write_budget_manifest(tree, out / "manifests" / level, 1.0, infrared=True)
    result = train_detector(
        data_yaml=data_yaml,
        init_weights=Path(args.init_weights),
        project=out / "detectors",
        name=f"{args.dataset}-{level}-s{seed}",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=seed,
        device=args.device,
    )
    metrics = evaluate_detector(
        weights=result.weights,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    labels = tree.pairing.infrared_path(tree.train_images).parent / LABELS_SEGMENT
    provenance = read_label_provenance(labels) or {}
    return Row(
        dataset=args.dataset,
        misalignment_px=misalignment.magnitude_px(),
        misalignment_digest=misalignment.digest(),
        seed=seed,
        init_weights=str(args.init_weights),
        epochs=args.epochs,
        precision=metrics.precision,
        recall=metrics.recall,
        map50=metrics.map50,
        primary_map50=(
            primary_mean(metrics.per_class_ap50, args.primary_classes)
            if args.primary_classes
            else None
        ),
        map50_95=metrics.map50_95,
        # Written by `write_misaligned_tree` on every tree, level 0 included.
        boxes_dropped=cast(int, provenance["boxes_dropped"]),
        calibration_digest=calibration_digest,
        weights=str(result.weights),
        data=str(data_yaml),
        per_class_ap50=dict(metrics.per_class_ap50),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        # Not defaulted: on the server the dataset lives outside the repo, and the aligned FLIR
        # tree is the de-rolled mirror, not the adapter's raw output.
        help="the aligned paired dataset's data.yaml; its thermal side must carry labels",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        # Stamped on every row, so one CSV can hold both datasets' curves (the -07 report).
        help="the name stamped on every row, e.g. custom or flir",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/registration"), help="sweep output root"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=list(REGISTRATION_LEVELS_PX),
        help="corner-field px at the dataset's own frame size (the x-axis)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--primary-classes",
        nargs="+",
        # Not defaulted to every class: that mean is already the `map50` column.
        help="the classes primary_map50 is the mean over; omitted, that column is blank",
    )
    parser.add_argument("--init-weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default=None, help="'cuda:0', 'cpu', or omit for auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    source = DatasetManifest.load(args.data)
    unknown = [name for name in args.primary_classes or () if name not in source.class_names]
    if unknown:
        raise SystemExit(f"--primary-classes {unknown} not in {source.class_names}")

    # Built before the first GPU minute: an impossible level fails here, not at cell 9.
    height, width = _split_image_shape(source.pairing.infrared_path(source.train_images))
    try:
        levels = {level: Misalignment(height, width, level) for level in args.levels}
    except MisalignmentError as error:
        raise SystemExit(f"--levels: {error}") from None
    source_labels = source.pairing.infrared_path(source.train_images).parent / LABELS_SEGMENT
    recorded = read_label_provenance(source_labels)
    calibration_digest = str(recorded.get("calibration_digest", "")) if recorded else ""

    out = Path(args.out).resolve()
    csv_path = out / CSV_FILENAME
    done = _completed(csv_path)
    cells = _plan(args.levels, args.seeds)
    pending = [
        (level, seed)
        for level, seed in cells
        if (args.dataset, levels[level].digest(), seed) not in done
    ]
    logger.info(
        "%s at %dx%d: %d cell(s), %d already in %s, %d to run",
        args.dataset,
        width,
        height,
        len(cells),
        len(cells) - len(pending),
        csv_path,
        len(pending),
    )

    for index, (level, seed) in enumerate(pending, start=1):
        started = time.monotonic()
        misalignment = levels[level]
        logger.info("[%d/%d] %g px  seed=%d", index, len(pending), level, seed)
        tree_yaml = _level_tree(source, misalignment, out)
        row = _cell(args, tree_yaml, misalignment, seed, calibration_digest)
        _append(csv_path, row, source.class_names)
        logger.info(
            "[%d/%d] %g px  seed=%d  mAP50 %.4f  (%.1f min)",
            index,
            len(pending),
            level,
            seed,
            row.map50,
            (time.monotonic() - started) / 60.0,
        )

    logger.info("sweep complete: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
