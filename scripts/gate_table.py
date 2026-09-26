"""Reproduce M1's gate table on another dataset: the visible ceiling against the thermal floor.

One visible-trained judge, two validation passes, no training. The ceiling is that judge on the
visible val split; the floor is the same judge on the *same scenes'* thermal frames. Their
difference is the headroom translation has to work in, and on the custom set it was
0.9213 - 0.1887 = +0.733 (TASKS.md M1, GATE DECISION). Reproducing it on a public dataset is
E9's kill-test: it costs minutes and it gates a ~126 GPU-h cell, so it runs before the cell.

**Both arms are built with `t2o.data.budget.write_budget_manifest`, at fraction 1.0 and differing
only in `infrared`.** Three reasons, and the first two are why the source `data.yaml` is not simply
handed to ultralytics twice:

1. *The floor can go silently wrong.* Every adapted public dataset writes labels on the visible
   side only, so a hand-rolled thermal manifest yields a split ultralytics reads as entirely
   unlabelled -- it validates, it reports, and the number is meaningless.
   `write_budget_manifest` resolves the thermal side through `Pairing` and refuses a modality with
   no labels beside it (`budget.py::_require_labels_beside`), so this also *verifies* that
   `scripts/mirror_thermal_labels.py` has been run, before a GPU starts.
2. *A source manifest's `path:` is absolute and adapt-time.* `t2o`'s own loader tolerates a stale
   one (`manifest.py::_resolve_root` falls through to the manifest's directory), but ultralytics
   honours it literally and raises `FileNotFoundError` -- observed on a tree that had moved since
   it was adapted. What `write_budget_manifest` emits carries resolved `train`/`val` and no `path:`
   at all, so neither arm depends on that field being current.
3. *Symmetry is the experiment.* One construction with one flag flipped is what makes the ceiling
   and the floor differ in the pixels and in nothing else.

The `train.txt` each call also writes is never read -- see `annotation_sweep.py`'s arm A0, which
builds its thermal manifest the same way for the same reason.

**Why a primary subset of classes.** A mean over every class weights each one equally however few
instances it has. FLIR val carries 13 dog instances against 4,124 cars, so dog's AP50 is noise
holding 25% of a 4-class mean -- enough to move the reported headroom by the width of the decision
band below. `--primary-classes` states which classes the verdict is read off; the all-class mAP50
is reported beside it, never instead of it.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig` the way
`annotation_sweep.py` and `mirror_thermal_labels.py` do.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from t2o.data.budget import write_budget_manifest
from t2o.data.manifest import DatasetManifest

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CSV_FILENAME = "gate.csv"

# Pre-registered in TASKS.md M3 E9 (blocker 3's rule table) *before* any public-dataset number
# existed. Constants rather than prose so the band is fixed by the file rather than chosen by
# whoever reads the output.
KILL_THRESHOLD = 0.15
STRONG_THRESHOLD = 0.40


@dataclass(frozen=True, slots=True)
class Row:
    """One arm of the gate table: one judge, one val split, one set of pixels."""

    arm: str
    val_pixels: str
    map50: float  # every class ultralytics scored, its own mean
    primary_map50: float  # the same judge over `--primary-classes` only
    map50_95: float
    precision: float
    recall: float
    per_class_ap50: dict[str, float]
    weights: str
    data: str


def _primary_mean(per_class_ap50: dict[str, float], primary: Sequence[str]) -> float:
    """Mean AP50 over `primary`, skipping classes with no ground truth in the split.

    `metrics.task._extract_per_class_ap` omits a zero-instance class entirely rather than
    reporting 0.0, because "the detector missed every instance" and "there was nothing to detect"
    are different facts. Intersecting keeps this a mean over what was actually scored instead of
    one diluted by an absence.
    """
    scored = [per_class_ap50[name] for name in primary if name in per_class_ap50]
    if not scored:
        raise SystemExit(
            f"none of the primary classes {list(primary)} were scored -- the split has no "
            f"instances of any of them (it scored {sorted(per_class_ap50)})"
        )
    return sum(scored) / len(scored)


def _score(
    args: argparse.Namespace, data_yaml: Path, arm: str, val_pixels: str, primary: Sequence[str]
) -> Row:
    from t2o.metrics.task import evaluate_detector

    metrics = evaluate_detector(
        weights=Path(args.weights),
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    return Row(
        arm=arm,
        val_pixels=val_pixels,
        map50=metrics.map50,
        primary_map50=_primary_mean(metrics.per_class_ap50, primary),
        map50_95=metrics.map50_95,
        precision=metrics.precision,
        recall=metrics.recall,
        per_class_ap50=dict(metrics.per_class_ap50),
        weights=str(args.weights),
        data=str(data_yaml),
    )


def _verdict(headroom: float) -> str:
    if headroom >= STRONG_THRESHOLD:
        return "PASS -- the premise transfers. Run the cell."
    if headroom >= KILL_THRESHOLD:
        return (
            "WEAK PASS -- transfers weakly. Still run the cell: a smaller gain where thermal "
            "is legible is honest cross-dataset evidence."
        )
    return (
        "KILL -- translation has nothing to buy on this dataset. Do not spend the cell's "
        "GPU-hours; this floor measurement is itself the finding."
    )


def _write_csv(csv_path: Path, rows: Sequence[Row], class_names: Sequence[str]) -> None:
    """One row per arm, per-class AP50 flattened into `ap50_<name>` columns.

    Overwritten rather than appended: unlike E8's hours-long sweep there is nothing here worth
    resuming, and two stacked runs of a two-row table is how a stale floor gets read as current.
    """
    scalars = ("arm", "val_pixels", "map50", "primary_map50", "map50_95", "precision", "recall")
    per_class = tuple(f"ap50_{name}" for name in class_names)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*scalars, *per_class, "weights", "data"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: getattr(row, key) for key in (*scalars, "weights", "data")}
                # An absent class stays blank, not 0.0 -- the distinction _primary_mean keeps.
                | {f"ap50_{n}": row.per_class_ap50.get(n, "") for n in class_names}
            )


def _report(rows: Sequence[Row], class_names: Sequence[str], primary: Sequence[str]) -> str:
    """The gate table as markdown, ready to paste into TASKS.md."""
    header = ["arm", "mAP50", f"mAP50 ({len(primary)}-class)", "mAP50-95", *class_names]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = [
            f"{row.arm} ({row.val_pixels})",
            f"{row.map50:.4f}",
            f"**{row.primary_map50:.4f}**",
            f"{row.map50_95:.4f}",
            *(
                f"{row.per_class_ap50[name]:.4f}" if name in row.per_class_ap50 else "--"
                for name in class_names
            ),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="the paired dataset's data.yaml (or its root); both arms come from here",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        # The judge must be visible-trained and must never have supplied a training gradient
        # to anything this project produced -- invariant 7. The in-loop detector would be
        # grading its own homework, the contamination M1.2 step 1 went out of its way to rule out.
        help="the independent visible-trained judge, e.g. "
        "runs/reference-flir-yolo11s/weights/best.pt",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/gate"), help="output root")
    parser.add_argument(
        "--primary-classes",
        nargs="+",
        help="the classes the verdict is read off; defaults to every class in the manifest",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", help="'cpu', '0', or omit for auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    manifest = DatasetManifest.load(args.data)
    primary = list(args.primary_classes or manifest.class_names)
    unknown = [name for name in primary if name not in manifest.class_names]
    if unknown:
        raise SystemExit(f"--primary-classes {unknown} not in {manifest.class_names}")
    if not Path(args.weights).is_file():
        raise SystemExit(f"--weights not found: {args.weights}")

    out = Path(args.out).resolve()
    # Both manifests first, so everything checkable without a GPU is checked before the first
    # pass -- in particular an unmirrored infrared/labels, which is the failure that would
    # otherwise be scored rather than raised.
    manifests = out / "manifests"
    visible = write_budget_manifest(manifest, manifests / "visible", 1.0)
    thermal = write_budget_manifest(manifest, manifests / "thermal", 1.0, infrared=True)

    rows = [
        _score(args, visible, "ceiling", "visible", primary),
        _score(args, thermal, "floor", "thermal", primary),
    ]
    _write_csv(out / CSV_FILENAME, rows, manifest.class_names)

    ceiling, floor = rows
    headroom = ceiling.primary_map50 - floor.primary_map50
    logger.info("gate table:\n%s", _report(rows, manifest.class_names, primary))
    logger.info(
        "headroom on %d-class mAP50 (%s): %.4f - %.4f = %+.4f  [all-class: %+.4f]",
        len(primary),
        ", ".join(primary),
        ceiling.primary_map50,
        floor.primary_map50,
        headroom,
        ceiling.map50 - floor.map50,
    )
    logger.info("%s", _verdict(headroom))
    logger.info("written: %s", out / CSV_FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
