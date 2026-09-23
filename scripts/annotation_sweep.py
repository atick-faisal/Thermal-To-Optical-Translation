"""E8: at what annotation budget does translation stop being worth it?

The project's causality, stability and faithfulness criteria are met; **margin against a
real baseline is not** (TASKS.md M3 E8). This script produces that baseline and the curve
around it: a detector trained directly on thermal, against the same detector trained on
translated frames, at annotation budgets from 10 images to the full 600.

Five arms land in one tidy CSV. Each row says which model produced it and which pixels it
was scored on, because those differ **by design** and a table that hid it would be wrong:

| arm | detector | trained on | scored on | annotations |
| --- | --- | --- | --- | --- |
| `A`  | fine-tuned from `--init-weights` | `N` thermal | thermal val | `N` |
| `B`  | fine-tuned from `--init-weights` | `N` translated (lambda=0) | translated val | `N` |
| `A0` | `--reference-weights`, untouched | -- | thermal val | **0** |
| `C`  | `--reference-weights`, untouched | -- | translated val (lambda=0) | **0** |
| `D`  | `--reference-weights`, untouched | -- | translated val (lambda>0) | full split |

Arms A and B are paired by construction -- same scenes, same boxes, same init, same epochs,
same seed, same budget -- so the only thing that differs is the pixels. `A0` and `C` are the
zero-annotation anchors and `A`'s crossover with `C` is the headline: below that `N`,
translating is worth more than annotating. `D` carries the full budget because its
annotations were spent inside the translator's coupling term rather than on a detector, which
is C1's practical claim stated in the same units.

**The seed does two jobs at once, deliberately.** It selects *which* `N` images get annotated
and it seeds the detector's training. Splitting them would report error bars that exclude the
luck of the draw, and at `N=10` that luck is most of the variance. One seed per row therefore
means "this whole pipeline, run again from the top".

**`A0` has no seed dependence** -- one fixed judge on one fixed split -- so it is written once
with `seed = -1` rather than duplicated across seeds to look symmetric.

**Why a script and not a CLI subcommand.** `t2o evaluate` and `t2o train-detector` only log
their numbers; a 42-cell sweep through them would leave the result in a terminal scrollback.
This calls `train_detector`/`evaluate_detector` in-process and appends each row to the CSV as
it completes, so an interrupted sweep keeps everything it has already paid for. Re-running
skips the cells already in the CSV. Cheap cells run first (zero-shot anchors, then ascending
`N`), so an interruption leaves a usable curve rather than a usable corner.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig` the
way `t2o/cli.py` does for the package proper, following `scripts/loss_share.py`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from t2o.data.budget import TRAIN_LIST_FILENAME, write_budget_manifest
from t2o.data.dataset import IMAGE_SUFFIXES
from t2o.data.manifest import DatasetManifest

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

CSV_FILENAME = "e8-tidy.csv"
TRAINED_ARMS = ("A", "B")
ZERO_SHOT_ARMS = ("A0", "C", "D")
ALL_ARMS = (*ZERO_SHOT_ARMS, *TRAINED_ARMS)
UNSEEDED = -1

# Defaults are the E3 campaign's own evaluation-detector settings
# (experiments/e3_pix2pix_control.yaml, `detector.evaluation`). Held constant across budgets:
# see the fixed-epoch caveat in TASKS.md M3 E8 -- at N=10 this overfits, which makes the
# absolute number unflattering but keeps A and B comparable, and comparability is the point.
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_BUDGETS = (10, 25, 50, 100, 200, 400, 600)


@dataclass(frozen=True, slots=True)
class Row:
    """One cell of the sweep. `arm`/`n_annotated`/`seed` identify it for resuming."""

    arm: str
    n_annotated: int
    seed: int
    detector: str  # "finetuned" or "reference" -- the arms are not all the same model family
    val_pixels: str  # "thermal", "translated-l0", "translated-lpos"
    epochs: int  # 0 on a zero-shot row, so a re-run at other epochs is distinguishable
    precision: float
    recall: float
    map50: float
    map50_95: float
    weights: str
    data: str


def _count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _budget_manifest(
    manifest: DatasetManifest, out_dir: Path, n: int, seed: int, *, infrared: bool
) -> Path:
    """A manifest holding exactly `n` annotated train images, or an error.

    `write_budget_manifest` is fraction-based because it shares one selector with the
    translator's own `annotation_fraction` cut -- that sharing is what makes "annotation
    budget" one quantity across E8's arms. Converting back here and *checking* the count is
    what stops a rounding edge turning an `N=25` row into a 24-image run labelled 25.
    """
    total = _count_images(manifest.train_images)
    written = write_budget_manifest(manifest, out_dir, n / total, seed=seed, infrared=infrared)
    listed = [line for line in (out_dir / TRAIN_LIST_FILENAME).read_text().splitlines() if line]
    if len(listed) != n:
        raise SystemExit(f"{written}: asked for {n} images, got {len(listed)} of {total}")
    return written


def _export_manifest(template: str, seed: int) -> DatasetManifest:
    """The stage-3 translated export belonging to one campaign seed.

    A template rather than a list of paths: the pairing of detector seed to *export* seed has
    to be structural. Passing six directories and three seeds would misalign silently, and the
    result would be error bars that look like translator variance and are not.
    """
    return DatasetManifest.load(Path(template.format(seed=seed)) / "data.yaml")


def _completed(csv_path: Path) -> set[tuple[str, int, int]]:
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline="") as handle:
        return {
            (row["arm"], int(row["n_annotated"]), int(row["seed"]))
            for row in csv.DictReader(handle)
        }


def _append(csv_path: Path, row: Row) -> None:
    """Write one row and close the file, every time.

    Flushing per cell rather than at the end is the whole resume story: a sweep is hours long
    and an interrupted one must keep the GPU time it has already spent.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.is_file()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Row)])
        if new:
            writer.writeheader()
        writer.writerow(asdict(row))


def _train_cell(
    args: argparse.Namespace, source: DatasetManifest, arm: str, n: int, seed: int
) -> Row:
    from t2o.engine.detector_stage import train_detector

    out = Path(args.out).resolve()
    name = f"{arm}-n{n}-s{seed}"
    infrared = arm == "A"
    data_yaml = _budget_manifest(source, out / "manifests" / name, n, seed, infrared=infrared)
    result = train_detector(
        data_yaml=data_yaml,
        init_weights=Path(args.init_weights),
        project=out / "detectors",
        name=name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=seed,
        device=args.device,
    )
    return Row(
        arm=arm,
        n_annotated=n,
        seed=seed,
        detector="finetuned",
        val_pixels="thermal" if infrared else "translated-l0",
        epochs=args.epochs,
        precision=result.precision,
        recall=result.recall,
        map50=result.map50,
        map50_95=result.map50_95,
        weights=str(result.weights),
        data=str(data_yaml),
    )


def _zero_shot_cell(
    args: argparse.Namespace, data_yaml: Path, arm: str, n: int, seed: int, val_pixels: str
) -> Row:
    from t2o.metrics.task import evaluate_detector

    metrics = evaluate_detector(
        weights=Path(args.reference_weights),
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    return Row(
        arm=arm,
        n_annotated=n,
        seed=seed,
        detector="reference",
        val_pixels=val_pixels,
        epochs=0,
        precision=metrics.precision,
        recall=metrics.recall,
        map50=metrics.map50,
        map50_95=metrics.map50_95,
        weights=str(args.reference_weights),
        data=str(data_yaml),
    )


def _plan(
    arms: Sequence[str], budgets: Sequence[int], seeds: Sequence[int], total: int
) -> list[tuple[str, int, int]]:
    """Every cell, cheapest first.

    Zero-shot anchors are minutes and the flat lines they draw are what the trained arms are
    read against, so they go first; the trained cells then run at ascending `N`. An
    interruption therefore leaves a curve with a thin tail rather than a tall left edge.
    """
    cells: list[tuple[str, int, int]] = []
    if "A0" in arms:
        cells.append(("A0", 0, UNSEEDED))
    for arm, n in (("C", 0), ("D", total)):
        if arm in arms:
            cells.extend((arm, n, seed) for seed in seeds)
    for n in sorted(budgets):
        cells.extend((arm, n, seed) for arm in TRAINED_ARMS if arm in arms for seed in seeds)
    return cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        # Not defaulted, and deliberately so: on the server the paired dataset lives outside
        # the repo entirely (TASKS.md M3 E8 step 0), so a default of `dataset/yolo_rgbt` would
        # be wrong on the only machine that runs this.
        help="the paired dataset's data.yaml -- arm A's thermal side comes from here",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/e8"), help="sweep output root")
    parser.add_argument(
        "--arms", nargs="+", default=list(ALL_ARMS), choices=ALL_ARMS, help="arms to run"
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS), help="the x-axis, in N"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--control-template",
        default="runs/e3b-control-s{seed}/stage3/translated",
        help="lambda=0 export per seed; '{seed}' is substituted (arms B and C)",
    )
    parser.add_argument(
        "--loop-template",
        default="runs/e3b-loop-s{seed}/stage3/translated",
        help="lambda>0 export per seed; '{seed}' is substituted (arm D)",
    )
    parser.add_argument("--init-weights", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument(
        "--reference-weights",
        type=Path,
        default=Path("runs/reference-yolo11s/weights/best.pt"),
        # The judge that never supplied a training gradient (M1.2 step 1). Arms A0/C/D are only
        # meaningful read off it; the in-loop yolo11n would be grading its own homework.
        help="the visible-trained reference judge, for the zero-shot arms",
    )
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
    total = _count_images(source.train_images)
    oversized = [n for n in args.budgets if n > total or n < 1]
    if oversized:
        raise SystemExit(f"budgets {oversized} are not in 1..{total} (the train split's size)")

    # Everything that can be checked without a GPU is checked before the first one starts: a
    # sweep is hours long and discovering a missing export at cell 30 wastes all of them.
    needs_reference = any(arm in args.arms for arm in ZERO_SHOT_ARMS)
    if needs_reference and not Path(args.reference_weights).is_file():
        raise SystemExit(f"--reference-weights not found: {args.reference_weights}")
    exports: dict[tuple[str, int], DatasetManifest] = {}
    for arm, template in (
        ("B", args.control_template),
        ("C", args.control_template),
        ("D", args.loop_template),
    ):
        if arm in args.arms:
            exports.update({(arm, seed): _export_manifest(template, seed) for seed in args.seeds})

    out = Path(args.out).resolve()
    csv_path = out / CSV_FILENAME
    done = _completed(csv_path)
    cells = _plan(args.arms, args.budgets, args.seeds, total)
    pending = [cell for cell in cells if cell not in done]
    logger.info(
        "%d cell(s): %d already in %s, %d to run",
        len(cells),
        len(cells) - len(pending),
        csv_path,
        len(pending),
    )

    for index, (arm, n, seed) in enumerate(pending, start=1):
        started = time.monotonic()
        logger.info("[%d/%d] arm %s  N=%d  seed=%d", index, len(pending), arm, n, seed)
        if arm in TRAINED_ARMS:
            manifest = source if arm == "A" else exports[("B", seed)]
            row = _train_cell(args, manifest, arm, n, seed)
        elif arm == "A0":
            # The raw-thermal floor: the judge, on thermal, having seen no thermal at all.
            # Built at fraction 1.0 only to obtain a thermal *val* path; nothing trains on it.
            thermal = _budget_manifest(source, out / "manifests" / "A0", total, 0, infrared=True)
            row = _zero_shot_cell(args, thermal, arm, n, seed, "thermal")
        else:
            manifest = exports[(arm, seed)]
            pixels = "translated-l0" if arm == "C" else "translated-lpos"
            row = _zero_shot_cell(args, manifest.path, arm, n, seed, pixels)
        _append(csv_path, row)
        logger.info(
            "[%d/%d] arm %s  N=%d  seed=%d  mAP50 %.4f  (%.1f min)",
            index,
            len(pending),
            arm,
            n,
            seed,
            row.map50,
            (time.monotonic() - started) / 60.0,
        )

    logger.info("sweep complete: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
