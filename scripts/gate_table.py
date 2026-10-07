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

**The third arm, and why the headroom is two gaps.** V - J (ceiling minus floor) is a *total*.
`--thermal-weights` adds T -- a thermal-trained detector of the judge's architecture and recipe,
scored on the floor's own thermal val split -- and splits it in two that mean different things
(docs/features/msrs-passive-gate/plan.md, Summary): V - T, the *sensor gap*, is what colour shows
that thermal does not, the only room translation has to beat direct thermal detection; T - J, the
*domain gap*, is what the thermal pixels already carry that the visible judge cannot read. On a
class where V ~ T, no loop can beat a thermal-trained detector, however well it closes T - J.

**Seeds.** Each weights flag takes one checkpoint per seed (plan.md Q7). `gate.csv` keeps one row
per validation pass, so it can always be re-averaged; only the report and the log carry the means,
spread and gaps. J is scored once per V checkpoint, so V and J are paired by seed and T is not --
which is why a gap is a difference of seed means and carries no spread of its own.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig` the way
`annotation_sweep.py` and `mirror_thermal_labels.py` do.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from t2o.data.budget import write_budget_manifest
from t2o.data.calibration import load_calibration
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import read_label_provenance
from t2o.data.pairing import LABELS_SEGMENT
from t2o.metrics.task import PrimaryClassError, primary_mean

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CSV_FILENAME = "gate.csv"
CEILING, FLOOR, THERMAL_TRAINED = "ceiling", "floor", "thermal-trained"

# Pre-registered in TASKS.md M3 E9 (blocker 3's rule table) *before* any public-dataset number
# existed. Constants rather than prose so the band is fixed by the file rather than chosen by
# whoever reads the output.
KILL_THRESHOLD = 0.15
STRONG_THRESHOLD = 0.40
# Pre-registered in docs/features/msrs-passive-gate/plan.md Q6, before any MSRS number existed:
# about 1.7x the 0.059 mAP50 judge-to-judge noise floor (F13, cited by F102), so a smaller V - T
# cannot be told from two judges disagreeing. Read per primary class, and only off a non-KILL gate.
PASSIVE_THRESHOLD = 0.10


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
    # Which residual-misalignment constant the thermal labels were de-rolled by, or "" for the
    # plain mirror. Stamped on both rows because a floor is only comparable to another floor
    # measured against the same boxes, and the two states are the same filenames on disk.
    calibration_digest: str


@dataclass(frozen=True, slots=True)
class Spread:
    """One number across seeds."""

    mean: float
    std: float  # sample std (ddof=1); nan at one seed, where spread is undefined rather than 0


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """One arm across its seeds -- what the report prints and the verdicts read."""

    arm: str
    val_pixels: str
    seeds: int
    map50: Spread
    primary_map50: Spread
    map50_95: Spread
    per_class_ap50: dict[str, Spread]  # a class no pass scored is absent, never 0.0


def _primary_mean(per_class_ap50: dict[str, float], primary: Sequence[str]) -> float:
    """`metrics.task.primary_mean`, with its error turned into a CLI exit.

    The arithmetic lives in the library because the campaign aggregator reads the same mean
    (`analysis/aggregate.py::add_primary_mean`), and a headline computed two ways is a headline
    that can disagree with itself. This wrapper exists only to keep the gate's failure a clean
    message rather than a traceback.
    """
    try:
        return primary_mean(per_class_ap50, primary)
    except PrimaryClassError as error:
        raise SystemExit(str(error)) from error


def _score(
    args: argparse.Namespace,
    weights: Path,
    data_yaml: Path,
    arm: str,
    val_pixels: str,
    primary: Sequence[str],
    calibration_digest: str,
) -> Row:
    from t2o.metrics.task import evaluate_detector

    metrics = evaluate_detector(
        weights=weights,
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
        weights=str(weights),
        data=str(data_yaml),
        calibration_digest=calibration_digest,
    )


def _spread(values: Sequence[float]) -> Spread:
    """Sample std, the `analysis/aggregate.py` convention, so a gate and a campaign agree."""
    return Spread(
        statistics.mean(values), statistics.stdev(values) if len(values) > 1 else math.nan
    )


def _summarise(rows: Sequence[Row]) -> list[ArmSummary]:
    """One summary per arm, in the order the arms were scored."""
    summaries = []
    for arm in dict.fromkeys(row.arm for row in rows):
        passes = [row for row in rows if row.arm == arm]
        names = dict.fromkeys(name for row in passes for name in row.per_class_ap50)
        summaries.append(
            ArmSummary(
                arm=arm,
                val_pixels=passes[0].val_pixels,
                seeds=len(passes),
                map50=_spread([row.map50 for row in passes]),
                primary_map50=_spread([row.primary_map50 for row in passes]),
                map50_95=_spread([row.map50_95 for row in passes]),
                per_class_ap50={
                    name: _spread(
                        [r.per_class_ap50[name] for r in passes if name in r.per_class_ap50]
                    )
                    for name in names
                },
            )
        )
    return summaries


def _per_class_gap(minuend: ArmSummary, subtrahend: ArmSummary) -> dict[str, float]:
    """Difference of seed means, per class both arms scored."""
    return {
        name: spread.mean - subtrahend.per_class_ap50[name].mean
        for name, spread in minuend.per_class_ap50.items()
        if name in subtrahend.per_class_ap50
    }


def _verified_digest(manifest: DatasetManifest, calibration: Path | None) -> str:
    """The digest to stamp on both rows, checked against the tree the floor will be scored on.

    E9's first de-rolled gate stamped a digest on a row whose boxes were *not* de-rolled: the
    column recorded that a flag had been passed, not that the tree was in that state, so it
    asserted a fact nothing had checked (TASKS.md M3 E9 step 3b). The check is symmetric -- a
    de-rolled tree scored without ``--calibration`` is refused too -- because a column that is only
    right when someone remembers the flag is not traceability.

    Val only: val is the split both arms are scored on, and a de-rolled train split cannot reach
    the number.
    """
    labels_dir = manifest.pairing.infrared_path(manifest.val_images).parent / LABELS_SEGMENT
    recorded = read_label_provenance(labels_dir)
    on_disk = str(recorded.get("calibration_digest", "")) if recorded else ""
    expected = load_calibration(calibration).digest() if calibration else ""
    if on_disk == expected:
        return expected
    if expected:
        raise SystemExit(
            f"{labels_dir} was not de-rolled by {calibration} (digest {expected}): it records "
            f"{on_disk or 'a plain mirror'}. Run scripts/mirror_thermal_labels.py --data "
            f"{manifest.root} --calibration {calibration} --force first -- scoring this tree "
            "would report an uncorrected floor under a de-rolled label."
        )
    raise SystemExit(
        f"{labels_dir} is de-rolled (digest {on_disk}) but no --calibration was given, so the "
        "table would claim a plain-mirror floor. Pass --calibration, or restore the plain copies "
        f"with scripts/mirror_thermal_labels.py --data {manifest.root} --force."
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


def _sensor_gap_verdict(gap: float) -> str:
    if gap >= PASSIVE_THRESHOLD:
        return (
            "passive GO -- colour shows what thermal does not; this class tests the passive claim."
        )
    return (
        "domain-gap GO -- a thermal-trained detector sees it nearly as well as colour does; worth "
        "running for Consistency, but it cannot test the passive claim."
    )


def _log_sensor_gap(
    ceiling: ArmSummary,
    floor: ArmSummary,
    trained: ArmSummary,
    primary: Sequence[str],
    headroom: float,
) -> None:
    """Q6's second part: per primary class, read only once V - J has cleared the kill line."""
    logger.info(
        "on %d-class mAP50: sensor gap V - T %+.4f, domain gap T - J %+.4f",
        len(primary),
        ceiling.primary_map50.mean - trained.primary_map50.mean,
        trained.primary_map50.mean - floor.primary_map50.mean,
    )
    if headroom < KILL_THRESHOLD:
        logger.info("no per-class GO marks: V - J is in the KILL band (plan.md Q6)")
        return
    sensor_gap = _per_class_gap(ceiling, trained)
    for name in primary:
        if name in sensor_gap:
            logger.info(
                "%s: V - T %+.4f -> %s",
                name,
                sensor_gap[name],
                _sensor_gap_verdict(sensor_gap[name]),
            )


def _write_csv(csv_path: Path, rows: Sequence[Row], class_names: Sequence[str]) -> None:
    """One row per validation pass, per-class AP50 flattened into `ap50_<name>` columns.

    Raw passes, never seed means: the means are derived, and a CSV of them could not be
    re-averaged. Overwritten rather than appended: unlike E8's hours-long sweep there is nothing
    here worth resuming, and two stacked runs of one table is how a stale floor gets read as
    current.
    """
    scalars = ("arm", "val_pixels", "map50", "primary_map50", "map50_95", "precision", "recall")
    per_class = tuple(f"ap50_{name}" for name in class_names)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*scalars, *per_class, "weights", "data", "calibration_digest"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: getattr(row, key)
                    for key in (*scalars, "weights", "data", "calibration_digest")
                }
                # An absent class stays blank, not 0.0 -- the distinction _primary_mean keeps.
                | {f"ap50_{n}": row.per_class_ap50.get(n, "") for n in class_names}
            )


def _cell(spread: Spread, seeds: int) -> str:
    return f"{spread.mean:.4f}" if seeds == 1 else f"{spread.mean:.4f} ± {spread.std:.4f}"


def _report(
    summaries: Sequence[ArmSummary], class_names: Sequence[str], primary: Sequence[str]
) -> str:
    """The gate table as markdown, ready to paste into an experiment record."""
    header = ["arm", "mAP50", f"mAP50 ({len(primary)}-class)", "mAP50-95", *class_names]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for arm in summaries:
        cells = [
            f"{arm.arm} ({arm.val_pixels})",
            _cell(arm.map50, arm.seeds),
            f"**{_cell(arm.primary_map50, arm.seeds)}**",
            _cell(arm.map50_95, arm.seeds),
            *(
                _cell(arm.per_class_ap50[name], arm.seeds) if name in arm.per_class_ap50 else "--"
                for name in class_names
            ),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    by_arm = {arm.arm: arm for arm in summaries}
    if THERMAL_TRAINED in by_arm:
        ceiling, floor, trained = by_arm[CEILING], by_arm[FLOOR], by_arm[THERMAL_TRAINED]
        for label, minuend, subtrahend in (
            ("V - T (sensor gap)", ceiling, trained),
            ("T - J (domain gap)", trained, floor),
        ):
            gap = _per_class_gap(minuend, subtrahend)
            cells = [
                label,
                f"{minuend.map50.mean - subtrahend.map50.mean:+.4f}",
                f"**{minuend.primary_map50.mean - subtrahend.primary_map50.mean:+.4f}**",
                f"{minuend.map50_95.mean - subtrahend.map50_95.mean:+.4f}",
                *(f"{gap[name]:+.4f}" if name in gap else "--" for name in class_names),
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
        nargs="+",
        required=True,
        # The judge must be visible-trained and must never have supplied a training gradient
        # to anything this project produced -- invariant 7. The in-loop detector would be
        # grading its own homework, the contamination M1.2 step 1 went out of its way to rule out.
        help="the independent visible-trained judge, one checkpoint per seed, e.g. "
        "runs/reference-flir-yolo11s/weights/best.pt",
    )
    parser.add_argument(
        "--thermal-weights",
        type=Path,
        nargs="+",
        # Must match the judge's architecture and recipe, trained on this tree's thermal
        # manifest (<out>/manifests/thermal/data.yaml) -- otherwise V - T measures the recipe,
        # not the sensor.
        help="the thermal-trained detector (T), one checkpoint per seed; adds a third row and the "
        "V - T / T - J gaps. Omitted, the output is the two-row gate",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/gate"), help="output root")
    parser.add_argument(
        "--primary-classes",
        nargs="+",
        help="the classes the verdict is read off; defaults to every class in the manifest",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="the residual-misalignment constant the thermal labels were de-rolled by "
        "(scripts/mirror_thermal_labels.py --calibration). Not applied here: its digest is "
        "checked against the val thermal side's LABELS_PROVENANCE.json and then stamped on both "
        "rows, so a floor is traceable to the boxes it was actually measured against",
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
    for flag, paths in (("--weights", args.weights), ("--thermal-weights", args.thermal_weights)):
        missing = [str(path) for path in paths or () if not path.is_file()]
        if missing:
            raise SystemExit(f"{flag} not found: {missing}")

    out = Path(args.out).resolve()
    # Both manifests first, so everything checkable without a GPU is checked before the first
    # pass -- in particular an unmirrored infrared/labels, which is the failure that would
    # otherwise be scored rather than raised.
    manifests = out / "manifests"
    visible = write_budget_manifest(manifest, manifests / "visible", 1.0)
    thermal = write_budget_manifest(manifest, manifests / "thermal", 1.0, infrared=True)

    digest = _verified_digest(manifest, args.calibration)
    rows = [
        *(_score(args, w, visible, CEILING, "visible", primary, digest) for w in args.weights),
        *(_score(args, w, thermal, FLOOR, "thermal", primary, digest) for w in args.weights),
        *(
            _score(args, w, thermal, THERMAL_TRAINED, "thermal", primary, digest)
            for w in args.thermal_weights or ()
        ),
    ]
    _write_csv(out / CSV_FILENAME, rows, manifest.class_names)

    summaries = _summarise(rows)
    ceiling, floor = summaries[:2]
    headroom = ceiling.primary_map50.mean - floor.primary_map50.mean
    logger.info("gate table:\n%s", _report(summaries, manifest.class_names, primary))
    logger.info(
        "headroom on %d-class mAP50 (%s): %.4f - %.4f = %+.4f  [all-class: %+.4f]",
        len(primary),
        ", ".join(primary),
        ceiling.primary_map50.mean,
        floor.primary_map50.mean,
        headroom,
        ceiling.map50.mean - floor.map50.mean,
    )
    logger.info("%s", _verdict(headroom))
    if args.thermal_weights:
        _log_sensor_gap(ceiling, floor, summaries[2], primary, headroom)
    logger.info("written: %s", out / CSV_FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
