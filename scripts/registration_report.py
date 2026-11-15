"""C2's registration-error axis, read: mAP@50 against misalignment px, both halves, one dataset.

The label-transfer half is already a table: ``scripts/registration_sweep.py`` wrote one CSV row
per (level, seed). The translation half is not. ``metrics.json`` keeps one scalar per stage, and
at one seed per level (plan Q4) a scalar says nothing about whether a level's drop is real
(record 037, F229). So each translation run's stage-3 export is re-scored per image with
``collect_image_stats``, as ``scripts/margin_resolvability.py`` does, and every point gets:

* an **image-bootstrap CI** of its own mAP@50, drawn n-out-of-n over val;
* a **paired CI of its delta to the 0 px run**, both runs scored on the same draw
  (``paired_image_bootstrap`` at n = pool). This is the interval that decides "a drop".

Both translation arms are reported (plan, "The degradation report"): **zero-shot**, the frozen
visible judge on the translated images, which is the translation route's headline; and
**adapted**, the detector fine-tuned on them, which F230 found tracks label transfer. Each run's
judge, image size and batch come from its own config snapshot, so the custom set and FLIR are
each scored by the judge their runs recorded.

**A re-score that misses ``metrics.json`` stops the script.** The zero-shot re-score is the same
judge on the same export through the same ``model.val()``, so it must reproduce the recorded
``zero_shot.map50``. A gap means a different export, judge or stale ``labels.cache``. Only the
zero-shot arm can be checked: the adapted arm's recorded value was written by ultralytics at the
end of training, not by ``evaluate_detector``.

**Image CIs carry no seed noise.** They say which val images were scored, not which training
seed was drawn. ``--seed-runs`` names the aligned seed set (the custom set's ``e3b-loop-s*``),
and its spread is printed beside the 0 px point as the yardstick.

**What the report says without being asked**, because the numbers alone would mislead:

* label-transfer seeds that came out identical are n = 1 in effect, with no seed band (F227);
* a stage-3 FID above ``--collapse-fid`` marks a collapsed translator (F231). One seed with
  collapses draws points, not a curve (the FLIR decision of 2026-11-15);
* neither 0 px is truly zero.

**FLIR's primary classes** (``--primary-classes bicycle car person``) apply to both halves: the
sweep's ``primary_map50`` column, and the translation records filtered with
``restrict_classes`` before every mAP.

Not resumable: a dataset is about ten val passes, and the bootstraps need every run's records at
once. The CSV is written once, at the end, and an existing one is refused rather than
overwritten. Standalone script, not part of the ``t2o`` package; it owns its own
``logging.basicConfig``, as ``scripts/margin_resolvability.py`` does.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from t2o.analysis.aggregate import AggregationError, RunRecord, load_run, metric_value
from t2o.analysis.resolvability import (
    UnpairedImagesError,
    image_bootstrap,
    paired_image_bootstrap,
)
from t2o.cli import _expand_run_globs
from t2o.config.base import CONFIG_FILENAME
from t2o.data.manifest import DatasetManifest
from t2o.metrics import task
from t2o.metrics.task import ImageStats, map50_from_stats, primary_mean, restrict_classes

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

LABEL_TRANSFER = "label_transfer"
ZERO_SHOT = "zero_shot"
ADAPTED = "adapted"
TRANSLATION_ARMS = (ZERO_SHOT, ADAPTED)

# F231: FLIR's collapsed stages read FID 211.81-350.88 and its healthy ones 93.13-163.96; the
# custom set never passed 148.96. 200 sits in the gap.
DEFAULT_COLLAPSE_FID = 200.0
# The re-score repeats the recorded val pass exactly, so it agrees to float noise. A gap this
# size is already a different export, judge or label set, never GPU nondeterminism.
RESCORE_TOLERANCE = 5e-4
# Levels are joined across the two halves at this many decimals. The sweep stamps the level it
# measured off the field (e.g. 12.000000001); a config states the nominal 12.0.
LEVEL_DECIMALS = 2

DISCLOSURE = """\
Neither 0 px is truly zero. The custom set's thermal boxes were moved from visible by the sister
repo's registration, so every level sits on that error. FLIR keeps about 2.25 px of median
per-pair scatter after calibration (record 012)."""


@dataclass(frozen=True, slots=True)
class Row:
    """One point of one route's curve. mAP and deltas are fractions; CI columns are mAP points."""

    dataset: str
    route: str  # label_transfer, zero_shot or adapted
    level_px: float
    n_seeds: int  # seeds behind `map50`: the sweep's seeds, or 1 for a translation run
    map50: float  # over --primary-classes when given
    seed_sd: float | None  # sample sd over the seeds; None at n = 1 or identical seeds
    seeds_identical: bool  # n > 1 seeds that all gave one value: n = 1 in effect (F227)
    # Translation only: the aligned seed set's sd at 0 px, from --seed-runs, and its size.
    seed_spread_sd: float | None
    seed_spread_n: int
    ci_low_pts: float | None  # image bootstrap of `map50`; translation only
    ci_high_pts: float | None
    delta: float  # `map50` minus this route's 0 px point
    delta_low_pts: float | None  # paired image bootstrap of `delta`; translation, level > 0
    delta_high_pts: float | None
    fid: float | None  # the export's stage FID; translation only
    collapsed: bool  # fid above --collapse-fid (F231)
    source: str  # the run directory, or the sweep CSV


@dataclass(frozen=True, slots=True)
class Scorer:
    """One detector pass over an export: which weights, at which settings."""

    weights: Path
    imgsz: int
    batch: int


@dataclass(frozen=True, slots=True)
class TranslationRun:
    """One translation run at one level, with everything its two val passes need."""

    record: RunRecord
    level_px: float
    stage: dict[str, Any]  # the scored stage's raw `metrics.json` entry
    export: Path  # the stage's translated export, data.yaml
    scorers: dict[str, Scorer]  # keyed by translation arm


def _level(value: float) -> float:
    return round(float(value), LEVEL_DECIMALS)


def _translation_run(path: Path, stage: int) -> TranslationRun:
    """Read one run's level, stage and scorers, refusing anything a val pass would trip on."""
    try:
        record = load_run(path)
    except AggregationError as error:
        raise SystemExit(str(error)) from None
    entry = next((s for s in record.stages if int(s["stage"]) == stage), None)
    if entry is None:
        raise SystemExit(f"{path} has no stage {stage}; stages {list(record.stage_indices)}")

    # Plain YAML, as `load_run` reads it: e3b's August snapshot predates fields Config.load
    # would now demand or reject. A snapshot with no misalignment field is an aligned run.
    snapshot = yaml.safe_load((path / CONFIG_FILENAME).read_text())
    detector = snapshot["detector"]
    reference = detector["reference"]
    evaluation = detector["evaluation"]
    if entry.get("detector") is None:
        raise SystemExit(f"{path} stage {stage} recorded no adapted detector (--no-detector?)")
    scorers = {
        # `engine/loop.py::_resolve_reference_weights`: the reference, else the eval init.
        ZERO_SHOT: Scorer(
            Path(reference.get("weights") or evaluation["init_weights"]),
            int(reference["imgsz"]),
            int(reference["batch"]),
        ),
        ADAPTED: Scorer(
            Path(entry["detector"]["weights"]), int(evaluation["imgsz"]), int(evaluation["batch"])
        ),
    }
    for arm, scorer in scorers.items():
        if not scorer.weights.is_file():
            raise SystemExit(f"{path}: the {arm} weights are not on disk: {scorer.weights}")
    export = path / f"stage{stage}" / "translated" / "data.yaml"
    if not export.is_file():
        raise SystemExit(f"{path}: no stage-{stage} export at {export}")
    return TranslationRun(
        record=record,
        level_px=_level(snapshot.get("data", {}).get("misalignment_px") or 0.0),
        stage=entry,
        export=export,
        scorers=scorers,
    )


def _by_level(runs: Sequence[TranslationRun]) -> dict[float, TranslationRun]:
    """One run per level, 0 px included, or the curve has no single point to stand on."""
    by_level: dict[float, TranslationRun] = {}
    for run in runs:
        if run.level_px in by_level:
            raise SystemExit(
                f"two runs at {run.level_px:g} px: {by_level[run.level_px].record.path} and "
                f"{run.record.path}. Pass the aligned seed set with --seed-runs, not --runs"
            )
        by_level[run.level_px] = run
    if 0.0 not in by_level:
        raise SystemExit(f"no 0 px run among --runs (levels {sorted(by_level)})")
    return dict(sorted(by_level.items()))


def _class_ids(classes: Sequence[str], exports: Sequence[Path]) -> list[int]:
    """`classes` as indices, refused unless every export names them at the same index."""
    orders: set[tuple[int, ...]] = set()
    for export in exports:
        names = DatasetManifest.load(export).class_names
        if not set(classes) <= set(names):
            raise SystemExit(f"--primary-classes {list(classes)} not all in {export}: {names}")
        orders.add(tuple(names.index(name) for name in classes))
    if len(orders) > 1:
        raise SystemExit(f"--primary-classes sit at different indices across exports: {orders}")
    return list(orders.pop())


def _collect(args: argparse.Namespace, scorer: Scorer, export: Path) -> dict[str, ImageStats]:
    # Looked up on the module at call time, so the tests can stand a stub in for the GPU.
    return task.collect_image_stats(
        weights=scorer.weights,
        data_yaml=export,
        imgsz=scorer.imgsz,
        batch=scorer.batch,
        device=args.device,
    )


def _check_rescore(run: TranslationRun, records: Mapping[str, ImageStats]) -> None:
    """Stop unless the zero-shot re-score reproduces the run's recorded value (module doc)."""
    recorded = metric_value(run.stage, "zero_shot.map50")
    rescored = map50_from_stats(list(records.values()))
    if abs(rescored - recorded) > RESCORE_TOLERANCE:
        raise SystemExit(
            f"{run.record.path}: the zero-shot re-score reads {rescored:.4f} mAP50, but "
            f"metrics.json recorded {recorded:.4f}. Check the export, the judge in its config "
            "snapshot, and delete the export's labels.cache"
        )


def _seed_spread(
    paths: Sequence[Path], stage: int, primary: Sequence[str] | None
) -> dict[str, tuple[float | None, int]]:
    """Each translation arm's recorded sd over the aligned seed set, read off `metrics.json`."""
    values: dict[str, list[float]] = {arm: [] for arm in TRANSLATION_ARMS}
    for path in paths:
        run = _translation_run(path, stage)
        if run.level_px != 0.0:
            raise SystemExit(f"--seed-runs must all be 0 px; {path} is {run.level_px:g} px")
        zero_shot = run.stage["zero_shot"]
        values[ZERO_SHOT].append(
            primary_mean(zero_shot["per_class_ap50"], primary)
            if primary
            else float(zero_shot["map50"])
        )
        if not primary:
            values[ADAPTED].append(float(run.stage["detector"]["map50"]))
    if primary:
        logger.info("adapted arm: no seed spread over --primary-classes (no per-class AP recorded)")
    return {arm: (_sd(found), len(found)) for arm, found in values.items()}


def _sd(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _label_transfer_rows(csv_path: Path, dataset: str, primary: Sequence[str] | None) -> list[Row]:
    """The sweep's rows for `dataset`, one per level: seed mean, sd and delta to 0 px."""
    column = "primary_map50" if primary else "map50"
    with csv_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset]
    if not rows:
        raise SystemExit(f"{csv_path} has no rows for --dataset {dataset!r}")
    if any(row[column] == "" for row in rows):
        raise SystemExit(
            f"{csv_path} has blank {column} cells for {dataset}; the sweep ran without "
            "--primary-classes, so this half can only be read all-class"
        )

    by_level: dict[float, list[float]] = {}
    for row in rows:
        by_level.setdefault(_level(float(row["misalignment_px"])), []).append(float(row[column]))
    if 0.0 not in by_level:
        raise SystemExit(f"{csv_path} has no 0 px rows for {dataset}")
    zero = statistics.mean(by_level[0.0])

    result: list[Row] = []
    for level, values in sorted(by_level.items()):
        identical = len(values) > 1 and len(set(values)) == 1
        result.append(
            Row(
                dataset=dataset,
                route=LABEL_TRANSFER,
                level_px=level,
                n_seeds=len(values),
                map50=statistics.mean(values),
                seed_sd=None if identical else _sd(values),
                seeds_identical=identical,
                seed_spread_sd=None,
                seed_spread_n=0,
                ci_low_pts=None,
                ci_high_pts=None,
                delta=statistics.mean(values) - zero,
                delta_low_pts=None,
                delta_high_pts=None,
                fid=None,
                collapsed=False,
                source=str(csv_path),
            )
        )
    return result


def _translation_rows(
    args: argparse.Namespace,
    runs: Mapping[float, TranslationRun],
    ids: Sequence[int] | None,
    spread: Mapping[str, tuple[float | None, int]],
) -> list[Row]:
    """Both arms of every run, scored per image, each against its own 0 px point."""
    records: dict[str, dict[float, dict[str, ImageStats]]] = {arm: {} for arm in TRANSLATION_ARMS}
    for level, run in runs.items():
        for arm in TRANSLATION_ARMS:
            logger.info("%g px %s: scoring %s", level, arm, run.export)
            found = _collect(args, run.scorers[arm], run.export)
            if arm == ZERO_SHOT:
                _check_rescore(run, found)
            records[arm][level] = restrict_classes(found, ids) if ids is not None else found

    result: list[Row] = []
    for arm in TRANSLATION_ARMS:
        zero = records[arm][0.0]
        zero_map50 = map50_from_stats(list(zero.values()))
        for level, run in runs.items():
            scored = records[arm][level]
            map50 = map50_from_stats(list(scored.values()))
            low, high = image_bootstrap(scored, resamples=args.resamples, seed=args.bootstrap_seed)
            delta_low = delta_high = None
            if level != 0.0:
                try:
                    interval = paired_image_bootstrap(
                        scored,
                        zero,
                        n=len(scored),
                        resamples=args.resamples,
                        seed=args.bootstrap_seed,
                    )
                except UnpairedImagesError as error:
                    raise SystemExit(f"{run.record.path} against the 0 px run: {error}") from None
                delta_low, delta_high = interval.low, interval.high
            fid = metric_value(run.stage, "fidelity.fid")
            sd, n = spread.get(arm, (None, 0)) if level == 0.0 else (None, 0)
            result.append(
                Row(
                    dataset=args.dataset,
                    route=arm,
                    level_px=level,
                    n_seeds=1,
                    map50=map50,
                    seed_sd=None,
                    seeds_identical=False,
                    seed_spread_sd=sd,
                    seed_spread_n=n,
                    ci_low_pts=low,
                    ci_high_pts=high,
                    delta=map50 - zero_map50,
                    delta_low_pts=delta_low,
                    delta_high_pts=delta_high,
                    fid=fid,
                    collapsed=fid > args.collapse_fid,
                    source=str(run.record.path),
                )
            )
    return result


def _log_report(rows: Sequence[Row], primary: Sequence[str] | None, collapse_fid: float) -> None:
    metric = f"mAP50 over {' '.join(primary)}" if primary else "mAP50, all classes"
    logger.info("=== %s: %s against misalignment px ===", rows[0].dataset, metric)
    for row in rows:
        if row.route == LABEL_TRANSFER:
            spread = (
                "seeds identical"
                if row.seeds_identical
                else f"sd {row.seed_sd:.4f}"
                if row.seed_sd is not None
                else "sd n/a"
            )
            logger.info(
                "%-14s %5g px  %.4f  %s (n=%d)  delta %+.4f",
                row.route,
                row.level_px,
                row.map50,
                spread,
                row.n_seeds,
                row.delta,
            )
            continue
        delta = (
            f"delta {row.delta:+.4f} [{row.delta_low_pts:+.2f}, {row.delta_high_pts:+.2f}] pts"
            if row.delta_low_pts is not None and row.delta_high_pts is not None
            else "0 px point"
        )
        logger.info(
            "%-14s %5g px  %.4f  CI [%.2f, %.2f] pts  %s  FID %.2f%s",
            row.route,
            row.level_px,
            row.map50,
            row.ci_low_pts,
            row.ci_high_pts,
            delta,
            row.fid,
            f"  COLLAPSED (FID > {collapse_fid:g})" if row.collapsed else "",
        )
        if row.seed_spread_sd is not None:
            logger.info(
                "%-14s          aligned seed set: sd %.4f over %d seeds (not in the CI above)",
                "",
                row.seed_spread_sd,
                row.seed_spread_n,
            )

    identical = [row for row in rows if row.seeds_identical]
    if identical:
        logger.info(
            "label transfer: the seeds gave identical results at %d of %d levels, so it is "
            "n = 1 in effect there, with no seed band (F227)",
            len(identical),
            sum(1 for row in rows if row.route == LABEL_TRANSFER),
        )
    collapsed = {row.level_px for row in rows if row.collapsed}
    if collapsed:
        logger.info(
            "translation: the translator collapsed at %s px (stage FID > %g, F231). One seed with "
            "collapses draws points, not a curve; a delta to a collapsed 0 px is not a drop",
            ", ".join(f"{level:g}" for level in sorted(collapsed)),
            collapse_fid,
        )
    logger.info(
        "translation CIs are over val images at one training seed; they carry no seed noise"
    )
    for line in DISCLOSURE.splitlines():
        logger.info("%s", line)


def _write(csv_path: Path, rows: Sequence[Row]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Row)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        # The sweep CSV's `dataset` column, and the name stamped on every output row.
        help="the dataset to report, e.g. custom or flir",
    )
    parser.add_argument(
        "--sweep-csv", required=True, type=Path, help="scripts/registration_sweep.py's CSV"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        # Globs are expanded here: PowerShell passes them through unexpanded.
        help="one translation run per level, 0 px included (custom: e3b-loop-s0)",
    )
    parser.add_argument(
        "--seed-runs",
        nargs="+",
        # Read off metrics.json only, never re-scored: they give the yardstick, not a point.
        help="the aligned seed set whose spread is printed beside 0 px (custom: e3b-loop-s*)",
    )
    parser.add_argument(
        "--primary-classes",
        nargs="+",
        # FLIR: bicycle car person (modtr-baseline ADR-06). Omitted, both halves are all-class.
        help="classes both halves are read over; omitted, every class",
    )
    parser.add_argument("--stage", type=int, default=3, help="the loop stage to score")
    parser.add_argument(
        "--collapse-fid",
        type=float,
        default=DEFAULT_COLLAPSE_FID,
        help="a stage FID above this marks a collapsed translator (F231)",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/registration-report"))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="'cuda:0', 'cpu', or omit for auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # Everything checkable without a GPU is checked before the first val pass.
    csv_path = Path(args.out).resolve() / f"{args.dataset}-registration-curve.csv"
    if csv_path.exists():
        raise SystemExit(f"{csv_path} already exists; move it aside to report again")
    label_transfer = _label_transfer_rows(args.sweep_csv, args.dataset, args.primary_classes)
    try:
        run_dirs = _expand_run_globs(args.runs)
        seed_dirs = _expand_run_globs(args.seed_runs) if args.seed_runs else []
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    runs = _by_level([_translation_run(path, args.stage) for path in run_dirs])
    ids = (
        _class_ids(args.primary_classes, [run.export for run in runs.values()])
        if args.primary_classes
        else None
    )
    spread = _seed_spread(seed_dirs, args.stage, args.primary_classes) if seed_dirs else {}
    logger.info(
        "%s: %d label-transfer level(s), %d translation run(s) at %s px, %d resamples",
        args.dataset,
        len(label_transfer),
        len(runs),
        ", ".join(f"{level:g}" for level in runs),
        args.resamples,
    )

    rows = [*label_transfer, *_translation_rows(args, runs, ids, spread)]
    _log_report(rows, args.primary_classes, args.collapse_fid)
    _write(csv_path, rows)
    logger.info("wrote %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
