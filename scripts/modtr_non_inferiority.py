"""Is the loop non-inferior to ModTr? Margin's score (modtr-baseline ADR-02).

The margin gate (record 027) showed a CI over the val images cannot resolve +2 mAP@50, so
`goal.md`'s Margin became non-inferiority to ModTr. This script scores it: for each training
seed it scores the loop's export and ModTr's once each, then bootstraps the seed-averaged paired
delta over the whole val pool (`t2o.analysis.resolvability.seed_averaged_bootstrap`). The loop
is non-inferior when that CI's lower bound is above -δ, δ = 5 mAP points, pre-registered in
ADR-02. δ is deliberately not a flag.

**Seeds are paired by number**, under one `--seeds` list for both templates. ModTr's campaign
runs seeds 0, 1, 3, which both the pix2pix (`e3b`, seeds 0-5) and the turbo (`e3t`, seeds
0 1 3) loop campaigns ran. The templates default to the pix2pix loop; the turbo loop is a second
invocation into its own `--out`.

**One scorer, the one the gate used.** Every arm is scored by the zero-shot reference yolo11s
(margin-gate ADR-01). Both arms' exports carry the visible labels they were mirrored from
(`engine/export.py::_mirror_label`), and the pairing check refuses any image whose labels differ.

**Not resumable** (modtr-baseline plan, Q9). The interval needs every seed's records at once, so
CSV rows alone cannot resume it, and a rerun repays only a few val runs. The CSV is written once,
after the bootstrap, and an existing one is refused rather than overwritten.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig`, as
`scripts/margin_resolvability.py` does.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from t2o.analysis.resolvability import (
    NON_INFERIORITY_MARGIN,
    SeedAveragedInterval,
    non_inferior,
    seed_averaged_bootstrap,
)
from t2o.data.manifest import DatasetManifest
from t2o.metrics import task
from t2o.metrics.task import ImageStats, map50_from_stats

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

CSV_FILENAME = "modtr-non-inferiority.csv"
MEAN = "mean"  # the `seed` of the seed-averaged row, the one that carries the CI and verdict

# The reference detector's own settings in every E3 config (`detector.reference`).
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16


@dataclass(frozen=True, slots=True)
class Row:
    """One seed's pool delta, or (`seed == MEAN`) the seed-averaged delta and its CI."""

    seed: str
    pool: int  # images both arms scored; every draw is this many, with replacement
    # Fractions over the pool, like every other CSV's map50; on the MEAN row, the seeds' mean.
    loop_map50: float
    modtr_map50: float
    delta: float  # loop - ModTr, mAP points
    # The CI is over the seed-averaged delta, so these four are empty on a per-seed row.
    low: float | None  # mAP points; the number the verdict reads
    high: float | None  # mAP points
    half_width: float | None  # mAP points
    seed_sd: float | None  # sd of the per-seed deltas, mAP points
    margin: float  # δ, mAP points (ADR-02)
    non_inferior: bool | None  # empty on a per-seed row
    resamples: int
    bootstrap_seed: int
    loop_data: str  # the export, or on the MEAN row the template
    modtr_data: str
    weights: str


def _export_manifest(template: str, seed: int) -> DatasetManifest:
    """One seed's export. A template, so the seed-to-export pairing is structural."""
    return DatasetManifest.load(Path(template.format(seed=seed)) / "data.yaml")


def _collect(args: argparse.Namespace, data_yaml: Path) -> dict[str, ImageStats]:
    # Looked up on the module at call time, so the tests can stand a stub in for the GPU.
    return task.collect_image_stats(
        weights=args.reference_weights,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


def _rows(
    args: argparse.Namespace,
    loops: dict[int, DatasetManifest],
    modtrs: dict[int, DatasetManifest],
    loop: dict[int, dict[str, ImageStats]],
    modtr: dict[int, dict[str, ImageStats]],
    interval: SeedAveragedInterval,
) -> list[Row]:
    """One row per seed, then the MEAN row, which keeps the run's constants from the first."""
    rows = [
        Row(
            seed=str(seed),
            pool=len(loop[seed]),
            loop_map50=map50_from_stats(list(loop[seed].values())),
            modtr_map50=map50_from_stats(list(modtr[seed].values())),
            delta=interval.seed_deltas[seed],
            low=None,
            high=None,
            half_width=None,
            seed_sd=None,
            margin=NON_INFERIORITY_MARGIN,
            non_inferior=None,
            resamples=args.resamples,
            bootstrap_seed=args.bootstrap_seed,
            loop_data=str(loops[seed].path),
            modtr_data=str(modtrs[seed].path),
            weights=str(args.reference_weights),
        )
        for seed in sorted(loop)
    ]
    mean = replace(
        rows[0],
        seed=MEAN,
        loop_map50=sum(row.loop_map50 for row in rows) / len(rows),
        modtr_map50=sum(row.modtr_map50 for row in rows) / len(rows),
        delta=interval.delta,
        low=interval.low,
        high=interval.high,
        half_width=interval.half_width,
        seed_sd=interval.seed_sd,
        non_inferior=non_inferior(interval.low),
        loop_data=args.loop_template,
        modtr_data=args.modtr_template,
    )
    return [*rows, mean]


def _write(csv_path: Path, rows: list[Row]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Row)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _log_verdict(interval: SeedAveragedInterval) -> None:
    logger.info(
        "per-seed deltas (loop - ModTr): %s; sd %.2f mAP points",
        ", ".join(f"s{seed} {delta:+.2f}" for seed, delta in interval.seed_deltas.items()),
        interval.seed_sd,
    )
    if non_inferior(interval.low):
        logger.info(
            "verdict: NON-INFERIOR -- lower bound %.2f > -%.1f mAP points (delta %+.2f)",
            interval.low,
            NON_INFERIORITY_MARGIN,
            interval.delta,
        )
    else:
        logger.info(
            "verdict: NOT NON-INFERIOR -- lower bound %.2f <= -%.1f mAP points (delta %+.2f)",
            interval.low,
            NON_INFERIORITY_MARGIN,
            interval.delta,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/modtr-non-inferiority"), help="output root"
    )
    # ModTr's campaign seeds: the ones both loop campaigns ran (plan, "Scoring Margin").
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 3])
    parser.add_argument(
        "--loop-template",
        default="runs/e3b-loop-s{seed}/stage3/translated",
        help="the loop's final export per seed; '{seed}' is substituted",
    )
    parser.add_argument(
        "--modtr-template",
        required=True,
        # Not defaulted: MODTR-BASELINE-06's configs name the runs, and a guessed path would
        # fail only after the loop's val runs had been paid for.
        help="ModTr's final export per seed; '{seed}' is substituted",
    )
    parser.add_argument(
        "--reference-weights",
        type=Path,
        default=Path("runs/reference-yolo11s/weights/best.pt"),
        # margin-gate ADR-01: the scorer the gate used and Margin is defined on.
        help="the zero-shot reference judge that scores every arm",
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default=None, help="'cuda:0', 'cpu', or omit for auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # Everything checkable without a GPU is checked before the first val run.
    csv_path = Path(args.out).resolve() / CSV_FILENAME
    if csv_path.is_file():
        raise SystemExit(f"{csv_path} already holds a result; pass another --out")
    if not Path(args.reference_weights).is_file():
        raise SystemExit(f"--reference-weights not found: {args.reference_weights}")
    loops = {seed: _export_manifest(args.loop_template, seed) for seed in args.seeds}
    modtrs = {seed: _export_manifest(args.modtr_template, seed) for seed in args.seeds}

    loop: dict[int, dict[str, ImageStats]] = {}
    modtr: dict[int, dict[str, ImageStats]] = {}
    for seed in args.seeds:
        logger.info("seed %d: scoring the loop and ModTr", seed)
        loop[seed] = _collect(args, loops[seed].path)
        modtr[seed] = _collect(args, modtrs[seed].path)

    interval = seed_averaged_bootstrap(
        loop, modtr, resamples=args.resamples, seed=args.bootstrap_seed
    )
    _write(csv_path, _rows(args, loops, modtrs, loop, modtr, interval))
    logger.info("wrote %s", csv_path)
    _log_verdict(interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
