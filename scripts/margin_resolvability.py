"""Can ~100 test images resolve +2 mAP@50? The margin gate's server run (margin-gate plan).

`goal.md`'s Margin criterion wants the loop ahead of every frozen-detector baseline by >= +2
mAP@50, with a 95% bootstrap CI over the ~100 held-out test images. If the validation split
shows that a CI at that size cannot resolve +2, Margin becomes non-inferiority to ModTr. This
script reads that off the 153 val images before the test split is ever touched: for each E3
seed it scores three arms once each, then bootstraps the paired delta at the test split's size
(`t2o.analysis.resolvability`).

Two pairs per seed, gating on the wider (plan Q4): the loop against its own no-loop control,
and the loop against raw thermal. The real baselines do not exist yet, so neither pair is a
perfect proxy for loop vs ModTr. The verdict is the widest half-width across every row in the
CSV, read after the loop, so a resumed run judges every seed and not only the ones it just
computed. "Resolves" means that half-width is <= 2 mAP points (plan Q3).

**One scorer, the one Margin will use.** Every arm is scored by the zero-shot reference
yolo11s (ADR-01), so the gate and the final number cannot measure two detectors' spreads.

**Raw thermal is built as `gate_table.py` and E8's arm A0 build it** --
`write_budget_manifest(..., 1.0, infrared=True)`, which refuses a thermal side with no labels
before a GPU starts. It is scored against the thermal-side labels, as every raw-thermal floor
in this repo has been, while an export carries the visible labels it was mirrored from
(`engine/export.py::_mirror_label`). The pairing check compares each image's target classes,
so a transferred label file that lost or gained a box stops the run on that pair rather than
being bootstrapped. It does not depend on the training seed, so it is collected once.

**One campaign per invocation.** The templates default to the pix2pix campaign (`e3b`, seeds
0-5). The turbo campaign (`e3t`, seeds 0 1 3) is a second invocation into its own `--out`, and
the gate reads the wider of the two verdicts.

**Resuming.** Each (seed, pair) row is appended as it completes, and a re-run skips the rows
already in the CSV. A val run is minutes and a bootstrap less, so a resumed seed re-pays its
val runs rather than caching records on disk.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig`, as
`scripts/annotation_sweep.py` does.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from t2o.analysis.resolvability import paired_image_bootstrap, resolves
from t2o.data.budget import write_budget_manifest
from t2o.data.manifest import DatasetManifest
from t2o.metrics import task
from t2o.metrics.task import ImageStats, map50_from_stats

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

CSV_FILENAME = "margin-resolvability.csv"
VS_CONTROL = "loop-vs-control"
VS_THERMAL = "loop-vs-thermal"
PAIRS = (VS_CONTROL, VS_THERMAL)

# The reference detector's own settings in every E3 config (`detector.reference`).
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16


@dataclass(frozen=True, slots=True)
class Row:
    """One seed's bootstrap of one pair. `seed`/`pair` identify it for resuming."""

    seed: int
    pair: str
    n: int  # images per draw: the test split's size, not the pool's
    pool: int  # images both arms scored, drawn from with replacement
    resamples: int
    bootstrap_seed: int
    # Over the whole pool, as fractions like every other CSV's map50; the CI below is in points.
    loop_map50: float
    baseline_map50: float
    low: float  # mAP points
    high: float  # mAP points
    half_width: float  # mAP points; the number the gate reads
    resolves: bool
    loop_data: str
    baseline_data: str
    weights: str


def _export_manifest(template: str, seed: int) -> DatasetManifest:
    """One seed's stage-3 export. A template, so the seed-to-export pairing is structural."""
    return DatasetManifest.load(Path(template.format(seed=seed)) / "data.yaml")


def _completed(csv_path: Path) -> set[tuple[int, str]]:
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline="") as handle:
        return {(int(row["seed"]), row["pair"]) for row in csv.DictReader(handle)}


def _append(csv_path: Path, row: Row) -> None:
    """Write one row and close the file, so an interrupted run keeps what it has paid for."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.is_file()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Row)])
        if new:
            writer.writeheader()
        writer.writerow(asdict(row))


def _collect(args: argparse.Namespace, data_yaml: Path) -> dict[str, ImageStats]:
    # Looked up on the module at call time, so the tests can stand a stub in for the GPU.
    return task.collect_image_stats(
        weights=args.reference_weights,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


def _row(
    args: argparse.Namespace,
    seed: int,
    pair: str,
    loop: dict[str, ImageStats],
    baseline: dict[str, ImageStats],
    loop_data: Path,
    baseline_data: Path,
) -> Row:
    interval = paired_image_bootstrap(
        loop, baseline, n=args.n, resamples=args.resamples, seed=args.bootstrap_seed
    )
    return Row(
        seed=seed,
        pair=pair,
        n=args.n,
        pool=len(loop),
        resamples=args.resamples,
        bootstrap_seed=args.bootstrap_seed,
        loop_map50=map50_from_stats(list(loop.values())),
        baseline_map50=map50_from_stats(list(baseline.values())),
        low=interval.low,
        high=interval.high,
        half_width=interval.half_width,
        resolves=resolves(interval.half_width),
        loop_data=str(loop_data),
        baseline_data=str(baseline_data),
        weights=str(args.reference_weights),
    )


def _log_verdict(csv_path: Path) -> None:
    """The gate: the widest half-width over every row in the CSV, this run's or an earlier one's."""
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    widest = max(rows, key=lambda row: float(row["half_width"]))
    half_width = float(widest["half_width"])
    logger.info(
        "verdict over %d row(s): widest half-width %.2f mAP points (seed %s, %s) at n=%s",
        len(rows),
        half_width,
        widest["seed"],
        widest["pair"],
        widest["n"],
    )
    if resolves(half_width):
        logger.info("verdict: RESOLVES -- a CI at the test split's size can resolve +2 mAP@50")
    else:
        logger.info(
            "verdict: DOES NOT RESOLVE -- +2 mAP@50 cannot be told from zero at n=%s; "
            "Margin becomes non-inferiority to ModTr (goal.md)",
            widest["n"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        # Not defaulted: on the server the paired dataset lives outside the repo.
        help="the paired dataset's data.yaml -- the raw-thermal arm's val split comes from here",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/margin-resolvability"), help="output root"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument(
        "--loop-template",
        default="runs/e3b-loop-s{seed}/stage3/translated",
        help="lambda>0 export per seed; '{seed}' is substituted",
    )
    parser.add_argument(
        "--control-template",
        default="runs/e3b-control-s{seed}/stage3/translated",
        help="lambda=0 export per seed; '{seed}' is substituted",
    )
    parser.add_argument(
        "--reference-weights",
        type=Path,
        default=Path("runs/reference-yolo11s/weights/best.pt"),
        # ADR-01: the scorer Margin itself will use. Another detector here would gate on one
        # detector's spread and claim on another's.
        help="the zero-shot reference judge that scores every arm",
    )
    parser.add_argument(
        "--n", type=int, default=100, help="images per draw: the held-out test split's size"
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
    if not Path(args.reference_weights).is_file():
        raise SystemExit(f"--reference-weights not found: {args.reference_weights}")
    loops = {seed: _export_manifest(args.loop_template, seed) for seed in args.seeds}
    controls = {seed: _export_manifest(args.control_template, seed) for seed in args.seeds}
    out = Path(args.out).resolve()
    source = DatasetManifest.load(args.data)
    thermal = write_budget_manifest(source, out / "manifests" / "thermal", 1.0, infrared=True)

    csv_path = out / CSV_FILENAME
    done = _completed(csv_path)
    pending = {seed: [p for p in PAIRS if (seed, p) not in done] for seed in args.seeds}
    logger.info(
        "%d row(s): %d already in %s, %d to run",
        len(args.seeds) * len(PAIRS),
        len(done & {(seed, p) for seed in args.seeds for p in PAIRS}),
        csv_path,
        sum(len(pairs) for pairs in pending.values()),
    )

    thermal_records: dict[str, ImageStats] | None = None
    for seed, pairs in pending.items():
        if not pairs:
            continue
        logger.info("seed %d: %s", seed, ", ".join(pairs))
        loop = _collect(args, loops[seed].path)
        for pair in pairs:
            if pair == VS_CONTROL:
                baseline_data = controls[seed].path
                baseline = _collect(args, baseline_data)
            else:
                baseline_data = thermal
                if thermal_records is None:
                    thermal_records = _collect(args, thermal)
                baseline = thermal_records
            row = _row(args, seed, pair, loop, baseline, loops[seed].path, baseline_data)
            _append(csv_path, row)
            logger.info(
                "seed %d %s: delta CI [%.2f, %.2f], half-width %.2f mAP points",
                seed,
                pair,
                row.low,
                row.high,
                row.half_width,
            )

    _log_verdict(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
