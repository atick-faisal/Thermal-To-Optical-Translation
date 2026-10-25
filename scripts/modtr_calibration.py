"""ModTr's calibration run: write its inner split, then read its verdict (modtr-baseline ADR-04).

Two subcommands, run on the server before and after the calibration run:

- ``split --data <data.yaml> --out <dir>`` copies the training pairs into a seeded 80/20 inner
  train/val split with its own ``data.yaml`` (`t2o.data.inner_split`). The real val split is
  not in it. ``--config`` splits the corpus that experiment trains on instead of every training
  image, which FLIR's 600-image cap needs (ADR-05).
- ``verdict --run runs/<name>`` reads the finished run's ``stage0/monitor.jsonl`` and logs both
  of the run's verdicts: ADR-04's epoch budget for the campaign, and ADR-03's loss rule, because
  the calibration run is also ModTr's smoke run. Exits 1 when the loss rule fails, since the
  campaign does not start then.

Between them, the run itself is an ordinary loop against the inner split::

    t2o loop --config experiments/modtr_custom.yaml --data <dir>/data.yaml \\
        --name modtr-calibration --seed 0 --monitor-map --no-detector

``verdict`` refuses a run whose manifest is not an inner split, or whose curve is shorter than
its ``epochs_per_stage``: a verdict read off a run still in flight, or one trained against the
reporting split, would be recorded as the calibration.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import yaml

from t2o.analysis.epoch_budget import LOSS_EPOCHS, epoch_budget, loss_falls
from t2o.config import CONFIG_FILENAME, Config
from t2o.data.inner_split import MARKER_KEY, write_inner_split
from t2o.data.manifest import DatasetManifest
from t2o.engine.trainer import load_monitor

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _split(args: argparse.Namespace) -> int:
    # Read from the experiment, never retyped: a cap or seed that differs from the campaign's
    # splits a different 600 images than the campaign trains on, and nothing would say so.
    data = Config.load(args.config).data if args.config else None
    write_inner_split(
        DatasetManifest.load(args.data),
        args.out,
        args.fraction,
        args.seed,
        max_images=data.max_train_images if data else None,
        subset_seed=data.subset_seed if data else 0,
    )
    return 0


def _verdict(args: argparse.Namespace) -> int:
    run = Path(args.run)
    config = Config.load(run / CONFIG_FILENAME)
    if len(config.coupling.task_weights) != 1:
        raise SystemExit(f"{run} has {len(config.coupling.task_weights)} stages; ModTr has one")
    declared = yaml.safe_load(Path(config.data.manifest).read_text()) or {}
    if MARKER_KEY not in declared:
        raise SystemExit(
            f"{run} trained on {config.data.manifest}, which is not an inner split; "
            "the calibration must never read the val split"
        )

    epochs = load_monitor(run / "stage0")
    expected = config.train.epochs_per_stage
    if len(epochs) != expected:
        raise SystemExit(f"{run} has {len(epochs)} of {expected} epochs; is it still running?")
    map50 = [e.val_map50 for e in epochs]
    if any(value is None for value in map50):
        raise SystemExit(f"{run} has epochs with no val_map50; was it run with --monitor-map?")

    budget = epoch_budget([100.0 * value for value in map50 if value is not None], expected)
    logger.info(
        "budget: %d epochs (%s) -- trailing mean peaks at %.2f points ending epoch %d, "
        "ends at %.2f",
        budget.budget,
        "trigger fired" if budget.fired else "trigger did not fire",
        budget.best_mean,
        budget.best_last_epoch,
        budget.final_mean,
    )

    loss_det = [e.train_losses["loss_det"] for e in epochs]
    passed = loss_falls(loss_det)
    logger.info(
        "loss: %s -- loss_det mean %.4f over the first %d epochs, %.4f over the last %d",
        "PASS" if passed else "FAIL, the campaign does not start",
        fmean(loss_det[:LOSS_EPOCHS]),
        LOSS_EPOCHS,
        fmean(loss_det[-LOSS_EPOCHS:]),
        LOSS_EPOCHS,
    )
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="write the inner 80/20 split of the training pairs")
    split.add_argument("--data", type=Path, required=True, help="the source data.yaml")
    split.add_argument("--out", type=Path, required=True, help="a new, empty directory")
    # ADR-04's 80/20, upstream's own inner split (`src/engine/config/config.py:27`).
    split.add_argument("--fraction", type=float, default=0.2, help="share held out as inner val")
    split.add_argument("--seed", type=int, default=0)
    split.add_argument(
        "--config",
        type=Path,
        help="the campaign's config; its data.max_train_images and data.subset_seed cut the "
        "training images before the split. Omit to split every training image",
    )
    split.set_defaults(handler=_split)

    verdict = commands.add_parser("verdict", help="read a finished calibration run")
    verdict.add_argument("--run", type=Path, required=True, help="the run directory")
    verdict.set_defaults(handler=_verdict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
