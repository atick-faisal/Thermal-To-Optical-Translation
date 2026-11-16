"""Re-score detectors on val with the clamp's loosest boxes removed, and apply plan Q3's rule.

``scripts/clipped_box_drop_tree.py`` builds copies of the val tree without the boxes whose
retained-area fraction is below a threshold. This script scores each named checkpoint on its
original val tree and on each of those copies, and reports what moved
(``docs/features/clipped-box-audit/plan.md``, step 3).

**One evaluation path.** Every mAP is ``t2o.metrics.task.evaluate_detector``, ultralytics' own
``model.val()`` (invariant 1). The drop biases mAP down through false positives on the objects
whose boxes are gone, where the loose labels biased it down through missed matches (plan Q2).
To keep that second bias visible, each tree is also scored per image with
``collect_image_stats`` -- the same ``model.val()`` call -- and ``lost_matches`` is the fall in
true positives at IoU 0.5 from the original tree to the drop tree. Each such prediction matched
a box the drop removed, so it now counts as a false positive. The count is near-exact rather
than exact: removing a box can, rarely, let a prediction re-match another box.

**The decision rule is fixed here, before any server score** (plan Q3). A threshold says
``fix`` when, on mAP50, any arm's rank changes or any arm's delta is larger than the seed
spread passed in for it. Otherwise it says ``disclose``.

**Every arm names its own trees.** Thermal arms score on a drop copy of the custom val tree;
translation arms score on their own translated export, so they need a drop copy of that export.
``--arm`` therefore takes the original ``data.yaml`` and one ``THRESHOLD=data.yaml`` per drop tree.

Standalone script, not part of the ``t2o`` package -- it owns its own ``logging.basicConfig``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from t2o.metrics import task

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
ORIGINAL = "original"  # the rank table's key for the untouched val tree
MIN_ARM_FIELDS = 5  # name, weights, spread, original data.yaml, at least one drop tree


class RescoreError(ValueError):
    """Raised for arm specs the re-score cannot run as given."""


@dataclass(frozen=True, slots=True)
class Arm:
    """One checkpoint and the val trees it is scored on."""

    name: str
    weights: Path
    spread: float  # the arm's reported seed spread in mAP50; the Q3 noise floor
    original: Path
    drops: dict[float, Path]  # threshold -> drop tree data.yaml


@dataclass(frozen=True, slots=True)
class Score:
    """One checkpoint on one tree."""

    map50: float
    map50_95: float
    true_positives: int  # predictions matched at IoU 0.5
    boxes: int  # ground-truth boxes in the tree


@dataclass(frozen=True, slots=True)
class Row:
    """One arm at one drop threshold, against its original val tree."""

    arm: str
    threshold: float
    spread: float
    map50_original: float
    map50_drop: float
    delta_map50: float
    map50_95_original: float
    map50_95_drop: float
    delta_map50_95: float
    boxes_dropped: int
    lost_matches: int
    rank_original: int
    rank_drop: int
    rank_changed: bool
    exceeds_spread: bool
    verdict: str  # Q3 for the whole threshold: "fix" or "disclose"


def parse_arm(values: Sequence[str]) -> Arm:
    """``NAME WEIGHTS SPREAD ORIGINAL THRESHOLD=DROP [THRESHOLD=DROP ...]``, checked on disk."""
    if len(values) < MIN_ARM_FIELDS:
        raise RescoreError(
            f"--arm {' '.join(values)}: expected NAME WEIGHTS SPREAD ORIGINAL THRESHOLD=DROP ..."
        )
    name, weights, spread, original, *drop_specs = values
    drops: dict[float, Path] = {}
    for spec in drop_specs:
        threshold, separator, path = spec.partition("=")
        if not separator:
            raise RescoreError(f"--arm {name}: {spec!r} is not THRESHOLD=DROP_DATA_YAML")
        drops[float(threshold)] = Path(path)
    arm = Arm(name, Path(weights), float(spread), Path(original), drops)
    if arm.spread < 0:
        raise RescoreError(f"--arm {name}: the seed spread {arm.spread} is negative")
    for path in (arm.weights, arm.original, *arm.drops.values()):
        if not path.is_file():
            raise RescoreError(f"--arm {name}: {path} does not exist")
    return arm


def check_arms(arms: Sequence[Arm]) -> list[float]:
    """The shared thresholds. Ranks only compare arms scored at the same threshold."""
    names = [arm.name for arm in arms]
    if len(set(names)) != len(names):
        raise RescoreError(f"arm names repeat: {names}")
    thresholds = sorted(arms[0].drops)
    for arm in arms:
        if sorted(arm.drops) != thresholds:
            raise RescoreError(
                f"--arm {arm.name} has thresholds {sorted(arm.drops)}, "
                f"--arm {arms[0].name} has {thresholds}; every arm needs the same ones"
            )
    return thresholds


def score(arm: Arm, data_yaml: Path, imgsz: int, batch: int, device: str | None) -> Score:
    """mAP from ``evaluate_detector``, match counts from the same ``model.val()`` per image."""
    metrics = task.evaluate_detector(
        weights=arm.weights, data_yaml=data_yaml, imgsz=imgsz, batch=batch, device=device
    )
    records = task.collect_image_stats(
        weights=arm.weights, data_yaml=data_yaml, imgsz=imgsz, batch=batch, device=device
    )
    return Score(
        map50=metrics.map50,
        map50_95=metrics.map50_95,
        true_positives=sum(int(record.tp[:, 0].sum()) for record in records.values()),
        boxes=sum(len(record.target_cls) for record in records.values()),
    )


def ranks(map50: dict[str, float]) -> dict[str, int]:
    """1 for the best mAP50. Ties keep the order the arms were given in."""
    order = sorted(map50, key=lambda name: -map50[name])
    return {name: position for position, name in enumerate(order, start=1)}


def build_rows(
    arms: Sequence[Arm], thresholds: Sequence[float], scores: dict[tuple[str, str], Score]
) -> list[Row]:
    """Compare every arm's drop scores with its original ones, and apply Q3 per threshold."""
    rank_original = ranks({arm.name: scores[(arm.name, ORIGINAL)].map50 for arm in arms})
    rows: list[Row] = []
    for threshold in thresholds:
        key = str(threshold)
        rank_drop = ranks({arm.name: scores[(arm.name, key)].map50 for arm in arms})
        drafts = []
        for arm in arms:
            before, after = scores[(arm.name, ORIGINAL)], scores[(arm.name, key)]
            delta = after.map50 - before.map50
            drafts.append(
                Row(
                    arm=arm.name,
                    threshold=threshold,
                    spread=arm.spread,
                    map50_original=before.map50,
                    map50_drop=after.map50,
                    delta_map50=delta,
                    map50_95_original=before.map50_95,
                    map50_95_drop=after.map50_95,
                    delta_map50_95=after.map50_95 - before.map50_95,
                    boxes_dropped=before.boxes - after.boxes,
                    lost_matches=before.true_positives - after.true_positives,
                    rank_original=rank_original[arm.name],
                    rank_drop=rank_drop[arm.name],
                    rank_changed=rank_original[arm.name] != rank_drop[arm.name],
                    exceeds_spread=abs(delta) > arm.spread,
                    verdict="",
                )
            )
        fix = any(row.rank_changed or row.exceeds_spread for row in drafts)
        verdict = "fix" if fix else "disclose"
        rows.extend(Row(**{**asdict(row), "verdict": verdict}) for row in drafts)
        logger.info("threshold %.2f: Q3 says %s", threshold, verdict)
    return rows


def write_csv(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(Row)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    """Arms and the output path are required: none of them has a safe default."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--arm",
        nargs="+",
        action="append",
        required=True,
        metavar="FIELD",
        help="NAME WEIGHTS SPREAD ORIGINAL_DATA_YAML THRESHOLD=DROP_DATA_YAML ... (repeatable)",
    )
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Score every arm on every tree, then write the CSV."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    arms = [parse_arm(values) for values in args.arm]
    thresholds = check_arms(arms)

    scores: dict[tuple[str, str], Score] = {}
    for arm in arms:
        trees = {ORIGINAL: arm.original} | {str(t): path for t, path in arm.drops.items()}
        for key, data_yaml in trees.items():
            logger.info("%s on %s tree: %s", arm.name, key, data_yaml)
            scores[(arm.name, key)] = score(arm, data_yaml, args.imgsz, args.batch, args.device)

    rows = build_rows(arms, thresholds, scores)
    write_csv(rows, args.out)
    logger.info("wrote %d row(s) to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
