"""Freeze and hash the train/val split membership of every adapted dataset
(PLAN.md invariant 2), writing `splits/<name>.json` -- the one thing about a dataset's split
identity that reaches git, since `dataset/` itself is never tracked (PLAN.md §9).

Discovers every `<name>/data.yaml` under `--data-root` (the adapters' own default output,
`dataset/processed/`) unless `--dataset` names a subset. The custom ~850-pair dataset lives
only on the server (PLAN.md §2) and has no `data.yaml` on this machine -- freeze it there
once it does, using this same script against its `data.yaml`.

`--check` verifies the current split against an existing frozen record instead of writing
one, raising `t2o.data.splits.SplitDriftError` on any mismatch -- the drift-detection half of
the frozen-split contract, suitable for a pre-training sanity check on the server.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig`
the way `t2o/cli.py` does for the package proper.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from t2o.data.manifest import DatasetManifest
from t2o.data.splits import (
    SplitDriftError,
    freeze_split,
    load_split_manifest,
    verify_split,
    write_split_manifest,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

DEFAULT_DATA_ROOT = Path("dataset/processed")
DEFAULT_SPLITS_ROOT = Path("splits")


def discover_datasets(data_root: Path) -> list[str]:
    return sorted(p.parent.name for p in Path(data_root).glob("*/data.yaml"))


def freeze_one(name: str, data_root: Path, splits_root: Path, *, check: bool) -> None:
    manifest = DatasetManifest.load(Path(data_root) / name / "data.yaml")
    path = Path(splits_root) / f"{name}.json"

    if check:
        if not path.is_file():
            raise SplitDriftError(f"{name}: no frozen record at {path} to check against")
        verify_split(load_split_manifest(path), manifest)
        logger.info("%s: matches its frozen record (%s)", name, path)
        return

    record = freeze_split(name, manifest)
    if path.is_file():
        try:
            verify_split(load_split_manifest(path), manifest)
        except SplitDriftError as exc:
            logger.warning("%s: overwriting a drifted frozen record -- %s", name, exc)
    write_split_manifest(record, path)
    logger.info(
        "%s: froze %d train / %d val stems -> %s",
        name,
        len(record.train_stems),
        len(record.val_stems),
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help="dataset name(s) under --data-root (default: every one discovered)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"root containing <name>/data.yaml directories (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=DEFAULT_SPLITS_ROOT,
        help=f"where frozen records live (default: {DEFAULT_SPLITS_ROOT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify against the existing frozen record instead of (re)writing it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    names = args.dataset or discover_datasets(args.data_root)
    if not names:
        raise SystemExit(f"no data.yaml found under {args.data_root}")
    for name in names:
        freeze_one(name, args.data_root, args.splits_root, check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
