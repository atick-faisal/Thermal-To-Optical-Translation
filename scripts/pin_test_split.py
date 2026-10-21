"""Pin a dataset's held-out test split membership, writing `splits/test/<name>.json`.

`scripts/freeze_splits.py` records train and val only, because `DatasetManifest` cannot see a
`test:` key (docs/design.md §9, The custom dataset). So a reshuffle *within* the ~100 held-out
custom pairs would go unseen. This script records their sorted stems and hash from an explicit
images directory, so the test split is pinned before it is first read, and no `data.yaml` is
needed or touched.

An existing pin with different stems is never overwritten: the script raises
`t2o.data.splits.SplitDriftError` instead, and re-pinning means deleting the file by hand.
`--check` verifies the directory against an existing pin instead of writing one.

The custom set's pin is written on the server and stays there, like its train/val record
(docs/design.md §9). It sits untracked inside a tracked directory, which `git clean -fd`
removes, so keep a copy outside the checkout.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig`
the way `t2o/cli.py` does for the package proper.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from t2o.data.splits import (
    SplitDriftError,
    load_test_pin,
    pin_test_split,
    verify_test_split,
    write_test_pin,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

DEFAULT_SPLITS_ROOT = Path("splits")


def pin_path(name: str, splits_root: Path) -> Path:
    # A subdirectory, so a `splits/*.json` glob over the train/val records never picks it up.
    return Path(splits_root) / "test" / f"{name}.json"


def pin_one(name: str, images_dir: Path, splits_root: Path, *, check: bool) -> None:
    path = pin_path(name, splits_root)

    if check:
        if not path.is_file():
            raise SplitDriftError(f"{name}: no test pin at {path} to check against")
        verify_test_split(load_test_pin(path), images_dir)
        logger.info("%s: test split matches its pin (%s)", name, path)
        return

    pin = pin_test_split(name, images_dir)
    write_test_pin(pin, path)
    logger.info("%s: pinned %d test stems from %s -> %s", name, len(pin.stems), images_dir, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="dataset name; names the pin file")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="the test split's images directory (either modality; stems are paired)",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=DEFAULT_SPLITS_ROOT,
        help=f"pins live under <splits-root>/test/ (default: {DEFAULT_SPLITS_ROOT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify against the existing pin instead of writing one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    pin_one(args.name, args.images, args.splits_root, check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
