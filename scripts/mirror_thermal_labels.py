"""Mirror an adapted dataset's `visible/labels` onto its `infrared/labels` side.

Operates on an already-adapted tree under `dataset/processed/<name>/`, not on a raw layout --
`adapt_datasets.py` short-circuits on a populated destination, and re-extracting FLIR's ~1.4GB
archive to add a few thousand small text files would be absurd. Run this once per dataset whose
thermal side needs to be evaluable.

Needed because `data/adapters/common.py` writes labels on the visible side only, so pointing a
detector at `infrared/images` there yields a split ultralytics reads as entirely unlabelled
(see `t2o.data.mirror` for why that is silent, and for the registration caveat the mirrored
boxes carry).

`--calibration` turns that caveat into a correction: each box is transformed onto the thermal
frame by the dataset's published residual rather than copied. On FLIR-aligned the plain copy
inherits a ~5.9 px offset that pushes the raw-thermal floor *down* and so **flatters**
translation, and E9's kill-test cleared the kill line by only +0.062 -- small enough that the
correction has to be measured rather than assumed. Restoring the uncorrected labels is the same
command without `--calibration`, which also removes the provenance marker beside them.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig` the
way `adapt_datasets.py` and `fetch_datasets.py` do.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from t2o.data.calibration import load_calibration
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import mirror_labels_onto_infrared

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="the dataset's data.yaml, or its root directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an infrared/labels that already holds files; only safe when those "
        "are known to be copies rather than thermal-side annotations of their own",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="a residual-misalignment constant, e.g. calibration/flir.json; de-rolls each box "
        "onto the thermal frame instead of copying it. Needs --force on an already-mirrored tree",
    )
    parser.add_argument(
        "--box-transform",
        choices=("corners", "centre"),
        default="corners",
        help="map all four box corners and re-bound (default), or map only the centre and keep "
        "the extent; the two agreeing is evidence the choice does not carry the result",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    manifest = DatasetManifest.load(args.data)
    calibration = load_calibration(args.calibration) if args.calibration else None
    written = mirror_labels_onto_infrared(
        manifest,
        force=args.force,
        calibration=calibration,
        box_transform=args.box_transform,
    )
    logger.info(
        "mirrored %d label file(s) in total: %s",
        sum(written.values()),
        ", ".join(f"{split}={count}" for split, count in written.items()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
