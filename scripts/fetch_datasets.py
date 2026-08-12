"""Fetch the trivially-scriptable public datasets: MSRS, CPLID, HIT-UAV, FLIR-aligned.

PLAN.md §9 splits public datasets by how they're obtained. This script covers only the set
reachable by a plain `git clone`/HTTP fetch with no browser or Google Drive step; LLVIP/M3FD/
TTPLA need `gdown` and a human once (fetched on the Mac, then re-hosted) and are out of scope
here.

Each dataset lands under `dataset/raw/<name>/` -- inside the wholesale-ignored `dataset/`
prefix (`.gitignore`), so no new ignore rule is needed, and next to where the M0.9 adapter
step will read raw layouts from. Nothing here converts a raw layout into the internal
`{split}/{visible,infrared}/{images,labels}` contract; that's the adapters bullet.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig`
the way `t2o/cli.py` does for the package proper.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

DEFAULT_DEST = Path("dataset/raw")


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    # Exactly one of these is set; it selects the fetch strategy.
    git_url: str | None = None
    hf_repo_id: str | None = None


SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(name="msrs", git_url="https://github.com/Linfeng-Tang/MSRS.git"),
    DatasetSource(name="cplid", git_url="https://github.com/InsulatorData/InsulatorDataSet.git"),
    DatasetSource(
        name="hituav",
        git_url="https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset.git",
    ),
    DatasetSource(name="flir", hf_repo_id="UserNae3/FLIR_aligned"),
)

_SOURCES_BY_NAME = {source.name: source for source in SOURCES}


class FetchError(RuntimeError):
    """Raised when a dataset name doesn't match any known source."""


def fetch_git(source: DatasetSource, dest: Path) -> None:
    assert source.git_url is not None
    logger.info("cloning %s -> %s", source.git_url, dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", source.git_url, str(dest)],
        check=True,
    )


def fetch_huggingface(source: DatasetSource, dest: Path) -> None:
    assert source.hf_repo_id is not None
    from huggingface_hub import snapshot_download

    logger.info("downloading hf dataset %s -> %s", source.hf_repo_id, dest)
    snapshot_download(repo_id=source.hf_repo_id, repo_type="dataset", local_dir=dest)


def fetch_dataset(source: DatasetSource, dest_root: Path) -> Path:
    """Fetch one dataset into ``dest_root/<name>``, skipping if already populated."""
    dest = dest_root / source.name
    if dest.exists() and any(dest.iterdir()):
        logger.info("%s already present at %s, skipping", source.name, dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    if source.git_url is not None:
        fetch_git(source, dest)
    else:
        fetch_huggingface(source, dest)
    return dest


def resolve_sources(names: Sequence[str]) -> list[DatasetSource]:
    if "all" in names:
        return list(SOURCES)
    try:
        return [_SOURCES_BY_NAME[name] for name in names]
    except KeyError as exc:
        raise FetchError(
            f"unknown dataset {exc.args[0]!r}; choose from {sorted(_SOURCES_BY_NAME)} or 'all'"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=[*sorted(_SOURCES_BY_NAME), "all"],
        help="which dataset(s) to fetch (default: all)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"destination root (default: {DEFAULT_DEST})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    for source in resolve_sources(args.dataset):
        fetch_dataset(source, args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
