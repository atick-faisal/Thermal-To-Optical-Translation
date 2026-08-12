"""Fetch the public datasets PLAN.md §9 lists: the trivially-scriptable set (MSRS, CPLID,
HIT-UAV, FLIR-aligned) plus the Google-Drive-hosted set (LLVIP, M3FD, TTPLA) that needs
`gdown`.

Each dataset lands under `dataset/raw/<name>/` -- inside the wholesale-ignored `dataset/`
prefix (`.gitignore`), so no new ignore rule is needed, and next to where the M0.9 adapter
step reads raw layouts from. Nothing here converts a raw layout into the internal
`{split}/{visible,infrared}/{images,labels}` contract; that's the adapters bullet
(`t2o.data.adapters`).

**The LLVIP/M3FD/TTPLA sources are not executed for real by this script's author.** They're
large (LLVIP is 15,488 pairs at 1024x1280) and this was written on a dev machine with ~31GB
free disk -- confirmed with the user rather than risking filling it. Run
`--dataset llvip m3fd ttpla` yourself once disk/bandwidth allow. PLAN.md's own plan is
"fetch once on the Mac, re-host, then the server script is a plain `curl`" -- the re-host
step needs a destination (your own HF/S3/etc.) that hasn't been decided yet, so it isn't
wired here; the M3FD Google Drive folder in particular also contains TNO and RoadScene
(unrelated legacy fusion datasets bundled alongside it in the upstream repo's own share) --
expect more than just M3FD to land in `dataset/raw/m3fd/`.

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
    gdown_file_id: str | None = None
    gdown_folder_id: str | None = None


SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(name="msrs", git_url="https://github.com/Linfeng-Tang/MSRS.git"),
    DatasetSource(name="cplid", git_url="https://github.com/InsulatorData/InsulatorDataSet.git"),
    DatasetSource(
        name="hituav",
        git_url="https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset.git",
    ),
    DatasetSource(name="flir", hf_repo_id="UserNae3/FLIR_aligned"),
    # File/folder ids read directly off each dataset's own README, not guessed -- Drive
    # links otherwise silently 404 or serve an HTML interstitial instead of data.
    DatasetSource(
        name="llvip",
        # bupt-ai-cz/LLVIP download_dataset.md -- the registered/aligned set, not the
        # separate "raw" (unregistered pairs + video) download this project doesn't need.
        gdown_file_id="1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv",
    ),
    DatasetSource(
        name="m3fd",
        # JinyuanLiu-CV/TarDAL README -- one shared folder covering M3FD + TNO + RoadScene.
        gdown_folder_id="1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6",
    ),
    DatasetSource(
        name="ttpla",
        # R3ab/ttpla_dataset README -- "The dataset images here" link.
        gdown_file_id="1Yz59yXCiPKS0_X4K3x9mW22NLnxjvrr0",
    ),
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


def fetch_gdown_file(source: DatasetSource, dest: Path) -> None:
    assert source.gdown_file_id is not None
    import gdown

    logger.info("downloading gdown file %s -> %s", source.gdown_file_id, dest)
    # gdown ships no py.typed marker, so pyright treats its re-exports as private.
    gdown.download(  # pyright: ignore[reportPrivateImportUsage]
        id=source.gdown_file_id, output=f"{dest}/", quiet=False
    )


def fetch_gdown_folder(source: DatasetSource, dest: Path) -> None:
    assert source.gdown_folder_id is not None
    import gdown

    logger.info("downloading gdown folder %s -> %s", source.gdown_folder_id, dest)
    gdown.download_folder(  # pyright: ignore[reportPrivateImportUsage]
        id=source.gdown_folder_id, output=str(dest), quiet=False
    )


def fetch_dataset(source: DatasetSource, dest_root: Path) -> Path:
    """Fetch one dataset into ``dest_root/<name>``, skipping if already populated."""
    dest = dest_root / source.name
    if dest.exists() and any(dest.iterdir()):
        logger.info("%s already present at %s, skipping", source.name, dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    if source.git_url is not None:
        fetch_git(source, dest)
    elif source.hf_repo_id is not None:
        fetch_huggingface(source, dest)
    elif source.gdown_file_id is not None:
        fetch_gdown_file(source, dest)
    elif source.gdown_folder_id is not None:
        fetch_gdown_folder(source, dest)
    else:
        raise FetchError(f"{source.name}: no fetch strategy configured")
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
