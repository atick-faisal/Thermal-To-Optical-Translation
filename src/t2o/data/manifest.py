"""The dataset's own ``data.yaml``, read as the single source of truth.

Everything about a dataset -- where it lives, its splits, its classes, and which path
segment names each modality -- is already declared in the file ultralytics and
``../RGBT-Fusion-Detection`` both consume. Re-declaring any of it in this project's config
would create a second source of truth that can silently disagree with the labels on disk,
so nothing here is configurable: point ``--data`` at a ``data.yaml`` and that file decides.

Only ``path``/``train``/``val``/``nc``/``names``/``rgbt`` are read; every other key is
ignored silently. In particular a stray ``channels: 6`` block -- left over from an earlier
RGBT-6-channel convention our detector does not use -- is harmless if a ``data.yaml`` still
carries one. Worth stating here rather than assuming: ``dataset/`` is never tracked
(PLAN.md §9), so the local copy of this file never reaches the server for anyone to read.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from t2o.data.pairing import Pairing

logger = logging.getLogger(__name__)

RGBT_BLOCK = "rgbt"


class ManifestError(ValueError):
    """Raised when a ``data.yaml`` is missing, malformed, or inconsistent."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """A resolved ``data.yaml``. All paths are absolute."""

    path: Path  # the data.yaml itself
    root: Path
    train_images: Path
    val_images: Path
    nc: int
    names: tuple[str, ...]
    pairing: Pairing

    @classmethod
    def load(cls, path: Path | str) -> DatasetManifest:
        path = Path(path)
        if path.is_dir():
            # A directory is very likely a dataset root; point at its manifest rather than
            # failing on something the user obviously meant.
            path = path / "data.yaml"
        if not path.is_file():
            raise ManifestError(f"dataset manifest not found: {path}")

        try:
            declared = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ManifestError(f"{path}: could not be parsed: {exc}") from exc
        if not isinstance(declared, Mapping):
            raise ManifestError(f"{path}: expected a mapping at the top level")

        names = _read_names(path, declared)
        nc = _read_nc(path, declared, names)
        root = _resolve_root(path, declared)

        train = _split_dir(path, root, declared, "train")
        val = _split_dir(path, root, declared, "val")

        block = declared.get(RGBT_BLOCK) or {}
        if not isinstance(block, Mapping):
            raise ManifestError(f"{path}: '{RGBT_BLOCK}' must be a mapping")
        pairing = Pairing(
            visible_token=str(block.get("visible_token", Pairing.visible_token)),
            infrared_token=str(block.get("infrared_token", Pairing.infrared_token)),
        )
        if pairing.visible_token == pairing.infrared_token:
            raise ManifestError(f"{path}: visible_token and infrared_token must differ")

        logger.info("%s: %d classes %s, root %s", path.name, nc, list(names), root)
        return cls(
            path=path.resolve(),
            root=root,
            train_images=train,
            val_images=val,
            nc=nc,
            names=names,
            pairing=pairing,
        )

    @property
    def class_names(self) -> list[str]:
        return list(self.names)


def _read_names(path: Path, declared: Mapping) -> tuple[str, ...]:
    names = declared.get("names")
    if isinstance(names, Mapping):
        # ultralytics also accepts {0: "a", 1: "b"}; order by index, not insertion.
        names = [names[key] for key in sorted(names)]
    if not isinstance(names, list) or not names:
        raise ManifestError(f"{path}: 'names' must be a non-empty list or index->name mapping")
    return tuple(str(n) for n in names)


def _read_nc(path: Path, declared: Mapping, names: tuple[str, ...]) -> int:
    if "nc" not in declared:
        return len(names)
    nc = declared["nc"]
    if not isinstance(nc, int) or isinstance(nc, bool):
        raise ManifestError(f"{path}: 'nc' must be an integer, got {nc!r}")
    if nc != len(names):
        raise ManifestError(
            f"{path}: nc ({nc}) does not match the {len(names)} entries in 'names'. "
            f"The manifest has to be internally consistent before anything can trust it."
        )
    return nc


def _resolve_root(manifest: Path, declared: Mapping) -> Path:
    """Find the dataset root a ``path:`` entry refers to.

    ``path`` is conventionally relative to wherever the project is run from, not to the
    manifest, so the same file resolves differently depending on the caller. Rather than
    guess one convention, try each and take the first that actually contains the declared
    training split.
    """
    train = declared.get("train")
    candidates: list[Path] = []

    raw = declared.get("path")
    if raw is not None:
        declared_path = Path(str(raw))
        if declared_path.is_absolute():
            candidates.append(declared_path)
        else:
            candidates.append(manifest.parent / declared_path)
            candidates.append(Path.cwd() / declared_path)
    candidates.append(manifest.parent)

    for candidate in candidates:
        if train is None or (candidate / str(train)).exists():
            return candidate.resolve()

    tried = "\n  ".join(str(c) for c in candidates)
    raise ManifestError(
        f"{manifest}: could not locate the dataset root containing '{train}'. Tried:\n  {tried}"
    )


def _split_dir(manifest: Path, root: Path, declared: Mapping, split: str) -> Path:
    if split not in declared:
        raise ManifestError(f"{manifest}: missing required '{split}' split")
    directory = (root / str(declared[split])).resolve()
    if not directory.is_dir():
        raise ManifestError(f"{manifest}: '{split}' points at {directory}, which does not exist")
    return directory
