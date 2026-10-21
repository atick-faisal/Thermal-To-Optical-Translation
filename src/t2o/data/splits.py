"""Freeze and hash a dataset's train/val split membership (PLAN.md invariant 2: "Frozen
data contract. Splits decided once, hashed, version-controlled.").

``dataset/`` is never tracked in git (PLAN.md §9) -- a fresh clone has no images at all --
so the images themselves can never be the thing that proves a split hasn't drifted. What
*can* be committed is small: which filename stems belong to ``train`` vs ``val``, and a hash
of that membership. A later re-fetch, an adapter re-run, or an upstream dataset revision that
reshuffles files then shows up as a loud, diffable ``git diff`` on ``splits/<name>.json``
(or a raised :class:`SplitDriftError`) instead of a silent change nobody notices.

Deliberately not wired into the engine yet -- freezing today's public-dataset splits doesn't
need it, and the custom ~850-pair dataset this matters most for lives only on the server
(PLAN.md §2). :func:`verify_split` is the reusable piece a future run-start check can call.

The held-out test split, which ``DatasetManifest`` cannot see, gets its own record: the pin at
the bottom of this module, written by ``scripts/pin_test_split.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from t2o.data.dataset import IMAGE_SUFFIXES
from t2o.data.manifest import DatasetManifest

# Matches Config.config_hash()'s sha256-hex-truncated convention (config/schema.py).
_HASH_LENGTH = 16


class SplitDriftError(ValueError):
    """Raised when a dataset's current split membership no longer matches its frozen record."""


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A frozen record of which image stems belong to ``train`` vs ``val``."""

    name: str
    train_stems: tuple[str, ...]  # sorted
    val_stems: tuple[str, ...]  # sorted
    train_hash: str
    val_hash: str

    @property
    def combined_hash(self) -> str:
        """One fingerprint for the whole split -- changes if either half does."""
        return _hash_text(self.train_hash + self.val_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train": {
                "count": len(self.train_stems),
                "hash": self.train_hash,
                "stems": list(self.train_stems),
            },
            "val": {
                "count": len(self.val_stems),
                "hash": self.val_hash,
                "stems": list(self.val_stems),
            },
            "combined_hash": self.combined_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitManifest:
        return cls(
            name=data["name"],
            train_stems=tuple(data["train"]["stems"]),
            val_stems=tuple(data["val"]["stems"]),
            train_hash=data["train"]["hash"],
            val_hash=data["val"]["hash"],
        )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:_HASH_LENGTH]


def _stems(images_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))


def freeze_split(name: str, manifest: DatasetManifest) -> SplitManifest:
    """Compute the current train/val stem membership and its hash for ``manifest``."""
    train_stems = _stems(manifest.train_images)
    val_stems = _stems(manifest.val_images)
    return SplitManifest(
        name=name,
        train_stems=train_stems,
        val_stems=val_stems,
        train_hash=_hash_text("\n".join(train_stems)),
        val_hash=_hash_text("\n".join(val_stems)),
    )


def write_split_manifest(record: SplitManifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n")
    return path


def load_split_manifest(path: Path) -> SplitManifest:
    return SplitManifest.from_dict(json.loads(Path(path).read_text()))


def verify_split(record: SplitManifest, manifest: DatasetManifest) -> None:
    """Raise :class:`SplitDriftError` if ``manifest``'s current membership no longer matches
    ``record`` -- the enforcement half of the frozen-split contract."""
    current = freeze_split(record.name, manifest)
    if current.train_hash != record.train_hash or current.val_hash != record.val_hash:
        raise SplitDriftError(_drift_message(record, current))


def _drift_message(record: SplitManifest, current: SplitManifest) -> str:
    parts = [f"{record.name}: split membership no longer matches the frozen record"]
    parts += _membership_diff("train", record.train_stems, current.train_stems)
    parts += _membership_diff("val", record.val_stems, current.val_stems)
    return "; ".join(parts)


def _membership_diff(split: str, before: tuple[str, ...], after: tuple[str, ...]) -> list[str]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    if not (added or removed):
        return []
    parts = [f"{split}: +{len(added)} -{len(removed)}"]
    if added:
        parts.append(f"  added: {', '.join(added[:5])}{'...' if len(added) > 5 else ''}")
    if removed:
        parts.append(f"  removed: {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")
    return parts


# --- The held-out test split --------------------------------------------------------------
#
# `DatasetManifest` cannot see a `test:` key (docs/design.md §9, The custom dataset), which is
# what keeps the ~100 held-out pairs out of every training and evaluation path. `SplitManifest`
# therefore cannot record them either, and a reshuffle *within* the test set would go unseen.
# The pin is a second, separate record built from an explicit images directory, so the
# manifest stays blind and every existing `splits/*.json` and `combined_hash` stays as it is.


class EmptySplitError(ValueError):
    """Raised when the directory to pin is missing or holds no images."""


@dataclass(frozen=True, slots=True)
class HeldOutSplitPin:
    """A frozen record of which image stems make up a dataset's held-out test split.

    Not ``TestSplitPin``: pytest tries to collect any ``Test*`` class a test module imports.
    """

    name: str
    stems: tuple[str, ...]  # sorted
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "test": {"count": len(self.stems), "hash": self.hash, "stems": list(self.stems)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeldOutSplitPin:
        return cls(name=data["name"], stems=tuple(data["test"]["stems"]), hash=data["test"]["hash"])


def pin_test_split(name: str, images_dir: Path) -> HeldOutSplitPin:
    """Compute the current stem membership and its hash for the test images in ``images_dir``.

    Stems ignore the file extension and pairs share a filename, so either modality's images
    directory gives the same pin.
    """
    images_dir = Path(images_dir)
    stems = _stems(images_dir) if images_dir.is_dir() else ()
    if not stems:
        # A mistyped server path must not pin zero stems.
        raise EmptySplitError(
            f"{name}: no images ({', '.join(sorted(IMAGE_SUFFIXES))}) to pin in {images_dir}"
        )
    return HeldOutSplitPin(name=name, stems=stems, hash=_hash_text("\n".join(stems)))


def write_test_pin(pin: HeldOutSplitPin, path: Path) -> Path:
    """Write ``pin`` to ``path``, refusing to replace an existing pin with different stems.

    ``scripts/freeze_splits.py`` warns and overwrites a drifted train/val record. Doing that
    here would let a re-run after a reshuffle silently re-pin the test split -- the one
    failure the pin exists to catch -- so re-pinning means deleting the file by hand.
    """
    path = Path(path)
    if path.is_file():
        existing = load_test_pin(path)
        if existing.hash != pin.hash:
            raise SplitDriftError(
                f"{_test_drift_message(existing, pin)}; refusing to overwrite {path} -- "
                "delete it by hand to re-pin"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pin.to_dict(), indent=2) + "\n")
    return path


def load_test_pin(path: Path) -> HeldOutSplitPin:
    return HeldOutSplitPin.from_dict(json.loads(Path(path).read_text()))


def verify_test_split(pin: HeldOutSplitPin, images_dir: Path) -> None:
    """Raise :class:`SplitDriftError` if ``images_dir`` no longer holds exactly ``pin``'s stems."""
    current = pin_test_split(pin.name, images_dir)
    if current.hash != pin.hash:
        raise SplitDriftError(_test_drift_message(pin, current))


def _test_drift_message(pin: HeldOutSplitPin, current: HeldOutSplitPin) -> str:
    parts = [f"{pin.name}: test split membership no longer matches its pin"]
    parts += _membership_diff("test", pin.stems, current.stems)
    return "; ".join(parts)
