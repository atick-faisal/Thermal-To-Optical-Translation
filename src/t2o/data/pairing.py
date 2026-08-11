"""Visible <-> infrared <-> label path derivation by path-segment substitution.

Pairing by *sorted index* into three separate directory listings is the failure mode this
avoids: a single missing or extra file silently shifts every subsequent pair by one and
trains on mismatched modalities. Deriving each counterpart from the visible path makes a
mismatch impossible and a missing file loud.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

IMAGES_SEGMENT = "images"
LABELS_SEGMENT = "labels"
LABEL_SUFFIX = ".txt"


def _substitute_segment(path: Path, old: str, new: str) -> Path:
    matches = [i for i, part in enumerate(path.parts) if part == old]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one '{old}' path segment in '{path}' (found {len(matches)}); "
            f"cannot derive the '{new}' counterpart"
        )
    parts = list(path.parts)
    parts[matches[0]] = new
    return Path(*parts)


@dataclass(frozen=True, slots=True)
class Pairing:
    """Maps a visible image path to its infrared and label counterparts."""

    visible_token: str = "visible"
    infrared_token: str = "infrared"

    def infrared_path(self, visible: Path) -> Path:
        return _substitute_segment(visible, self.visible_token, self.infrared_token)

    def label_path(self, visible: Path) -> Path:
        """Labels live beside the images, read from the visible side only."""
        return _substitute_segment(visible, IMAGES_SEGMENT, LABELS_SEGMENT).with_suffix(
            LABEL_SUFFIX
        )

    def validate_pairs(self, visible_paths: Iterable[Path]) -> None:
        """Raise if any visible image lacks an infrared counterpart.

        Reports the full count and a preview, not just the first failure -- a systematic
        problem (wrong token, half-synced directory) should be diagnosable in one run.
        """
        paths: Sequence[Path] = list(visible_paths)
        missing = [p for p in paths if not self.infrared_path(p).exists()]
        if missing:
            preview = ", ".join(str(self.infrared_path(p)) for p in missing[:5])
            more = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
            raise FileNotFoundError(
                f"{len(missing)}/{len(paths)} infrared counterparts missing: {preview}{more}"
            )
