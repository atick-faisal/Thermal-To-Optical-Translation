"""YOLO ``.txt`` label parsing, and the cache ultralytics keeps beside it."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

_FIELDS_PER_LINE = 5  # cls cx cy w h


def load_yolo_labels(path: Path) -> tuple[Tensor, Tensor]:
    """Read one YOLO label file into ``(cls, bboxes)``.

    ``cls`` is ``(M, 1)`` float and ``bboxes`` is ``(M, 4)`` normalised ``xywh`` -- the
    exact per-sample layout ultralytics' collate expects. A missing or empty file is a
    legitimate negative sample and yields zero instances rather than an error; a
    *malformed* line is a data bug and raises.
    """
    text = path.read_text().strip() if path.exists() else ""
    if not text:
        return torch.zeros(0, 1), torch.zeros(0, 4)

    rows: list[list[float]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not (line := line.strip()):
            continue
        fields = line.split()
        if len(fields) != _FIELDS_PER_LINE:
            raise ValueError(
                f"{path}:{number}: expected {_FIELDS_PER_LINE} fields (cls cx cy w h), "
                f"got {len(fields)}: {line!r}"
            )
        try:
            rows.append([float(f) for f in fields])
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: non-numeric field in {line!r}") from exc

    if not rows:
        return torch.zeros(0, 1), torch.zeros(0, 4)

    table = torch.tensor(rows, dtype=torch.float32)
    return table[:, 0:1], table[:, 1:5]


def invalidate_label_cache(labels_dir: Path) -> Path | None:
    """Delete the ultralytics label cache beside ``labels_dir``. Returns it, or ``None``.

    **Anything that rewrites label files in place must call this.** ultralytics caches a split's
    parsed labels in ``<labels_dir>.cache`` and decides whether that cache is current with
    ``ultralytics/data/utils.py::get_hash``, which hashes the *sum of the files' sizes* and the
    path strings -- never their contents. ``YOLODataset.get_labels`` then reuses the pickle
    whenever that hash matches.

    A rewritten YOLO label file collides with that hash by construction. ``:.6f`` on a normalised
    coordinate is always exactly 8 characters and the class index does not change, so a file whose
    every box moved has byte-for-byte the same length as the one it replaced: across FLIR's
    ``val/infrared`` split all 1,013 files changed content and **none** changed size. E9's first
    de-rolled gate therefore scored the uncorrected boxes out of a day-old cache and returned
    numbers identical to the previous run in all 17 significant figures (TASKS.md M3 E9 step 3b).
    """
    cache = labels_dir.with_suffix(".cache")
    if not cache.is_file():
        return None
    cache.unlink()
    return cache
