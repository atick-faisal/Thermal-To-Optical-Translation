"""YOLO ``.txt`` label parsing."""

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
