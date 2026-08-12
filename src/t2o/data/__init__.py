"""The dataset contract: manifest, filename pairing, YOLO labels, paired dataset."""

from __future__ import annotations

from t2o.data.dataset import (
    TranslationBatch,
    TranslationPairDataset,
    TranslationSample,
    collate_translation_batch,
)
from t2o.data.labels import load_yolo_labels
from t2o.data.manifest import DatasetManifest, ManifestError
from t2o.data.pairing import Pairing
from t2o.data.splits import (
    SplitDriftError,
    SplitManifest,
    freeze_split,
    load_split_manifest,
    verify_split,
    write_split_manifest,
)

__all__ = [
    "DatasetManifest",
    "ManifestError",
    "Pairing",
    "SplitDriftError",
    "SplitManifest",
    "TranslationBatch",
    "TranslationPairDataset",
    "TranslationSample",
    "collate_translation_batch",
    "freeze_split",
    "load_split_manifest",
    "load_yolo_labels",
    "verify_split",
    "write_split_manifest",
]
