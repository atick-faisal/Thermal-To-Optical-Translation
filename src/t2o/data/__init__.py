"""The dataset contract: manifest, filename pairing, YOLO labels, paired dataset."""

from __future__ import annotations

from t2o.data.budget import BudgetError, write_budget_manifest
from t2o.data.dataset import (
    TranslationBatch,
    TranslationPairDataset,
    TranslationSample,
    annotated_subset,
    collate_translation_batch,
)
from t2o.data.labels import load_yolo_labels
from t2o.data.manifest import DatasetManifest, ManifestError
from t2o.data.mirror import MirrorError, mirror_labels_onto_infrared
from t2o.data.pairing import Pairing
from t2o.data.splits import (
    EmptySplitError,
    HeldOutSplitPin,
    SplitDriftError,
    SplitManifest,
    freeze_split,
    load_split_manifest,
    load_test_pin,
    pin_test_split,
    verify_split,
    verify_test_split,
    write_split_manifest,
    write_test_pin,
)

__all__ = [
    "BudgetError",
    "DatasetManifest",
    "EmptySplitError",
    "HeldOutSplitPin",
    "ManifestError",
    "MirrorError",
    "Pairing",
    "SplitDriftError",
    "SplitManifest",
    "TranslationBatch",
    "TranslationPairDataset",
    "TranslationSample",
    "annotated_subset",
    "collate_translation_batch",
    "freeze_split",
    "load_split_manifest",
    "load_test_pin",
    "load_yolo_labels",
    "mirror_labels_onto_infrared",
    "pin_test_split",
    "verify_split",
    "verify_test_split",
    "write_budget_manifest",
    "write_split_manifest",
    "write_test_pin",
]
