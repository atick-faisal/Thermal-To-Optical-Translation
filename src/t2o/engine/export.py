"""Export translated images + mirrored labels to a YOLO dataset the detector can train on.

Ported from ``../Clean-SeAFusion/src/seafusion/engine/export.py``. That version recombines
a fused luma plane with the visible image's chroma (``ycbcr_to_rgb``) because its fusion
network only predicts luma. t2o's ``Translator.translate() -> Tensor`` already returns RGB
directly (``translators/protocol.py``), so export calls it as-is -- no color-space step.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import cast

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from t2o.config import Config
from t2o.data.dataset import TranslationPairDataset, collate_translation_batch
from t2o.data.manifest import DatasetManifest
from t2o.imaging import Normalize, to_uint8
from t2o.translators.protocol import Translator

logger = logging.getLogger(__name__)

DATA_FILENAME = "data.yaml"
EXPORT_SUFFIX = ".png"


@torch.inference_mode()
def export_split(
    translator: nn.Module,
    dataset: TranslationPairDataset,
    out_dir: Path,
    normalize: Normalize = Normalize.CLAMP,
    device: torch.device | None = None,
    batch_size: int = 8,
    workers: int = 0,
) -> int:
    """Translate every pair in ``dataset`` into ``out_dir/{images,labels}``.

    Normalisation is per image, always (``t2o.imaging.to_uint8``). The original min-max
    normalised across the whole batch (``train.py:369``), so an exported pixel depended on
    which other images happened to share its batch -- making the detector's training data a
    function of batch size and shuffle order. Batch size here changes throughput and
    nothing else.
    """
    if not isinstance(translator, Translator):
        raise TypeError("translator must implement fit()/translate() (the Translator protocol)")

    device = device or next(translator.parameters()).device
    was_training = translator.training
    translator.eval()

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    loader: DataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_translation_batch,
    )

    translator_proto = cast(Translator, translator)
    written = 0
    for batch in loader:
        translated = translator_proto.translate(batch)

        for image, name in zip(translated, batch["names"], strict=True):
            stem = Path(name).stem
            array = to_uint8(image, normalize).permute(1, 2, 0).numpy()
            Image.fromarray(array, mode="RGB").save(images_dir / f"{stem}{EXPORT_SUFFIX}")
            _mirror_label(dataset, name, labels_dir / f"{stem}.txt")
            written += 1

    translator.train(was_training)
    logger.info("exported %d translated images to %s (normalize=%s)", written, out_dir, normalize)
    return written


def _mirror_label(dataset: TranslationPairDataset, name: str, destination: Path) -> None:
    """Copy the visible-side label verbatim.

    Copying rather than re-serialising the parsed tensors keeps the exported labels
    byte-identical to the source. Export applies no augmentation, so the geometry is
    unchanged and a copy is exactly right.
    """
    source = dataset.pairing.label_path(dataset.images_dir / name)
    if source.exists():
        shutil.copyfile(source, destination)
    else:
        destination.write_text("")


def export_translated(
    translator: nn.Module,
    config: Config,
    out_root: Path,
    device: torch.device | None = None,
    manifest: DatasetManifest | None = None,
) -> Path:
    """Export train and val splits plus a ``data.yaml`` ultralytics can consume.

    Returns the path of the written ``data.yaml``.
    """
    out_root = Path(out_root)
    manifest = manifest or DatasetManifest.load(config.data.manifest)

    for split, images_dir in (
        ("train", manifest.train_images),
        ("val", manifest.val_images),
    ):
        dataset = TranslationPairDataset(
            images_dir, pairing=manifest.pairing, hflip=0.0, crop=None, num_classes=manifest.nc
        )
        export_split(
            translator,
            dataset,
            out_root / split,
            normalize=config.export.normalize,
            device=device,
            batch_size=config.detector.evaluation.batch,
            workers=config.train.workers,
        )

    return write_data_yaml(out_root, manifest)


def write_data_yaml(out_root: Path, manifest: DatasetManifest) -> Path:
    """Emit a data.yaml pointing at the translated images, in ultralytics' own layout.

    Classes are copied from the source manifest, so the exported dataset cannot disagree
    with the labels that were just mirrored into it.
    """
    destination = Path(out_root) / DATA_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            {
                "path": str(Path(out_root).resolve()),
                "train": "train/images",
                "val": "val/images",
                "nc": manifest.nc,
                "names": manifest.class_names,
            },
            sort_keys=False,
        )
    )
    return destination
