"""An RT-DETR checkpoint runs through the existing scoring paths with no loader of its own.

The out-of-family judge (`docs/features/out-of-family-judge/plan.md`) rests on one upstream
behaviour: `YOLO(path)` inspects the loaded model's last layer and, when it is an RT-DETR
decoder, turns itself into `RTDETR` with RT-DETR's own predictor and validator
(`ultralytics/models/yolo/model.py:83`, ultralytics 8.4.117). Every scoring call in `t2o`
constructs `YOLO(...)`, so if that holds, none of them needs a seam. These tests pin it
against the real ultralytics, with no fakes, so an upgrade that drops the switch fails here
rather than mid-campaign on the server.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from conftest import NAMES
from t2o.data.manifest import DatasetManifest
from t2o.metrics.faithfulness import evaluate_faithfulness
from t2o.metrics.task import evaluate_detector

# The smallest stride-32 size RT-DETR runs at, to keep the CPU pass short. Its decoder takes
# the top `num_queries` (300) of the encoder's anchors, one per cell of the stride-8/16/32 grids:
# 336 at 128, but only 84 at 64, where `torch.topk` raises "selected index k out of range".
IMGSZ = 128


def test_yolo_turns_an_rtdetr_checkpoint_into_rtdetr(rtdetr_weights: Path) -> None:
    from ultralytics import RTDETR, YOLO
    from ultralytics.models.rtdetr.predict import RTDETRPredictor

    model = YOLO(str(rtdetr_weights))
    assert isinstance(model, RTDETR)

    # The predictor is what decodes boxes; a YOLO one would read the decoder output wrongly.
    image = Image.new("RGB", (IMGSZ, IMGSZ))
    model.predict(image, imgsz=IMGSZ, device="cpu", verbose=False)
    assert isinstance(model.predictor, RTDETRPredictor)


def test_evaluate_detector_scores_an_rtdetr_judge(rtdetr_weights: Path, data_yaml: Path) -> None:
    metrics = evaluate_detector(rtdetr_weights, data_yaml, imgsz=IMGSZ, batch=2, device="cpu")

    # ultralytics reports AP only for classes with val instances; the synthetic val split
    # happens to hold no "Fuse" box, so this is a subset check, not an equality.
    assert metrics.per_class_ap50
    assert set(metrics.per_class_ap50) <= set(NAMES)
    assert 0.0 <= metrics.map50 <= 1.0


def test_evaluate_faithfulness_scores_an_export_with_an_rtdetr_judge(
    rtdetr_weights: Path, data_yaml: Path, tmp_path: Path
) -> None:
    manifest = DatasetManifest.load(data_yaml)
    # A stand-in export: the real val frames rewritten as .png, the suffix export.py writes.
    translated = tmp_path / "translated" / "val" / "images"
    translated.mkdir(parents=True)
    for frame in manifest.val_images.iterdir():
        Image.open(frame).save(translated / f"{frame.stem}.png")

    metrics = evaluate_faithfulness(
        translated,
        manifest.val_images,
        rtdetr_weights,
        pairing=manifest.pairing,
        imgsz=IMGSZ,
        batch=2,
        device="cpu",
    )

    for rate in (
        metrics.false_object_rate,
        metrics.missed_object_rate,
        metrics.detection_consistency,
    ):
        assert 0.0 <= rate <= 1.0
