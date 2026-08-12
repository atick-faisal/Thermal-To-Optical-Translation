"""M0.6: the `Translator` protocol and its CPU stand-in.

Round-trips only as far as data -> translate -> fidelity-eval. The coupling and export
legs of PLAN.md's full data -> coupling -> export -> eval path don't exist yet (M0.7,
M0.8); this is the scope M0.6 can actually exercise.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from t2o.config import Config
from t2o.data.dataset import TranslationBatch, TranslationPairDataset, collate_translation_batch
from t2o.translators import StubTranslator, Translator


def _batch(n: int, h: int, w: int, seed: int = 0) -> TranslationBatch:
    generator = torch.Generator().manual_seed(seed)
    return TranslationBatch(
        visible=torch.rand((n, 3, h, w), generator=generator),
        infrared=torch.rand((n, 1, h, w), generator=generator),
        batch_idx=torch.zeros(0),
        cls=torch.zeros(0, 1),
        bboxes=torch.zeros(0, 4),
        names=[f"sample_{i}" for i in range(n)],
    )


# --------------------------------------------------------------------------- protocol


def test_stub_satisfies_the_translator_protocol() -> None:
    assert isinstance(StubTranslator(), Translator)


def test_config_hidden_channels_reaches_the_module() -> None:
    from t2o.config import StubTranslatorConfig

    config = Config.load(overrides={"translator": {"backbone": "stub", "hidden_channels": 7}})
    assert isinstance(config.translator, StubTranslatorConfig)  # narrows the union for pyright
    translator = StubTranslator(hidden_channels=config.translator.hidden_channels)
    first_conv = translator.net[0]
    assert isinstance(first_conv, torch.nn.Conv2d)
    assert first_conv.out_channels == 7


# --------------------------------------------------------------------------- translate


def test_translate_shape_dtype_and_range() -> None:
    batch = _batch(n=2, h=32, w=32)
    output = StubTranslator().translate(batch)

    assert output.shape == (2, 3, 32, 32)
    assert output.dtype == torch.float32
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_translate_is_grad_connected() -> None:
    translator = StubTranslator()
    batch = _batch(n=1, h=32, w=32)

    output = translator.translate(batch)
    output.sum().backward()

    grads = [p.grad for p in translator.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads if g is not None)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- fit


def test_fit_returns_a_finite_loss() -> None:
    result = StubTranslator().fit(_batch(n=2, h=32, w=32))
    assert math.isfinite(result["loss_l2"])


def test_fit_drives_the_loss_down_on_a_fixed_batch() -> None:
    translator = StubTranslator(lr=1e-2)
    batch = _batch(n=2, h=32, w=32)

    first = translator.fit(batch)["loss_l2"]
    last = first
    for _ in range(20):
        last = translator.fit(batch)["loss_l2"]

    assert last < first


def test_fit_ignores_task_loss_when_weight_is_zero() -> None:
    translator = StubTranslator()
    calls = 0

    def task_loss(pred: torch.Tensor, batch: TranslationBatch) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return pred.sum()

    result = translator.fit(_batch(n=2, h=32, w=32), task_loss=task_loss, task_weight=0.0)

    assert calls == 0
    assert "loss_det" not in result


def test_fit_adds_the_task_loss_when_weight_is_positive() -> None:
    translator = StubTranslator()

    def task_loss(pred: torch.Tensor, batch: TranslationBatch) -> torch.Tensor:
        return pred.sum()

    result = translator.fit(_batch(n=2, h=32, w=32), task_loss=task_loss, task_weight=0.5)

    assert "loss_det" in result
    assert math.isfinite(result["loss_total"])
    assert result["loss_total"] != result["loss_l2"]


# --------------------------------------------------------------------------- round trip


@pytest.mark.slow
def test_round_trips_through_the_dataset_and_fidelity_eval(dataset_root: Path) -> None:
    from t2o.metrics.fidelity import FidelityEvaluator

    dataset = TranslationPairDataset(dataset_root / "train" / "visible" / "images")
    batch = collate_translation_batch([dataset[i] for i in range(len(dataset))])

    translator = StubTranslator()
    with torch.no_grad():
        prediction = translator.translate(batch)

    evaluator = FidelityEvaluator(kid_subset_size=len(dataset))
    evaluator.update(prediction, batch["visible"])
    result = evaluator.compute()

    assert math.isfinite(result.psnr)
    assert math.isfinite(result.ssim)
    assert math.isfinite(result.lpips)
