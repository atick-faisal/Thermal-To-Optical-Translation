"""ModTr as a `Translator` (modtr-baseline ADR-01, ADR-04, Q8): fusion, objective, schedule.

Every translator here is built with `pretrained=False`, so no ImageNet weights are downloaded.
The gradient test uses the offline `detector_weights` fixture for its gradient only. That
fixture scores mAP@50 0.0 on anything (margin-gate pitfall MARGIN-GATE-01), so nothing here
reads a detection metric from it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from t2o.config import Config, ConfigError
from t2o.coupling.detection_loss import DetectionTaskLoss
from t2o.data.dataset import TranslationBatch
from t2o.detection.frozen import FrozenDetector
from t2o.translators import ModTrTranslator, Translator, build_translator
from t2o.translators.modtr import ModTrObjectiveError

# A detection-only config: what ModTr's own experiment YAMLs must say.
_DETECTION_ONLY: dict[str, Any] = {
    "translator": {"backbone": "modtr", "pretrained": False},
    "loss": {"l2": 0.0, "lpips": 0.0, "gan": 0.0},
    "coupling": {"task_weights": [1.0]},
}


def _batch(n: int = 2, size: int = 64, seed: int = 0) -> TranslationBatch:
    generator = torch.Generator().manual_seed(seed)
    return TranslationBatch(
        visible=torch.rand((n, 3, size, size), generator=generator),
        infrared=torch.rand((n, 1, size, size), generator=generator),
        batch_idx=torch.arange(n, dtype=torch.float32),
        cls=torch.zeros((n, 1)),
        bboxes=torch.tensor([[0.5, 0.5, 0.4, 0.4]] * n),
        names=[f"sample_{i}" for i in range(n)],
    )


def _translator(total_batches: int = 10, lr: float = 1.0e-4) -> ModTrTranslator:
    torch.manual_seed(0)
    return ModTrTranslator(total_batches=total_batches, lr=lr, pretrained=False)


def _mean_loss(pred: torch.Tensor, batch: TranslationBatch) -> torch.Tensor:
    return pred.mean()


def _config(data_yaml: Path, **sections: dict[str, Any]) -> Config:
    overrides: dict[str, Any] = {**_DETECTION_ONLY, "data": {"manifest": str(data_yaml)}}
    for name, section in sections.items():
        overrides[name] = {**overrides.get(name, {}), **section}
    return Config.load(overrides=overrides)


# --------------------------------------------------------------------------- translate


def test_modtr_satisfies_the_translator_protocol() -> None:
    assert isinstance(_translator(), Translator)


def test_translate_shape_dtype_and_range() -> None:
    output = _translator().translate(_batch())

    assert output.shape == (2, 3, 64, 64)
    assert output.dtype == torch.float32
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_translate_is_the_multiplicative_fusion_of_the_thermal_input() -> None:
    # ModTr-dot, `x * h(x)` with h in [0, 1]: no pixel can come out brighter than its thermal
    # input, and a black thermal pixel stays black. An additive or unfused output breaks both.
    batch = _batch()
    batch["infrared"][:, :, :8, :8] = 0.0
    output = _translator().translate(batch)

    assert torch.all(output <= batch["infrared"].repeat(1, 3, 1, 1))
    assert torch.all(output[:, :, :8, :8] == 0.0)


def test_translate_is_grad_connected() -> None:
    translator = _translator()
    translator.translate(_batch()).sum().backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in translator.net.parameters())


# --------------------------------------------------------------------------- fit


@pytest.mark.parametrize(("task_loss", "task_weight"), [(None, 1.0), (_mean_loss, 0.0)])
def test_fit_refuses_to_run_without_a_detection_term(task_loss: Any, task_weight: float) -> None:
    with pytest.raises(ModTrObjectiveError, match="detector's loss alone"):
        _translator().fit(_batch(), task_loss=task_loss, task_weight=task_weight)


def test_detection_gradient_accumulates_over_two_fits_before_one_step(
    detector_weights: Path,
) -> None:
    # Two checks, because either alone passes without proving anything. AdamW's weight decay
    # moves the weights even at zero gradient, so "the weights moved" says nothing about the
    # detector; and with accumulation 2, the first fit must not step at all.
    translator = _translator()
    task_loss = DetectionTaskLoss(FrozenDetector(detector_weights, nc=4))
    before = [p.detach().clone() for p in translator.net.parameters()]

    stats = translator.fit(_batch(), task_loss=task_loss, task_weight=1.0)

    assert math.isfinite(stats["loss_det"])
    for part in (translator.net.stem, translator.net.stage4, translator.net.decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    assert all(torch.equal(b, p) for b, p in zip(before, translator.net.parameters(), strict=True))

    translator.fit(_batch(seed=1), task_loss=task_loss, task_weight=1.0)

    assert not all(
        torch.equal(b, p) for b, p in zip(before, translator.net.parameters(), strict=True)
    )
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in translator.net.parameters())


def test_the_cosine_steps_once_per_optimiser_step_over_t_max_batches() -> None:
    # ADR-04 keeps upstream's quirk: T_max counts batches but the scheduler steps once per
    # optimiser step, i.e. every second batch. After 4 batches of a T_max = 8 run, the cosine
    # is at step 2 of 8, not 4 of 8.
    lr = 1.0e-3
    translator = _translator(total_batches=8, lr=lr)
    for seed in range(4):
        translator.fit(_batch(seed=seed), task_loss=_mean_loss, task_weight=1.0)

    expected = lr * 0.5 * (1.0 + math.cos(math.pi * 2 / 8))
    assert translator.optimizer.param_groups[0]["lr"] == pytest.approx(expected)


def test_a_reloaded_state_dict_continues_the_schedule_and_the_optimiser() -> None:
    lr = 1.0e-3
    trained = _translator(total_batches=8, lr=lr)
    for seed in range(3):  # one optimiser step, then one batch left half-accumulated
        trained.fit(_batch(seed=seed), task_loss=_mean_loss, task_weight=1.0)

    resumed = _translator(total_batches=8, lr=lr)
    resumed.load_state_dict(trained.state_dict())

    assert resumed.scheduler.last_epoch == 1
    assert resumed.optimizer.param_groups[0]["lr"] == trained.optimizer.param_groups[0]["lr"]
    assert (
        resumed.optimizer.state_dict()["state"].keys()
        == trained.optimizer.state_dict()["state"].keys()
    )
    # The half-accumulated gradient is not in the checkpoint, so the next fit starts a fresh
    # pair rather than stepping on one batch's gradient.
    resumed.fit(_batch(seed=3), task_loss=_mean_loss, task_weight=1.0)
    assert resumed.scheduler.last_epoch == 1


# --------------------------------------------------------------------------- config + build


def test_build_translator_counts_t_max_from_the_training_corpus(data_yaml: Path) -> None:
    # The synthetic train split has 5 images: ceil(5 / 2) = 3 batches, x 4 epochs x 2 stages.
    config = _config(
        data_yaml,
        train={"batch_size": 2, "epochs_per_stage": 4},
        coupling={"task_weights": [1.0, 1.0]},
    )
    translator = build_translator(config)

    assert isinstance(translator, ModTrTranslator)
    assert translator.scheduler.T_max == 3 * 4 * 2
    assert translator.optimizer.param_groups[0]["lr"] == config.train.lr


def test_build_translator_counts_only_the_capped_corpus(data_yaml: Path) -> None:
    config = _config(
        data_yaml,
        data={"max_train_images": 4},
        train={"batch_size": 2, "epochs_per_stage": 4},
    )
    translator = build_translator(config)

    assert isinstance(translator, ModTrTranslator)
    assert translator.scheduler.T_max == 2 * 4


def test_build_translator_rejects_the_default_reconstruction_weights(data_yaml: Path) -> None:
    # `LossConfig` defaults to lpips 5.0; a ModTr config that leaves `loss` out must not build.
    config = Config.load(
        overrides={
            "translator": _DETECTION_ONLY["translator"],
            "coupling": _DETECTION_ONLY["coupling"],
            "data": {"manifest": str(data_yaml)},
        }
    )
    with pytest.raises(ConfigError, match="detection loss only"):
        build_translator(config)


def test_build_translator_rejects_a_stage_with_no_detection_term(data_yaml: Path) -> None:
    config = _config(data_yaml, coupling={"task_weights": [0.0, 1.0]})
    with pytest.raises(ConfigError, match="detection term in every stage"):
        build_translator(config)


def test_unknown_modtr_config_keys_are_rejected() -> None:
    with pytest.raises(ConfigError):
        Config.load(overrides={"translator": {"backbone": "modtr", "fuse_data": "sum"}})


def test_config_hash_differs_from_pix2pix() -> None:
    modtr = Config.load(overrides=_DETECTION_ONLY)
    pix2pix = Config.load(overrides={**_DETECTION_ONLY, "translator": {"backbone": "pix2pix"}})
    assert modtr.config_hash() != pix2pix.config_hash()


# --------------------------------------------------------------------------- slow: the loop


@pytest.mark.slow
def test_full_loop_runs_end_to_end_with_modtr(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """`build_translator` dispatches through `Trainer`/`run_loop` with the real detection term."""
    from t2o.engine.loop import run_loop

    config = _config(
        data_yaml,
        train={"batch_size": 2, "epochs_per_stage": 1},
        detector={"in_loop": {"weights": str(detector_weights)}},
        runtime={"device": "cpu", "workers": 0},
    )
    translator = build_translator(config)
    results = run_loop(config, translator, run_dir=tmp_path, train_detector_stages=False)

    assert len(results) == 1
    assert math.isfinite(results[0].epochs[-1].train_losses["loss_det"])
