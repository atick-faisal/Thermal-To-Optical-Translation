"""`engine/trainer.py`'s epoch loop, checkpointing, and resume.

Built on the synthetic `data_yaml` fixture (`tests/conftest.py`): 5 train / 3 val
images, disjoint stems. Config overrides keep every run tiny and CPU-fast.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn

from t2o.config import Config
from t2o.data.dataset import TranslationBatch
from t2o.data.manifest import DatasetManifest
from t2o.engine.trainer import BEST_CHECKPOINT, LAST_CHECKPOINT, Trainer, resolve_device
from t2o.translators import StubTranslator


def _config(**train: object) -> Config:
    return Config.load(
        overrides={"train": {"batch_size": 2, "workers": 0, "epochs_per_stage": 2, **train}}
    )


# --------------------------------------------------------------------------- resolve_device


def test_resolve_device_accepts_a_bare_digit() -> None:
    # torch.device("0") itself raises -- ultralytics' own convention accepts a bare digit,
    # but a real torch.device needs the "cuda:" prefix.
    assert resolve_device("0") == torch.device("cuda:0")


def test_resolve_device_passes_through_an_explicit_spelling() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda:0") == torch.device("cuda:0")


def test_resolve_device_defaults_to_whatever_is_available() -> None:
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_device(None) == expected


def test_train_returns_epoch_stats_per_epoch(data_yaml: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    config = _config()
    trainer = Trainer(config, manifest, StubTranslator(hidden_channels=4), tmp_path / "run")

    history = trainer.train()

    assert len(history) == config.train.epochs_per_stage
    for i, stats in enumerate(history):
        assert stats.epoch == i
        assert "loss_l2" in stats.train_losses
        assert math.isfinite(stats.val_loss)


def test_checkpoints_are_written(data_yaml: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    run_dir = tmp_path / "run"
    trainer = Trainer(_config(), manifest, StubTranslator(hidden_channels=4), run_dir)

    trainer.train()

    assert (run_dir / LAST_CHECKPOINT).is_file()
    assert (run_dir / BEST_CHECKPOINT).is_file()


def test_resume_continues_from_saved_epoch_with_matching_weights(
    data_yaml: Path, tmp_path: Path
) -> None:
    manifest = DatasetManifest.load(data_yaml)
    run_dir = tmp_path / "run"
    config = _config(epochs_per_stage=2)

    first = Trainer(config, manifest, StubTranslator(hidden_channels=4), run_dir)
    first.train()

    second_translator = StubTranslator(hidden_channels=4)
    second = Trainer(config, manifest, second_translator, run_dir)
    second.resume()

    assert second.start_epoch == 2
    for key, value in first.translator.state_dict().items():
        assert torch.equal(value, second_translator.state_dict()[key])


def test_resume_warns_on_config_drift(
    data_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    manifest = DatasetManifest.load(data_yaml)
    run_dir = tmp_path / "run"

    original = Trainer(_config(), manifest, StubTranslator(hidden_channels=4), run_dir)
    original.train()

    drifted_config = Config.load(
        overrides={
            "train": {"batch_size": 2, "workers": 0, "epochs_per_stage": 2},
            "loss": {"gan": 1.0},
        }
    )
    resumed = Trainer(drifted_config, manifest, StubTranslator(hidden_channels=4), run_dir)

    with caplog.at_level(logging.WARNING):
        resumed.resume()

    assert "config_hash" in caplog.text


def test_resume_without_a_checkpoint_raises(data_yaml: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    trainer = Trainer(_config(), manifest, StubTranslator(hidden_channels=4), tmp_path / "run")

    with pytest.raises(FileNotFoundError):
        trainer.resume()


def test_translator_must_implement_the_protocol(data_yaml: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)

    class NotATranslator(nn.Module):
        pass

    with pytest.raises(TypeError, match="Translator protocol"):
        Trainer(_config(), manifest, NotATranslator(), tmp_path / "run")


def test_trainer_passes_task_loss_and_weight_through_to_fit(
    data_yaml: Path, tmp_path: Path
) -> None:
    manifest = DatasetManifest.load(data_yaml)
    calls: list[tuple[object, float]] = []

    class RecordingTranslator(nn.Module):
        def translate(self, batch: TranslationBatch) -> Tensor:
            return torch.zeros_like(batch["visible"])

        def fit(
            self,
            batch: TranslationBatch,
            task_loss: object = None,
            task_weight: float = 0.0,
        ) -> dict[str, float]:
            calls.append((task_loss, task_weight))
            return {"loss_l2": 0.0}

    def fake_task_loss(pred: Tensor, batch: TranslationBatch) -> Tensor:
        return pred.sum()

    trainer = Trainer(
        _config(epochs_per_stage=1),
        manifest,
        RecordingTranslator(),
        tmp_path / "run",
        task_loss=fake_task_loss,
        task_weight=1.5,
    )

    trainer.train()

    assert calls
    assert all(loss is fake_task_loss and weight == 1.5 for loss, weight in calls)


def test_trainer_defaults_to_no_task_loss(data_yaml: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    trainer = Trainer(_config(), manifest, StubTranslator(hidden_channels=4), tmp_path / "run")

    assert trainer.task_loss is None
    assert trainer.task_weight == 0.0


def test_trainer_moves_an_nn_module_task_loss_to_its_device(
    data_yaml: Path, tmp_path: Path
) -> None:
    """Regression test for a real bug: a `DetectionTaskLoss` wraps a frozen detector that
    stays wherever `torch.load` put it (CPU) unless something moves it. On a CUDA box this
    crashes the moment `task_weight > 0` first calls it, with the translator's output on
    `cuda:0` meeting detector weights still on `cpu` -- invisible on this CPU-only dev
    machine unless `Trainer` is checked to actually call `.to()` on it.
    """
    manifest = DatasetManifest.load(data_yaml)

    class RecordingTaskLoss(nn.Module):
        def forward(self, pred: Tensor, batch: TranslationBatch) -> Tensor:
            return pred.sum()

    task_loss = RecordingTaskLoss()
    with patch.object(task_loss, "to", wraps=task_loss.to) as to_spy:
        trainer = Trainer(
            _config(),
            manifest,
            StubTranslator(hidden_channels=4),
            tmp_path / "run",
            task_loss=task_loss,
            task_weight=1.0,
        )

    to_spy.assert_called_once_with(trainer.device)


def test_shuffle_is_deterministic_for_a_given_epoch_seed(data_yaml: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    trainer = Trainer(_config(), manifest, StubTranslator(hidden_channels=4), tmp_path / "run")

    trainer._train_generator.manual_seed(123)
    first_names = next(iter(trainer.train_loader))["names"]

    trainer._train_generator.manual_seed(123)
    second_names = next(iter(trainer.train_loader))["names"]

    assert first_names == second_names
