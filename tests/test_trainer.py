"""`engine/trainer.py`'s epoch loop, checkpointing, and resume.

Built on the synthetic `data_yaml` fixture (`tests/conftest.py`): 5 train / 3 val
images, disjoint stems. Config overrides keep every run tiny and CPU-fast.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from t2o.config import Config
from t2o.data.manifest import DatasetManifest
from t2o.engine.trainer import BEST_CHECKPOINT, LAST_CHECKPOINT, Trainer
from t2o.translators import StubTranslator


def _config(**train: object) -> Config:
    return Config.load(
        overrides={"train": {"batch_size": 2, "workers": 0, "epochs_per_stage": 2, **train}}
    )


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


def test_shuffle_is_deterministic_for_a_given_epoch_seed(data_yaml: Path, tmp_path: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    trainer = Trainer(_config(), manifest, StubTranslator(hidden_channels=4), tmp_path / "run")

    trainer._train_generator.manual_seed(123)
    first_names = next(iter(trainer.train_loader))["names"]

    trainer._train_generator.manual_seed(123)
    second_names = next(iter(trainer.train_loader))["names"]

    assert first_names == second_names
