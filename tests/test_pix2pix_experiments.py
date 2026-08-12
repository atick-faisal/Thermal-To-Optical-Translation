"""`experiments/pix2pix_baseline.yaml` / `pix2pix_loop.yaml` — M1's two real-scale configs.

Same split as `test_experiments.py`'s smoke coverage: shape/hash checks are fast and need no
overrides; the one end-to-end pass needs the synthetic fixture and real pix2pix modules, so
it's `slow`, mirroring `test_pix2pix_translator.py`'s own full-loop test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from t2o.config import Backbone, Config
from t2o.translators import build_translator

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
BASELINE_CONFIG_PATH = EXPERIMENTS_DIR / "pix2pix_baseline.yaml"
LOOP_CONFIG_PATH = EXPERIMENTS_DIR / "pix2pix_loop.yaml"


def test_baseline_config_loads_with_a_single_zero_weight_stage() -> None:
    config = Config.load(BASELINE_CONFIG_PATH)

    assert config.translator.backbone == Backbone.PIX2PIX
    assert config.coupling.task_weights == (0.0,)
    assert config.loss.gan > 0.0  # the full GAN recipe, not a GAN-less regression stand-in
    assert config.runtime.name == "pix2pix-baseline"


def test_loop_config_loads_with_the_real_four_stage_ramp() -> None:
    config = Config.load(LOOP_CONFIG_PATH)

    assert config.translator.backbone == Backbone.PIX2PIX
    assert config.coupling.task_weights == (0.0, 1.0, 2.0, 3.0)
    assert config.runtime.name == "pix2pix-loop"


def test_baseline_and_loop_configs_differ_only_by_design() -> None:
    """The two files are meant to be identical except `coupling.task_weights` and
    `runtime.name` (each file's own header comment) -- assert that stays true rather than
    letting the two silently drift apart on every other field.
    """
    baseline = Config.load(BASELINE_CONFIG_PATH)
    loop = Config.load(LOOP_CONFIG_PATH)

    assert baseline.data == loop.data
    assert baseline.train == loop.train
    assert baseline.loss == loop.loss
    assert baseline.detector == loop.detector
    assert baseline.translator == loop.translator
    assert baseline.metrics == loop.metrics
    assert baseline.export == loop.export
    assert baseline.coupling.task_weights != loop.coupling.task_weights
    assert baseline.runtime.name != loop.runtime.name


@pytest.mark.parametrize("path", [BASELINE_CONFIG_PATH, LOOP_CONFIG_PATH])
def test_config_hash_is_stable_across_loads(path: Path) -> None:
    first = Config.load(path).config_hash()
    second = Config.load(path).config_hash()

    assert first == second


@pytest.mark.slow
def test_loop_config_runs_end_to_end_on_the_synthetic_fixture(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """Proves the tracked file itself -- not just `Pix2PixTranslator` in isolation
    (`test_pix2pix_translator.py`) -- drives `run_loop` correctly end to end. Overrides mirror
    `test_experiments.py`'s own pattern: real paths only exist per-machine, and epoch/batch
    counts shrink to fixture scale (PLAN.md §9's smoke-fixture discipline).
    """
    from t2o.engine.loop import run_loop

    config = Config.load(
        LOOP_CONFIG_PATH,
        overrides={
            "data": {"manifest": str(data_yaml)},
            "train": {"epochs_per_stage": 1, "batch_size": 2, "workers": 0},
            "translator": {"net_g": "resnet_6blocks"},  # smaller, faster on CPU
            "loss": {"gan": 0.0, "lpips": 0.0},  # skip building D/LPIPS -- shape check only
            "detector": {
                "in_loop": {"weights": str(detector_weights)},
                "evaluation": {"init_weights": str(detector_weights)},
            },
            "runtime": {"run_dir": str(tmp_path), "name": "run"},
        },
    )
    translator = build_translator(config)

    results = run_loop(config, translator, train_detector_stages=False)

    assert len(results) == len(config.coupling.task_weights)
    assert all(result.detector is None for result in results)
