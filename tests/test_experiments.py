"""`experiments/smoke.yaml` — the tiny-scale mirror of a real experiment config.

Only this file's own shape is verified here (loads, is small, is hash-stable, and actually
drives one fast end-to-end pass on the synthetic fixture). The fuller end-to-end contract --
metrics.json across every stage, resume, same-config-same-seed reproducibility -- is a
separate TASKS.md item and lives in its own test module once that step lands.
"""

from __future__ import annotations

from pathlib import Path

from t2o.config import Backbone, Config
from t2o.engine.loop import run_loop
from t2o.translators import build_translator

SMOKE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiments" / "smoke.yaml"


def test_smoke_config_loads_and_is_tiny() -> None:
    config = Config.load(SMOKE_CONFIG_PATH)

    assert config.train.epochs_per_stage == 1
    assert config.train.batch_size == 2
    assert config.translator.backbone == Backbone.STUB
    assert len(config.coupling.task_weights) == 2
    assert config.runtime.device == "cpu"


def test_smoke_config_hash_is_stable_across_loads() -> None:
    first = Config.load(SMOKE_CONFIG_PATH).config_hash()
    second = Config.load(SMOKE_CONFIG_PATH).config_hash()

    assert first == second


def test_smoke_config_runs_end_to_end_on_the_synthetic_fixture(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """The translator-only path (`train_detector_stages=False`) — fast, no ultralytics
    training. Overrides mirror how a real experiment config resolves machine-specific paths
    (PLAN.md §9): `data.manifest` and the detector weights only exist per-machine, so the
    committed file's own values are placeholders every caller is expected to override.
    """
    config = Config.load(
        SMOKE_CONFIG_PATH,
        overrides={
            "data": {"manifest": str(data_yaml)},
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
