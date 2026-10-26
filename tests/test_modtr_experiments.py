"""ModTr's arms -- `experiments/modtr_{custom,flir}.yaml` -- against the loop cells they score.

Margin compares ModTr with the loop on one detector and one corpus (margin-gate ADR-01), so a
ModTr file that drifted from its paired loop config on data or detectors would make the
comparison a detector or corpus comparison without a word. These tests are that check, the
way `test_e3_experiments.py` checks E3's two arms.

FLIR's loop cell, e3f, had no YAML: it ran `e3_pix2pix_loop.yaml` with command-line overrides,
and `E3F_OVERRIDES` transcribes them from record 016's resolved config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from t2o.config import Backbone, Config
from t2o.translators import ModTrTranslator, build_translator

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
MODTR_CUSTOM = EXPERIMENTS_DIR / "modtr_custom.yaml"
MODTR_FLIR = EXPERIMENTS_DIR / "modtr_flir.yaml"
PIX2PIX_LOOP = EXPERIMENTS_DIR / "e3_pix2pix_loop.yaml"
TURBO_LOOP = EXPERIMENTS_DIR / "e3_turbo_loop.yaml"

# Record 016, block 3: every e3f key that differs from `e3_pix2pix_loop.yaml` outside `runtime`.
E3F_OVERRIDES = {
    "data": {
        "manifest": "dataset/processed/flir/data.yaml",
        "max_train_images": 600,
        "val_loss_images": 153,
    },
    "detector": {
        "in_loop": {"weights": "runs/inloop-flir-yolo11n/weights/best.pt"},
        "evaluation": {"init_weights": "runs/inloop-flir-yolo11n/weights/best.pt"},
        "reference": {"weights": "runs/reference-flir-yolo11s/weights/best.pt"},
    },
}

# ModTr's file, then every loop cell it is scored against: e3b and e3t on the custom set,
# e3f on FLIR. The first loop config is the one the ModTr file was copied from.
_CELLS: list[tuple[Path, list[Config]]] = [
    (MODTR_CUSTOM, [Config.load(PIX2PIX_LOOP), Config.load(TURBO_LOOP)]),
    (MODTR_FLIR, [Config.load(PIX2PIX_LOOP, overrides=E3F_OVERRIDES)]),
]
CELLS = [pytest.param(*cell, id=cell[0].stem) for cell in _CELLS]

# ADR-04's budget, as each cell's own calibration run set it (ADR-05).
CALIBRATED_EPOCHS = {
    MODTR_CUSTOM: 400,  # record 028: the trigger did not fire, so the loop's 4 x 100 stands
    MODTR_FLIR: 339,  # record 029: the trigger fired
}


@pytest.mark.parametrize(("modtr_path", "loops"), CELLS)
def test_data_and_detectors_match_every_paired_loop_cell(
    modtr_path: Path, loops: list[Config]
) -> None:
    """The same in-loop yolo11n, the same reference yolo11s, the same corpus."""
    modtr = Config.load(modtr_path)

    for loop in loops:
        assert modtr.data == loop.data
        assert modtr.detector == loop.detector
        assert modtr.metrics == loop.metrics
        assert modtr.export == loop.export


@pytest.mark.parametrize(("modtr_path", "loops"), CELLS)
def test_train_differs_from_the_copied_loop_only_by_modtrs_recipe(
    modtr_path: Path, loops: list[Config]
) -> None:
    """Epochs and LR are ModTr's (ADR-04); every other training knob is the loop's."""
    modtr, loop = Config.load(modtr_path).train, loops[0].train

    assert modtr.batch_size == loop.batch_size
    assert modtr.lr_gamma == loop.lr_gamma
    assert modtr.amp == loop.amp
    assert modtr.amp_dtype == loop.amp_dtype
    assert modtr.hflip == loop.hflip
    assert modtr.crop == loop.crop
    assert modtr.seed == loop.seed

    assert modtr.lr == 1.0e-4  # upstream `main.py:53`, not the pix2pix loop's 2.0e-4
    assert modtr.lr != loop.lr


@pytest.mark.parametrize(("modtr_path", "loops"), CELLS)
def test_modtr_trains_its_calibrated_budget_in_one_detection_only_stage(
    modtr_path: Path, loops: list[Config]
) -> None:
    """Plan Q1's single stage, ADR-04's calibrated budget, ADR-03's unhinged objective.

    Calibration can only lower the loop's 4 x 100 epochs, never raise it.
    """
    modtr = Config.load(modtr_path)

    assert modtr.translator.backbone == Backbone.MODTR
    assert modtr.coupling.task_weights == (1.0,)
    assert modtr.coupling.grad_scale == 1.0
    assert modtr.coupling.reward_target is None
    assert (modtr.loss.l2, modtr.loss.lpips, modtr.loss.gan) == (0.0, 0.0, 0.0)
    assert modtr.train.epochs_per_stage == CALIBRATED_EPOCHS[modtr_path]
    for loop in loops:
        loop_epochs = loop.train.epochs_per_stage * len(loop.coupling.task_weights)
        assert loop_epochs == 400
        assert modtr.train.epochs_per_stage <= loop_epochs


@pytest.mark.parametrize("modtr_path", [MODTR_CUSTOM, MODTR_FLIR])
def test_build_translator_accepts_the_config(modtr_path: Path, data_yaml: Path) -> None:
    """`build_translator` refuses a ModTr config with a reconstruction weight or a 0 stage."""
    config = Config.load(
        modtr_path,
        overrides={
            "data": {"manifest": str(data_yaml), "max_train_images": None},
            "translator": {"pretrained": False},  # no ImageNet download on the Mac
        },
    )

    assert isinstance(build_translator(config), ModTrTranslator)


def test_every_modtr_cell_is_a_distinct_run() -> None:
    """A copied file whose name was not changed would write into a loop campaign's run dirs."""
    configs = [Config.load(path) for path in (MODTR_CUSTOM, MODTR_FLIR, PIX2PIX_LOOP, TURBO_LOOP)]

    assert len({config.config_hash() for config in configs}) == len(configs)
    assert len({config.runtime.name for config in configs}) == len(configs)
    assert len({config.runtime.group for config in configs}) == len(configs)
