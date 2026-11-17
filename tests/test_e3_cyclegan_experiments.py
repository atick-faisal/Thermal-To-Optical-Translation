"""CycleGAN as the direct translator on FLIR -- `experiments/e3_cyclegan_{control,loop}.yaml`.

The arms are meant to differ in the detection weight and the run name only. The differ-only test
is what stops any other edit becoming a second, unrecorded difference between them. These are
not E3's four-stage pix2pix pairs (`test_e3_experiments.py`): two stages of 100 epochs, read
against e3f's stages 0 and 1 (cyclegan-direct-flir plan Q1, Q5).

The configs are read through the same parser and override table the runbook's `t2o loop` lines
use, so a flag that would reach a field the files fix fails here, not on the server. The `slow`
test is the first loop run of a 1-channel CycleGAN with the detection term on.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from t2o.cli import build_parser, overrides_from_args
from t2o.config import Backbone, Config
from t2o.translators import build_translator
from t2o.translators.cyclegan import CycleGANTranslator

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
CONTROL = EXPERIMENTS_DIR / "e3_cyclegan_control.yaml"
LOOP = EXPERIMENTS_DIR / "e3_cyclegan_loop.yaml"

# Arm -> (its config, its task weights).
ARMS = {
    "control": (CONTROL, (0.0, 0.0)),
    "loop": (LOOP, (0.0, 1.0)),
}

FLIR = "dataset/processed/flir/data.yaml"


def _load_as_the_runbook_does(path: Path, *flags: str) -> Config:
    """The runbook's campaign line: FLIR, e3f's corpus cuts, seed, card and W&B."""
    args = build_parser().parse_args(
        [
            "loop",
            "--config",
            str(path),
            "--data",
            FLIR,
            "--max-train-images",
            "600",
            "--val-loss-images",
            "153",
            "--workers",
            "4",
            "--resume",
            "--device",
            "cuda:0",
            "--wandb",
            "--group",
            "e3f-cyclegan",
            *flags,
        ]
    )
    return Config.load(args.config, overrides_from_args(args))


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_is_two_cyclegan_stages_at_e3f_length(arm: str) -> None:
    path, task_weights = ARMS[arm]
    config = Config.load(path)

    assert config.translator.backbone == Backbone.CYCLEGAN
    assert config.coupling.task_weights == task_weights
    assert config.train.epochs_per_stage == 100  # plan Q5, e3f's stage length
    assert config.train.batch_size == 1
    assert config.runtime.name == f"e3f-cyclegan-{arm}"


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_reads_flir_thermal_as_grey_with_unpaired_targets(arm: str) -> None:
    path, _ = ARMS[arm]
    data = _load_as_the_runbook_does(path).data

    assert data.manifest == Path(FLIR)
    assert (data.max_train_images, data.val_loss_images) == (600, 153)  # e3f's cuts
    assert data.infrared_channels == 1  # plan Q3
    assert data.unpaired_visible


def test_control_and_loop_differ_only_by_design() -> None:
    """Two fields differ, by design; everything else must be identical.

    `runtime` and `coupling` are compared with their intended difference copied across, so every
    other field in them is still checked rather than skipped along with it.
    """
    control = Config.load(CONTROL)
    loop = Config.load(LOOP)

    assert control.runtime.model_copy(update={"name": loop.runtime.name}) == loop.runtime
    assert (
        control.coupling.model_copy(update={"task_weights": loop.coupling.task_weights})
        == loop.coupling
    )
    assert control.data == loop.data
    assert control.train == loop.train
    assert control.loss == loop.loss
    assert control.detector == loop.detector
    assert control.translator == loop.translator
    assert control.metrics == loop.metrics
    assert control.export == loop.export

    assert control.coupling.task_weights != loop.coupling.task_weights
    assert control.runtime.name != loop.runtime.name
    assert control.config_hash() != loop.config_hash()


def test_the_judge_is_flirs_and_not_the_detector_that_supplies_the_gradient() -> None:
    for path in (CONTROL, LOOP):
        detector = _load_as_the_runbook_does(path).detector

        assert detector.reference.weights == Path("runs/reference-flir-yolo11s/weights/best.pt")
        assert detector.reference.weights != detector.in_loop.weights
        assert detector.reference.weights != detector.evaluation.init_weights


def test_the_probe_changes_only_its_length_weights_and_name() -> None:
    """The dose probe runs the loop file as one weight-1 stage of 50 epochs (record 039's method).

    50, not 25: the schedule decays over the second half of whatever length it is given, so only
    a 50-epoch probe holds the rate constant over the 25 epochs `loss_share.py` reads.
    """
    campaign = _load_as_the_runbook_does(LOOP)
    probe = _load_as_the_runbook_does(
        LOOP, "--epochs", "50", "--task-weights", "1", "--name", "probe-e3f-cyclegan"
    )

    assert probe.train.epochs_per_stage == 50
    assert probe.coupling.task_weights == (1.0,)
    assert (
        probe.model_copy(
            update={
                "train": campaign.train,
                "coupling": campaign.coupling,
                "runtime": campaign.runtime,
            }
        )
        == campaign
    )
    # Outside `runs/e3f-cyclegan-*`, the glob the comparison report reads.
    assert not probe.runtime.name.startswith("e3f-cyclegan-")


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_builds_a_one_channel_cyclegan(arm: str, data_yaml: Path) -> None:
    """Passes `build_translator`'s refusals: no paired loss, a GAN term, identity off at 1 channel.

    The manifest is the synthetic fixture because FLIR exists on the server alone; the step
    count is read from it.
    """
    path, _ = ARMS[arm]
    config = Config.load(path, overrides={"data": {"manifest": str(data_yaml)}})

    assert isinstance(build_translator(config), CycleGANTranslator)


@pytest.mark.slow
def test_loop_config_runs_end_to_end_on_grey_thermal(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """The loop file, shrunk, through `run_loop`: no loop run has used 1-channel CycleGAN yet.

    Only the sizes change. Stage 1 must carry the detection term and stage 0 must not.
    """
    from t2o.engine.loop import run_loop

    weights = str(detector_weights)
    config = Config.load(
        LOOP,
        overrides={
            "data": {"manifest": str(data_yaml)},
            "train": {"epochs_per_stage": 1},
            "translator": {"net_g": "resnet_6blocks", "ngf": 8, "ndf": 8},
            "detector": {
                "in_loop": {"weights": weights},
                "evaluation": {"init_weights": weights},
                "reference": {"weights": weights},
            },
            "runtime": {"device": "cpu", "workers": 0},
        },
    )
    translator = build_translator(config)

    results = run_loop(config, translator, run_dir=tmp_path, train_detector_stages=False)

    assert [r.task_weight for r in results] == [0.0, 1.0]
    assert "loss_det" not in results[0].epochs[-1].train_losses
    assert math.isfinite(results[1].epochs[-1].train_losses["loss_det"])
    assert all(math.isfinite(r.epochs[-1].val_loss) for r in results)
