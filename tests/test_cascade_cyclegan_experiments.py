"""The CycleGAN cascade's two arms -- `experiments/cascade_cyclegan_{control,loop}.yaml`.

The arms are meant to differ in the detection weight, the run name and the stage-1 tree each
reads (plan Q5, matched). The differ-only test is what stops any other edit becoming a second,
unrecorded difference between them. These are not E3 pairs: `test_e3_experiments.py` assumes
E3's four-stage ramp, while these are one stage of 200 epochs (plan Q9).

The configs are read through the same parser and override table the runbook's `t2o loop`
lines use, so a flag that would reach a field the files fix fails here, not on the server. The
end-to-end CycleGAN loop is already `test_cyclegan_translator.py`'s `slow` test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from t2o.cli import build_parser, overrides_from_args
from t2o.config import Backbone, Config
from t2o.translators import build_translator
from t2o.translators.cyclegan import CycleGANTranslator

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
CONTROL = EXPERIMENTS_DIR / "cascade_cyclegan_control.yaml"
LOOP = EXPERIMENTS_DIR / "cascade_cyclegan_loop.yaml"

# Arm -> (its config, the stage-1 run its tree was exported from, its task weights).
ARMS = {
    "control": (CONTROL, "e3b-control-s0", (0.0,)),
    "loop": (LOOP, "e3b-loop-s0", (1.0,)),
}


def _load_as_the_runbook_does(path: Path, *flags: str) -> Config:
    """The runbook's campaign line: weights, workers, resume and card, never --data."""
    args = build_parser().parse_args(
        [
            "loop",
            "--config",
            str(path),
            "--in-loop-weights",
            "optical_best_n_1.pt",
            "--eval-init-weights",
            "optical_best_n_1.pt",
            "--workers",
            "4",
            "--resume",
            "--device",
            "cuda:0",
            *flags,
        ]
    )
    return Config.load(args.config, overrides_from_args(args))


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_is_one_cyclegan_stage_with_its_own_weight(arm: str) -> None:
    path, _, task_weights = ARMS[arm]
    config = Config.load(path)

    assert config.translator.backbone == Backbone.CYCLEGAN
    assert config.coupling.task_weights == task_weights
    assert config.train.epochs_per_stage == 200  # plan Q9
    assert config.train.batch_size == 1  # plan Q11
    assert config.runtime.name == f"cascade-cyclegan-{arm}"


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_reads_its_own_stage_one_tree_as_rgb_with_unpaired_targets(arm: str) -> None:
    path, stage_one_run, _ = ARMS[arm]
    data = _load_as_the_runbook_does(path).data

    assert data.manifest == Path("runs") / stage_one_run / "stage3" / "paired" / "data.yaml"
    assert data.infrared_channels == 3  # ADR-01
    assert data.unpaired_visible  # plan Q6; nothing refuses a cyclegan config without it


def test_control_and_loop_differ_only_by_design() -> None:
    """Three fields differ, by design; everything else must be identical.

    `data` and `runtime` are compared with their intended difference copied across, so every
    other field in them is still checked rather than skipped along with it.
    """
    control = Config.load(CONTROL)
    loop = Config.load(LOOP)

    assert control.data.model_copy(update={"manifest": loop.data.manifest}) == loop.data
    assert control.runtime.model_copy(update={"name": loop.runtime.name}) == loop.runtime
    assert (
        control.coupling.model_copy(update={"task_weights": loop.coupling.task_weights})
        == loop.coupling
    )
    assert control.train == loop.train
    assert control.loss == loop.loss
    assert control.detector == loop.detector
    assert control.translator == loop.translator
    assert control.metrics == loop.metrics
    assert control.export == loop.export

    assert control.data.manifest != loop.data.manifest
    assert control.coupling.task_weights != loop.coupling.task_weights
    assert control.runtime.name != loop.runtime.name
    assert control.config_hash() != loop.config_hash()


def test_the_judge_is_not_the_detector_that_supplies_the_gradient() -> None:
    for path in (CONTROL, LOOP):
        detector = _load_as_the_runbook_does(path).detector

        assert detector.reference.weights is not None
        assert detector.reference.weights != detector.in_loop.weights
        assert detector.reference.weights != detector.evaluation.init_weights


def test_the_probe_changes_only_its_length_and_name() -> None:
    """The dose probe (plan, Dose and probe) runs the loop file at 50 epochs under its own name.

    50, not 25: the schedule decays over the second half of whatever length it is given, so
    only a 50-epoch probe holds the rate constant over the 25 epochs `loss_share.py` reads.
    """
    campaign = _load_as_the_runbook_does(LOOP)
    probe = _load_as_the_runbook_does(LOOP, "--epochs", "50", "--name", "cascade-cyclegan-probe")

    assert probe.train.epochs_per_stage == 50
    assert (
        probe.model_copy(update={"train": campaign.train, "runtime": campaign.runtime}) == campaign
    )
    assert probe.runtime.name != campaign.runtime.name


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_builds_a_three_channel_cyclegan(arm: str, data_yaml: Path) -> None:
    """Passes `build_translator`'s refusals: no paired loss, a GAN term, identity on RGB.

    The manifest is pointed at the synthetic fixture only because the real tree exists on the
    server alone; the step count is read from it.
    """
    path, _, _ = ARMS[arm]
    config = Config.load(path, overrides={"data": {"manifest": str(data_yaml)}})

    translator = build_translator(config)

    assert isinstance(translator, CycleGANTranslator)
