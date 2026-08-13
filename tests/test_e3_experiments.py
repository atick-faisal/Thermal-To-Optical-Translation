"""`experiments/e3_pix2pix_control.yaml` / `e3_pix2pix_loop.yaml` — E3's ablation pair.

The differ-only-by-design test here is not documentation of an intent; it *is* E3's integrity
check. PLAN.md §11 calls E3 the most important experiment in the project, and its entire claim
is that the two arms differ by lambda_det and nothing else -- so a stray hyperparameter edit to
one file is the specific failure that turns a controlled ablation into a confound, and this is
what catches it.

Same fast/slow split as `test_pix2pix_experiments.py`: shape and hash checks need no fixture,
the one end-to-end pass needs the synthetic fixture and real pix2pix modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from t2o.config import Backbone, Config
from t2o.coupling.schedule import build_detection_loss
from t2o.translators import build_translator

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
CONTROL_CONFIG_PATH = EXPERIMENTS_DIR / "e3_pix2pix_control.yaml"
LOOP_CONFIG_PATH = EXPERIMENTS_DIR / "e3_pix2pix_loop.yaml"


def test_control_config_is_four_zero_weight_stages() -> None:
    config = Config.load(CONTROL_CONFIG_PATH)

    assert config.translator.backbone == Backbone.PIX2PIX
    assert config.coupling.task_weights == (0.0, 0.0, 0.0, 0.0)
    assert config.runtime.name == "e3-pix2pix-control"


def test_loop_config_is_the_real_ramp() -> None:
    config = Config.load(LOOP_CONFIG_PATH)

    assert config.translator.backbone == Backbone.PIX2PIX
    assert config.coupling.task_weights == (0.0, 1.0, 2.0, 3.0)
    assert config.runtime.name == "e3-pix2pix-loop"


def test_control_and_loop_configs_differ_only_by_design() -> None:
    """Two fields differ, by design; everything else must be bit-identical.

    `runtime` is compared field by field rather than wholesale because `name` is one of the
    two intended differences -- so the fields that must NOT drift (`workers`, `group`, and the
    rest) are named explicitly instead of being skipped along with it.
    """
    control = Config.load(CONTROL_CONFIG_PATH)
    loop = Config.load(LOOP_CONFIG_PATH)

    assert control.data == loop.data
    assert control.train == loop.train
    assert control.loss == loop.loss
    assert control.detector == loop.detector
    assert control.translator == loop.translator
    assert control.metrics == loop.metrics
    assert control.export == loop.export

    assert control.coupling.grad_scale == loop.coupling.grad_scale
    assert control.coupling.reward_target == loop.coupling.reward_target
    assert control.runtime.device == loop.runtime.device
    assert control.runtime.workers == loop.runtime.workers
    assert control.runtime.wandb == loop.runtime.wandb
    assert control.runtime.wandb_project == loop.runtime.wandb_project
    assert control.runtime.run_dir == loop.runtime.run_dir
    assert control.runtime.group == loop.runtime.group
    assert control.runtime.group is not None  # 12 sibling runs need one navigable group

    assert control.coupling.task_weights != loop.coupling.task_weights
    assert control.runtime.name != loop.runtime.name


def test_the_two_arms_are_budget_matched() -> None:
    """The confound M1.1 identified: a 4-stage loop arm against a 1-stage control compares
    lambda_det against simply training 4x longer. Equal stage counts at equal
    `epochs_per_stage` is what makes the two arms 400 warm-started epochs each.
    """
    control = Config.load(CONTROL_CONFIG_PATH)
    loop = Config.load(LOOP_CONFIG_PATH)

    assert len(control.coupling.task_weights) == len(loop.coupling.task_weights) == 4
    assert control.train.epochs_per_stage == loop.train.epochs_per_stage


def test_stage_zero_is_the_same_condition_in_both_arms() -> None:
    """The free within-experiment null control: paired stage-0 runs differ only by run-to-run
    noise, which is the yardstick the stage-3 effect has to beat.
    """
    control = Config.load(CONTROL_CONFIG_PATH)
    loop = Config.load(LOOP_CONFIG_PATH)

    assert control.coupling.task_weights[0] == loop.coupling.task_weights[0] == 0.0


def test_the_judge_is_not_the_detector_that_supplies_the_gradient() -> None:
    """E3's whole point over M1: a gain measured by the checkpoint that trained the
    translator is not separable from reward hacking. `reference.weights` is a concrete path
    rather than null precisely so it cannot fall back to `evaluation.init_weights`, which on
    the server is the same file as `in_loop.weights`.
    """
    for path in (CONTROL_CONFIG_PATH, LOOP_CONFIG_PATH):
        detector = Config.load(path).detector

        assert detector.reference.weights is not None
        assert detector.reference.weights != detector.in_loop.weights
        assert detector.reference.weights != detector.evaluation.init_weights


def test_the_control_arm_never_constructs_a_detector(detector_weights: Path) -> None:
    """Asserted against `build_detection_loss` directly rather than trusting its docstring.

    Note this is fast even though it is handed real weights: the weight-0 branch returns
    before `FrozenDetector` is ever built, which is the no-op property being tested.
    """
    control = Config.load(CONTROL_CONFIG_PATH)

    for stage in range(len(control.coupling.task_weights)):
        assert build_detection_loss(control.coupling, stage, detector_weights, nc=4) is None


def test_the_loop_arm_couples_in_a_detector_from_stage_one(detector_weights: Path) -> None:
    loop = Config.load(LOOP_CONFIG_PATH)

    assert build_detection_loss(loop.coupling, 0, detector_weights, nc=4) is None
    for stage in (1, 2, 3):
        assert build_detection_loss(loop.coupling, stage, detector_weights, nc=4) is not None


@pytest.mark.parametrize("path", [CONTROL_CONFIG_PATH, LOOP_CONFIG_PATH])
def test_config_hash_is_stable_across_loads(path: Path) -> None:
    assert Config.load(path).config_hash() == Config.load(path).config_hash()


@pytest.mark.slow
def test_control_config_runs_end_to_end_on_the_synthetic_fixture(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """The control arm specifically -- `test_pix2pix_experiments.py` already covers a
    ramped 4-stage file end to end, and the untested shape here is the all-zero ramp
    actually completing four stages rather than short-circuiting somewhere.
    """
    from t2o.engine.loop import run_loop

    config = Config.load(
        CONTROL_CONFIG_PATH,
        overrides={
            "data": {"manifest": str(data_yaml)},
            "train": {"epochs_per_stage": 1, "batch_size": 2},
            "translator": {"net_g": "resnet_6blocks"},  # smaller, faster on CPU
            "loss": {"gan": 0.0, "lpips": 0.0},  # skip building D/LPIPS -- shape check only
            "detector": {
                "in_loop": {"weights": str(detector_weights)},
                "evaluation": {"init_weights": str(detector_weights)},
                "reference": {"weights": str(detector_weights)},
            },
            "runtime": {"run_dir": str(tmp_path), "name": "run", "workers": 0},
        },
    )
    translator = build_translator(config)

    results = run_loop(config, translator, train_detector_stages=False)

    assert len(results) == 4
    assert all(result.detector is None for result in results)
