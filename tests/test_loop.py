"""`engine/loop.py`'s staged alternation of translator training and detector fine-tune.

Built on the synthetic `data_yaml`/`detector_weights` fixtures (`tests/conftest.py`). The
fast test exercises the translator-only path (`train_detector_stages=False`) but still
wires a real `FrozenDetector` through the coupling loss on the stage with a positive
`task_weight` -- constructing that detector is cheap (one model load, no training), unlike
`train_detector`'s full ultralytics training loop. The slow test is the first end-to-end
exercise of `train_detector` itself, deferred from M0.4 to this milestone (TASKS.md).

`resume`/reproducibility tests below close out the M0.8 "Tests" bullet: `run_loop`'s
`resume=True` path and `translators.build_translator`'s seed-determined init (both added
alongside these tests) are what make "same config+seed twice -> identical metrics" and
"resume produces identical state" true rather than aspirational.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from t2o.config import Config
from t2o.engine.loop import _resolve_reference_weights, run_loop
from t2o.translators import StubTranslator, build_translator


def _config(
    data_yaml: Path, detector_weights: Path, task_weights: tuple[float, ...] = (0.0, 1.0)
) -> Config:
    return Config.load(
        overrides={
            "data": {"manifest": str(data_yaml)},
            "train": {"batch_size": 2, "epochs_per_stage": 1},
            "coupling": {"task_weights": list(task_weights)},
            "detector": {
                "in_loop": {"weights": str(detector_weights)},
                "evaluation": {"init_weights": str(detector_weights), "epochs": 1, "batch": 2},
            },
            "runtime": {"device": "cpu", "workers": 0},
        }
    )


def test_translator_only_loop_runs_every_stage_and_wires_the_coupling_loss(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    config = _config(data_yaml, detector_weights)
    translator = StubTranslator(hidden_channels=4)
    before = next(translator.parameters()).clone()

    results = run_loop(config, translator, run_dir=tmp_path / "run", train_detector_stages=False)

    assert [r.stage for r in results] == [0, 1]
    assert [r.task_weight for r in results] == [0.0, 1.0]
    assert all(r.detector is None for r in results)
    assert not torch.equal(before, next(translator.parameters()))

    # task_weights[0] == 0 must stay a clean no-op (M0.7's contract, exercised end-to-end
    # here): no coupling term at all, so no "loss_det" key. Stage 1's positive weight must
    # actually reach StubTranslator.fit() and add one.
    assert "loss_det" not in results[0].epochs[-1].train_losses
    assert "loss_det" in results[1].epochs[-1].train_losses

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(metrics) == 2
    assert [m["task_weight"] for m in metrics] == [0.0, 1.0]
    assert all(m["detector"] is None for m in metrics)


@pytest.mark.slow
def test_full_loop_fine_tunes_a_detector_every_stage(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    config = _config(data_yaml, detector_weights)
    translator = StubTranslator(hidden_channels=4)

    results = run_loop(config, translator, run_dir=tmp_path / "run")

    assert len(results) == 2
    for result in results:
        assert result.detector is not None
        assert result.detector.weights.is_file()
        assert 0.0 <= result.detector.map50 <= 1.0
        # The gate metric (M1): the reference detector scored on the same translated export,
        # with no fine-tuning at all. Recorded alongside the adapted number, never instead
        # of it -- see engine/loop.py's docstring for why only this arm answers the gate.
        assert result.zero_shot is not None
        assert 0.0 <= result.zero_shot.map50 <= 1.0
        assert 0.0 <= result.zero_shot.map50_95 <= 1.0
        # Fidelity on the same export. Recorded beside zero_shot because the pair is the
        # reward-hacking check -- neither number diagnoses it alone (PLAN.md §8).
        assert result.fidelity is not None
        assert math.isfinite(result.fidelity.lpips)
        assert math.isfinite(result.fidelity.fid)

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(metrics) == 2
    assert metrics[0]["detector"]["weights"]
    assert metrics[1]["detector"]["weights"]
    for entry in metrics:
        assert entry["zero_shot"] is not None
        assert 0.0 <= entry["zero_shot"]["map50"] <= 1.0
        assert "per_class_ap50" in entry["zero_shot"]
        assert entry["fidelity"] is not None
        assert "lpips" in entry["fidelity"]


def test_resume_on_a_fresh_run_dir_behaves_like_a_normal_run(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """No `metrics.json` yet -- `resume=True` must be a silent no-op, not an error."""
    config = _config(data_yaml, detector_weights)
    translator = build_translator(config)

    results = run_loop(
        config, translator, run_dir=tmp_path / "run", train_detector_stages=False, resume=True
    )

    assert [r.stage for r in results] == [0, 1]


def test_resume_skips_completed_stages_and_restores_warm_started_weights(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """An unbroken 2-stage run vs. one interrupted right after stage 0 and then resumed must
    land on identical final translator weights and identical per-epoch metrics -- the literal
    "resume produces identical state" (TASKS.md). `build_translator` gives both translators
    the same seed-determined initial weights (M0.8, this same step) so the comparison is
    apples to apples.
    """
    full_config = _config(data_yaml, detector_weights)
    stage0_only_config = _config(data_yaml, detector_weights, task_weights=(0.0,))

    unbroken_translator = build_translator(full_config)
    unbroken_results = run_loop(
        full_config,
        unbroken_translator,
        run_dir=tmp_path / "unbroken",
        train_detector_stages=False,
    )

    resumed_translator = build_translator(full_config)  # same seed -> identical initial weights
    run_loop(
        stage0_only_config,
        resumed_translator,
        run_dir=tmp_path / "resumed",
        train_detector_stages=False,
    )  # simulates a crash right after stage 0
    resumed_results = run_loop(
        full_config,
        resumed_translator,
        run_dir=tmp_path / "resumed",
        train_detector_stages=False,
        resume=True,
    )

    assert [r.stage for r in resumed_results] == [r.stage for r in unbroken_results] == [0, 1]
    for key, value in unbroken_translator.state_dict().items():
        assert torch.equal(value, resumed_translator.state_dict()[key])

    unbroken_metrics = json.loads((tmp_path / "unbroken" / "metrics.json").read_text())
    resumed_metrics = json.loads((tmp_path / "resumed" / "metrics.json").read_text())
    assert unbroken_metrics == resumed_metrics


def test_running_the_same_config_and_seed_twice_yields_identical_metrics(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    config = _config(data_yaml, detector_weights)

    translator_a = build_translator(config)
    run_loop(config, translator_a, run_dir=tmp_path / "a", train_detector_stages=False)

    translator_b = build_translator(config)
    run_loop(config, translator_b, run_dir=tmp_path / "b", train_detector_stages=False)

    metrics_a = json.loads((tmp_path / "a" / "metrics.json").read_text())
    metrics_b = json.loads((tmp_path / "b" / "metrics.json").read_text())
    assert metrics_a == metrics_b
    for key, value in translator_a.state_dict().items():
        assert torch.equal(value, translator_b.state_dict()[key])


def test_a_run_is_unaffected_by_the_ambient_rng_state_it_inherits(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """M1.2 step 2: each stage reseeds from `train.seed + stage`, so a stage's RNG stream
    does not depend on how many draws anything before it happened to make. Without that,
    the two runs below diverge -- and on a real 4-stage run the "anything before it" is
    ultralytics' detector fine-tune, whose draw count is entirely outside our control.
    """
    config = _config(data_yaml, detector_weights)

    translator_a = build_translator(config)
    torch.rand(17)  # an arbitrary amount of ambient RNG consumed before the run starts
    run_loop(config, translator_a, run_dir=tmp_path / "a", train_detector_stages=False)

    translator_b = build_translator(config)
    torch.rand(3)  # ...and a different amount before the second
    run_loop(config, translator_b, run_dir=tmp_path / "b", train_detector_stages=False)

    assert json.loads((tmp_path / "a" / "metrics.json").read_text()) == json.loads(
        (tmp_path / "b" / "metrics.json").read_text()
    )
    for key, value in translator_a.state_dict().items():
        assert torch.equal(value, translator_b.state_dict()[key])


def test_reference_weights_default_to_the_evaluation_bootstrap(
    data_yaml: Path, detector_weights: Path
) -> None:
    config = _config(data_yaml, detector_weights)

    assert config.detector.reference.weights is None
    assert _resolve_reference_weights(config) == detector_weights


def test_an_explicit_reference_detector_wins_over_the_fallback(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    independent = tmp_path / "independent.pt"
    config = Config.load(
        overrides={
            "data": {"manifest": str(data_yaml)},
            "detector": {
                "in_loop": {"weights": str(detector_weights)},
                "evaluation": {"init_weights": str(detector_weights)},
                "reference": {"weights": str(independent)},
            },
        }
    )

    assert _resolve_reference_weights(config) == independent


def test_a_self_graded_reference_detector_warns_only_when_the_loop_couples(
    data_yaml: Path, detector_weights: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The reference detector falling back onto the in-loop checkpoint is the realistic
    configuration while only one optical detector exists, so it must not fail -- but a
    `task_weight > 0` stage scored by the detector that trained it is not clean evidence,
    and that has to be visible in the log rather than inferred from the config later.
    """
    coupled = _config(data_yaml, detector_weights, task_weights=(0.0, 1.0))
    with caplog.at_level("WARNING"):
        _resolve_reference_weights(coupled)
    assert "same checkpoint as the in-loop detector" in caplog.text

    caplog.clear()
    baseline = _config(data_yaml, detector_weights, task_weights=(0.0,))
    with caplog.at_level("WARNING"):
        _resolve_reference_weights(baseline)
    assert caplog.text == ""


def test_metrics_json_written_before_the_zero_shot_arm_existed_still_resumes(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """A long server run must not die on `KeyError: 'zero_shot'` halfway through."""
    run_dir = tmp_path / "run"
    config = _config(data_yaml, detector_weights)
    translator = build_translator(config)
    run_loop(
        _config(data_yaml, detector_weights, task_weights=(0.0,)),
        translator,
        run_dir=run_dir,
        train_detector_stages=False,
    )

    legacy = json.loads((run_dir / "metrics.json").read_text())
    for entry in legacy:
        del entry["zero_shot"]
        del entry["fidelity"]
    (run_dir / "metrics.json").write_text(json.dumps(legacy))

    results = run_loop(
        config, translator, run_dir=run_dir, train_detector_stages=False, resume=True
    )

    assert [r.stage for r in results] == [0, 1]
    assert results[0].zero_shot is None
    assert results[0].fidelity is None


@pytest.mark.slow
def test_resume_continues_through_a_completed_detector_stage(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """The one scenario the fast resume tests above don't reach: `eval_weights` carrying
    forward from a stage whose detector fine-tune already completed and was recorded.
    """
    full_config = _config(data_yaml, detector_weights)
    stage0_only_config = _config(data_yaml, detector_weights, task_weights=(0.0,))
    translator = build_translator(full_config)

    run_loop(stage0_only_config, translator, run_dir=tmp_path / "run")  # simulated crash point
    results = run_loop(full_config, translator, run_dir=tmp_path / "run", resume=True)

    assert [r.stage for r in results] == [0, 1]
    for result in results:
        assert result.detector is not None
        assert result.detector.weights.is_file()

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(metrics) == 2
