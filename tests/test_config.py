from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from t2o.config import (
    AmpDtype,
    Backbone,
    Config,
    ConfigError,
    StubTranslatorConfig,
)
from t2o.imaging import Normalize


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


# --------------------------------------------------------------------------- defaults


def test_defaults_are_the_documented_ones() -> None:
    config = Config.load()

    assert config.coupling.task_weights == (0.0, 1.0, 2.0, 3.0)
    assert config.train.amp_dtype is AmpDtype.BFLOAT16
    assert config.export.normalize is Normalize.CLAMP
    assert config.runtime.path == Path("runs/t2o")
    # Nothing about the dataset lives here beyond the manifest path.
    assert config.data.manifest == Path("dataset/yolo_rgbt/data.yaml")
    # The E8 hook defaults to the full annotation set.
    assert config.data.annotation_fraction == pytest.approx(1.0)


def test_the_three_detector_roles_are_separate_objects() -> None:
    # PLAN.md invariant 7 encoded structurally: nothing can conflate them by accident.
    # `reference` is the third role (M1's gate metric) -- never trained, never in the loop.
    config = Config.load(overrides={"detector": {"in_loop": {"weights": "frozen.pt"}}})
    assert config.detector.in_loop.weights == Path("frozen.pt")
    assert config.detector.evaluation.init_weights == Path("yolo11n.pt")
    # Defaults to null, which engine/loop.py resolves to evaluation.init_weights. Stated as
    # a test because that fallback is what makes the field optional in every config on disk.
    assert config.detector.reference.weights is None


def test_the_reference_detector_participates_in_the_config_hash() -> None:
    # It changes the reported gate number, so it is experiment identity, not invocation
    # detail -- the same reasoning that put MetricsConfig in the hash (M0.5).
    baseline = Config.load().config_hash()
    swapped = Config.load(overrides={"detector": {"reference": {"weights": "independent.pt"}}})
    assert swapped.config_hash() != baseline


# --------------------------------------------------------------------------- loading


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "c.yaml",
        {
            "train": {"epochs_per_stage": 3, "crop": [64, 64]},
            "coupling": {"task_weights": [0, 1]},
            "export": {"normalize": "per_image"},
            "runtime": {"device": "cpu", "name": "exp1"},
        },
    )
    config = Config.load(path)

    assert config.train.epochs_per_stage == 3
    assert config.train.crop == (64, 64)
    assert config.coupling.task_weights == (0.0, 1.0)
    assert config.export.normalize is Normalize.PER_IMAGE
    assert config.runtime.device == "cpu"
    # Untouched keys keep their defaults rather than being dropped by the merge.
    assert config.train.lr == pytest.approx(1e-4)


def test_overrides_take_precedence_over_the_file(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", {"train": {"lr": 0.5, "batch_size": 4}})

    config = Config.load(path, overrides={"train": {"lr": 0.001}})
    assert config.train.lr == pytest.approx(0.001)  # override wins
    assert config.train.batch_size == 4  # file value survives


def test_empty_yaml_file_yields_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert Config.load(path) == Config.load()


def test_top_level_yaml_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        Config.load(path)


def test_config_is_frozen() -> None:
    config = Config.load()
    with pytest.raises(ValueError, match="frozen"):
        config.train.lr = 1.0  # pyright: ignore[reportAttributeAccessIssue]


# --------------------------------------------------------------------------- rejection


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="trian: Extra inputs are not permitted"):
        Config.load(overrides={"trian": {}})


def test_unknown_nested_key_reports_its_dotted_location() -> None:
    with pytest.raises(ConfigError, match=r"train\.learning_rate: Extra inputs"):
        Config.load(overrides={"train": {"learning_rate": 1e-3}})


def test_multiple_unknown_keys_are_all_reported() -> None:
    # A config with three typos should fail once, not over three consecutive runs.
    with pytest.raises(ConfigError) as exc:
        Config.load(overrides={"train": {"foo": 1, "bar": 2}, "baz": 3})
    message = str(exc.value)
    assert "3 errors" in message
    assert "train.foo" in message
    assert "train.bar" in message
    assert "baz" in message


def test_invalid_enum_value_lists_the_alternatives() -> None:
    with pytest.raises(ConfigError, match="'clamp' or 'per_image'"):
        Config.load(overrides={"export": {"normalize": "batch"}})


def test_bad_scalar_type_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"train\.batch_size"):
        Config.load(overrides={"train": {"batch_size": "eight"}})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"coupling": {"task_weights": []}}, "at least one stage"),
        ({"coupling": {"task_weights": [0, -1]}}, "non-negative"),
        ({"train": {"hflip": 1.5}}, "probability"),
        ({"train": {"epochs_per_stage": 0}}, ">= 1"),
        ({"data": {"annotation_fraction": 0.0}}, r"\(0, 1\]"),
        ({"data": {"annotation_fraction": 1.5}}, r"\(0, 1\]"),
        ({"detector": {"in_loop": {"imgsz": 641}}}, "stride 32"),
    ],
)
def test_validators_fire(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        Config.load(overrides=overrides)


# --------------------------------------------------------------------------- coercion


def test_paths_are_coerced() -> None:
    config = Config.load(
        overrides={
            "data": {"manifest": "/tmp/x/data.yaml"},
            "detector": {"in_loop": {"weights": "a.pt"}},
        }
    )
    assert config.data.manifest == Path("/tmp/x/data.yaml")
    assert config.detector.in_loop.weights == Path("a.pt")


def test_crop_null_means_full_frames() -> None:
    assert Config.load(overrides={"train": {"crop": None}}).train.crop is None


def test_reward_target_null_means_unbounded_minimisation() -> None:
    assert Config.load().coupling.reward_target is None
    assert Config.load(overrides={"coupling": {"reward_target": 2.0}}).coupling.reward_target == 2.0


# --------------------------------------------------------- translator discriminated union


def test_stub_backbone_builds_the_stub_model() -> None:
    config = Config.load(overrides={"translator": {"backbone": "stub", "hidden_channels": 4}})
    assert isinstance(config.translator, StubTranslatorConfig)
    assert config.translator.backbone is Backbone.STUB
    assert config.translator.hidden_channels == 4


def test_unknown_backbone_lists_the_valid_discriminators() -> None:
    with pytest.raises(ConfigError, match="does not match any of the expected tags"):
        Config.load(overrides={"translator": {"backbone": "pix2pix_turbo"}})


def test_pix2pix_backbone_builds_the_pix2pix_model() -> None:
    from t2o.config import Pix2PixTranslatorConfig

    config = Config.load(overrides={"translator": {"backbone": "pix2pix", "ngf": 32}})
    assert isinstance(config.translator, Pix2PixTranslatorConfig)
    assert config.translator.backbone is Backbone.PIX2PIX
    assert config.translator.ngf == 32


def test_a_partial_translator_section_must_still_name_its_backbone() -> None:
    # Omitting `translator` entirely keeps the default; touching it at all requires the
    # tag. That is the point of invariant 3 -- which backbone is running is never implicit.
    with pytest.raises(ConfigError, match="Unable to extract tag"):
        Config.load(overrides={"translator": {"hidden_channels": 4}})


def test_unknown_key_under_a_backbone_is_rejected() -> None:
    # The union member the key was checked against is named in the reported location.
    with pytest.raises(ConfigError, match=r"translator\.stub\.ngf"):
        Config.load(overrides={"translator": {"backbone": "stub", "ngf": 64}})


# --------------------------------------------------------------------------- hashing


def test_hash_is_stable_across_key_reordering(tmp_path: Path) -> None:
    a = write_yaml(tmp_path / "a.yaml", {"train": {"lr": 0.5, "seed": 3}, "loss": {"gan": 1.0}})
    b = write_yaml(tmp_path / "b.yaml", {"loss": {"gan": 1.0}, "train": {"seed": 3, "lr": 0.5}})
    assert Config.load(a).config_hash() == Config.load(b).config_hash()


def test_hash_changes_when_a_scientific_value_changes() -> None:
    baseline = Config.load().config_hash()
    assert Config.load(overrides={"train": {"lr": 0.5}}).config_hash() != baseline
    assert Config.load(overrides={"coupling": {"grad_scale": 1.0}}).config_hash() != baseline
    translator = {"backbone": "stub", "hidden_channels": 4}
    assert Config.load(overrides={"translator": translator}).config_hash() != baseline
    # The seed is part of experiment identity, which is why it lives under `train`.
    assert Config.load(overrides={"train": {"seed": 1}}).config_hash() != baseline


@pytest.mark.parametrize(
    "runtime",
    [
        {"device": "cuda:0"},
        {"name": "some-other-run"},
        {"run_dir": "/scratch/runs"},
        {"wandb": True},
        {"wandb_project": "elsewhere"},
    ],
)
def test_hash_ignores_runtime(runtime: dict[str, Any]) -> None:
    # The same experiment on two GPUs under two names is the same experiment. Diverges
    # from Clean-SeAFusion, which hashed everything and warned on a renamed run.
    assert Config.load(overrides={"runtime": runtime}).config_hash() == Config.load().config_hash()


def test_hash_is_short_and_hex() -> None:
    digest = Config.load().config_hash()
    assert len(digest) == 16
    assert int(digest, 16) >= 0


# --------------------------------------------------------------------------- snapshot


def test_snapshot_round_trips(tmp_path: Path) -> None:
    config = Config.load(
        overrides={
            "train": {"crop": [64, 64], "seed": 7},
            "coupling": {"task_weights": [0, 2], "reward_target": 2.0},
            "export": {"normalize": "per_image"},
            "runtime": {"run_dir": str(tmp_path), "name": "run"},
        }
    )
    written = config.snapshot()

    assert written == tmp_path / "run" / "config.yaml"
    assert Config.load(written) == config


def test_snapshot_creates_the_run_directory(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested"
    Config.load().snapshot(target)
    assert (target / "config.yaml").is_file()


def test_to_dict_is_yaml_safe() -> None:
    dumped = yaml.safe_dump(Config.load().to_dict())
    assert "python/object" not in dumped
