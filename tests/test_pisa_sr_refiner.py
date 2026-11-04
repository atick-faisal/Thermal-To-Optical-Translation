"""cascade-gate stage 2: the PiSA-SR refiner (ADR-01, ADR-02).

Fast tests build a tiny `AutoencoderKL`/`UNet2DConditionModel` and a synthetic `pisa_sr.pkl`
the way upstream's `save_model` writes the real one (`pisasr.py:296-305`): six adapters'
module lists, two ranks, and every LoRA tensor in one `state_dict_unet`. The adapter injection,
strict load and merge are the real ones; only the widths are small.

The same tiny components drive `t2o refine` end to end over a real stage-1 export of
conftest's synthetic pairs (CASCADE-GATE-06).

What a tiny model cannot cover is whether the real checkpoint's keys still match a current
diffusers UNet. That is the one `slow` test. It skips unless `T2O_PISA_SR_CKPT` names the
downloaded `pisa_sr.pkl` and SD-2.1-base's `unet/config.json` is in the HuggingFace cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from PIL import Image
from torch import Tensor, nn
from transformers import CLIPTextConfig, CLIPTextModel

from t2o.cli import _REFINE_OVERRIDES, build_parser, main, overrides_from_args
from t2o.config import CONFIG_FILENAME, Config, ConfigError, RefineConfig, RefineRuntimeConfig
from t2o.data.manifest import DatasetManifest
from t2o.engine.export import export_translated
from t2o.engine.refine import RefineError
from t2o.metrics.faithfulness import _translated_counterparts
from t2o.translators import build_translator
from t2o.translators.pisa_sr import (
    SD21_BASE,
    SD21_BASE_REVISION,
    PiSASRCheckpointError,
    PiSASRComponents,
    PiSASRRefiner,
    adain_color_fix,
    merge_pisa_sr_lora,
)

_CROSS_ATTENTION_DIM = 16
_PROMPT_TOKENS = 8
_FAMILIES = ("pix", "sem")
_PARTS = ("encoder", "decoder", "others")
_RANKS = {"pix": 4, "sem": 2}


class _Tokenizer:
    """Stands in for the CLIP tokenizer: the refiner only calls it once, at construction."""

    model_max_length = _PROMPT_TOKENS

    def __call__(self, prompt: str, max_length: int, **_: Any) -> Any:
        class _Encoded:
            input_ids = torch.zeros((1, max_length), dtype=torch.long)

        return _Encoded()


def _tiny_unet() -> UNet2DConditionModel:
    # Splatted through `dict[str, Any]`: diffusers annotates these tuple parameters as
    # one-element `Tuple[str]`, so any real block list is a type error at the call site.
    kwargs: dict[str, Any] = {
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "block_out_channels": (8, 16),
        "layers_per_block": 1,
        "cross_attention_dim": _CROSS_ATTENTION_DIM,
        "attention_head_dim": 2,
        "norm_num_groups": 2,
    }
    return UNet2DConditionModel(**kwargs)


def _target_modules(unet: nn.Module) -> dict[str, list[str]]:
    """Every Linear/Conv2d, split by UNet half the way upstream's three lists are named."""
    names = [n for n, m in unet.named_modules() if isinstance(m, nn.Linear | nn.Conv2d)]
    return {
        "encoder": [n for n in names if n.startswith("down_blocks.")],
        "decoder": [n for n in names if n.startswith("up_blocks.")],
        "others": [n for n in names if not n.startswith(("down_blocks.", "up_blocks."))],
    }


def _checkpoint(zero_family: str | None = None) -> dict[str, Any]:
    """A synthetic `pisa_sr.pkl`, seeded, with non-zero `lora_B` so every adapter matters.

    `zero_family` zeroes one family's tensors, which is what a pixel or semantic LoRA that
    never reached the output would look like.
    """
    donor = _tiny_unet()
    targets = _target_modules(donor)
    checkpoint: dict[str, Any] = {}
    for family in _FAMILIES:
        checkpoint[f"lora_rank_unet_{family}"] = _RANKS[family]
        for part in _PARTS:
            checkpoint[f"unet_lora_{part}_modules_{family}"] = targets[part]
            donor.add_adapter(
                LoraConfig(r=_RANKS[family], target_modules=targets[part]),
                adapter_name=f"default_{part}_{family}",
            )
    generator = torch.Generator().manual_seed(1)
    weights = {}
    for key, value in donor.state_dict().items():
        if "lora" not in key:
            continue
        filled = torch.randn(value.shape, generator=generator) * 0.1
        weights[key] = torch.zeros_like(filled) if f"_{zero_family}." in key else filled
    checkpoint["state_dict_unet"] = weights
    return checkpoint


def _components(checkpoint: dict[str, Any] | None = None) -> PiSASRComponents:
    # Seeded so every refiner built in a test shares the same base weights: only the
    # checkpoint differs between two refiners a test compares.
    torch.manual_seed(0)
    vae_kwargs: dict[str, Any] = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",) * 2,
        "up_block_types": ("UpDecoderBlock2D",) * 2,
        "block_out_channels": (4, 8),
        "layers_per_block": 1,
        "latent_channels": 4,
        "norm_num_groups": 2,
    }
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=16,
            hidden_size=_CROSS_ATTENTION_DIM,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=_PROMPT_TOKENS,
            bos_token_id=0,
            eos_token_id=1,
        )
    )
    return PiSASRComponents(
        tokenizer=_Tokenizer(),
        text_encoder=text_encoder,
        vae=AutoencoderKL(**vae_kwargs),
        unet=_tiny_unet(),
        checkpoint=_checkpoint() if checkpoint is None else checkpoint,
    )


def _images(n: int = 2, h: int = 48, w: int = 72, seed: int = 0) -> Tensor:
    # 48x72: neither side is a multiple of 64, so both get padded and sliced back. Mid-range
    # values, so the final clamp rarely binds and AdaIN's statistics survive it.
    generator = torch.Generator().manual_seed(seed)
    return 0.3 + 0.4 * torch.rand((n, 3, h, w), generator=generator)


@pytest.fixture(scope="module")
def refiner() -> PiSASRRefiner:
    return PiSASRRefiner(_components())


# ------------------------------------------------------------------------------ the recipe


def test_output_is_the_inputs_size_float32_in_unit_range(refiner: PiSASRRefiner) -> None:
    refined = refiner(_images())
    assert refined.shape == (2, 3, 48, 72)
    assert refined.dtype == torch.float32
    assert float(refined.min()) >= 0.0 and float(refined.max()) <= 1.0


def test_refining_is_deterministic(refiner: PiSASRRefiner) -> None:
    # Deviation 2: the posterior's mode, not a sample, so one input refines one way.
    images = _images()
    assert torch.equal(refiner(images), refiner(images))


def test_refining_changes_the_image(refiner: PiSASRRefiner) -> None:
    images = _images()
    assert not torch.allclose(refiner(images), images, atol=1e-3)


def test_nothing_trains_and_no_graph_is_built(refiner: PiSASRRefiner) -> None:
    assert not refiner.training
    assert not any(param.requires_grad for param in refiner.parameters())
    images = _images().requires_grad_(True)
    assert refiner(images).grad_fn is None


def test_a_float64_refiner_still_returns_float32() -> None:
    # Deviation 5: precision is the caller's `.to(dtype)`, and `t2o refine` writes float32
    # images whatever it ran at. float64 stands in for bf16, which CPU kernels cover unevenly.
    refiner = PiSASRRefiner(_components()).double()
    assert refiner(_images()).dtype == torch.float32


def test_colour_matches_each_input_image_and_channel(refiner: PiSASRRefiner) -> None:
    # AdaIN runs after the slice back, so the refined frame carries the input's own
    # per-channel mean, image by image.
    images = _images()
    refined = refiner(images)
    torch.testing.assert_close(
        refined.mean(dim=(2, 3)), images.mean(dim=(2, 3)), atol=1e-2, rtol=0.0
    )


# ---------------------------------------------------------------------------- the LoRAs


def test_every_adapter_is_merged_and_unloaded(refiner: PiSASRRefiner) -> None:
    assert not any("lora" in key for key in refiner.unet.state_dict())
    assert not any(isinstance(module, BaseTunerLayer) for module in refiner.unet.modules())


@pytest.mark.parametrize("family", _FAMILIES)
def test_both_lora_families_reach_the_output(refiner: PiSASRRefiner, family: str) -> None:
    # The trap the plan names: `add_adapter` leaves only its newest adapter active
    # (`diffusers/loaders/peft.py:544`). If one family were silently dropped, zeroing its
    # weights would change nothing.
    without = PiSASRRefiner(_components(_checkpoint(zero_family=family)))
    images = _images()
    assert not torch.allclose(without(images), refiner(images), atol=1e-6)


def test_merged_weights_are_base_plus_both_deltas() -> None:
    # The merge is additive per family: (base + pix + sem) + base == (base + pix) + (base +
    # sem), weight by weight, for every merged parameter.
    def merged(checkpoint: dict[str, Any]) -> dict[str, Tensor]:
        unet = _components(checkpoint).unet
        merge_pisa_sr_lora(unet, checkpoint)
        return unet.state_dict()

    both = merged(_checkpoint())
    pix_only = merged(_checkpoint(zero_family="sem"))
    sem_only = merged(_checkpoint(zero_family="pix"))
    base = _components().unet.state_dict()
    for key, value in base.items():
        torch.testing.assert_close(both[key] + value, pix_only[key] + sem_only[key])


# ---------------------------------------------------------------------- strict loading


def test_a_checkpoint_tensor_that_fits_no_key_is_refused() -> None:
    checkpoint = _checkpoint()
    checkpoint["state_dict_unet"]["down_blocks.9.lora_A.default_encoder_pix.weight"] = torch.zeros(
        1
    )
    with pytest.raises(PiSASRCheckpointError, match=r"1 tensor\(s\) fit no key"):
        PiSASRRefiner(_components(checkpoint))


def test_an_unfilled_lora_parameter_is_refused() -> None:
    checkpoint = _checkpoint()
    weights = checkpoint["state_dict_unet"]
    del weights[next(key for key in weights if "_sem." in key)]
    with pytest.raises(PiSASRCheckpointError, match=r"1 LoRA parameter\(s\) left unfilled"):
        PiSASRRefiner(_components(checkpoint))


def test_a_checkpoint_without_its_schema_is_refused() -> None:
    checkpoint = _checkpoint()
    del checkpoint["lora_rank_unet_sem"]
    with pytest.raises(PiSASRCheckpointError, match="lora_rank_unet_sem"):
        PiSASRRefiner(_components(checkpoint))


def test_a_non_lora_tensor_in_the_checkpoint_is_loaded_not_dropped() -> None:
    # Matched against the full `state_dict()`: a norm no adapter wraps still lands.
    checkpoint = _checkpoint()
    replacement = torch.full_like(_tiny_unet().state_dict()["conv_norm_out.weight"], 0.5)
    checkpoint["state_dict_unet"]["conv_norm_out.weight"] = replacement
    refiner = PiSASRRefiner(_components(checkpoint))
    assert torch.equal(refiner.unet.state_dict()["conv_norm_out.weight"], replacement)


# ------------------------------------------------------------------------------- AdaIN


def test_adain_gives_each_image_its_sources_statistics() -> None:
    generator = torch.Generator().manual_seed(2)
    target = torch.rand((2, 3, 16, 16), generator=generator)
    source = torch.stack(
        [
            0.2 + 0.1 * torch.rand((3, 16, 16), generator=generator),
            0.6 + 0.3 * torch.rand((3, 16, 16), generator=generator),
        ]
    )
    fixed = adain_color_fix(target, source)
    torch.testing.assert_close(fixed.mean(dim=(2, 3)), source.mean(dim=(2, 3)))
    # Upstream adds eps = 1e-5 to each variance before the square root, which moves a std of
    # ~0.03 by under 1%: close, not equal, by design.
    torch.testing.assert_close(
        fixed.flatten(2).std(dim=2), source.flatten(2).std(dim=2), atol=0.0, rtol=1e-2
    )


# ------------------------------------------------------------------------- `t2o refine`
#
# End to end (CASCADE-GATE-06): a real stage-1 export of conftest's synthetic pairs, refined
# through `main` with the tiny components standing in for SD-2.1 and `pisa_sr.pkl`.


def _refine_yaml(export: Path, root: Path) -> Path:
    checkpoint = root / "pisa_sr.pkl"
    checkpoint.touch()  # only its existence is checked; the tiny components replace the load
    config = root / "refine.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "source": {"export": str(export)},
                "refiner": {"checkpoint": str(checkpoint)},
                "runtime": {"device": "cpu", "run_dir": str(root), "name": "refined"},
            }
        )
    )
    return config


@pytest.fixture(scope="module")
def stage1_export(data_yaml: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = Config.load(
        overrides={
            "data": {"manifest": str(data_yaml)},
            "translator": {"backbone": "stub", "hidden_channels": 4},
            "runtime": {"device": "cpu", "workers": 0},
        }
    )
    root = tmp_path_factory.mktemp("stage1") / "translated"
    export_translated(build_translator(config), config, root)
    # What a scored export carries after ultralytics has read it; a refine must not copy it.
    (root / "val" / "labels.cache").write_bytes(b"stale")
    return root


@pytest.fixture(scope="module")
def refined(stage1_export: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = _refine_yaml(stage1_export, tmp_path_factory.mktemp("refine"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("t2o.translators.pisa_sr.load_pisa_sr", lambda *_, **__: _components())
        assert main(["refine", "--config", str(config)]) == 0
    return config.parent / "refined"


def test_refine_writes_every_frame_at_its_size_under_its_stem(
    stage1_export: Path, refined: Path
) -> None:
    sources = sorted((stage1_export / "val" / "images").iterdir())
    outputs = sorted((refined / "val" / "images").iterdir())

    assert [p.name for p in outputs] == [p.name for p in sources]
    for source, output in zip(sources, outputs, strict=True):
        with Image.open(source) as before, Image.open(output) as after:
            assert (after.size, after.mode) == (before.size, "RGB")
            assert np.abs(np.asarray(after, float) - np.asarray(before, float)).mean() > 0


def test_refine_copies_labels_byte_for_byte_and_never_the_cache(
    stage1_export: Path, refined: Path
) -> None:
    sources = sorted((stage1_export / "val" / "labels").glob("*.txt"))

    assert [(p.name, p.read_bytes()) for p in sorted((refined / "val" / "labels").iterdir())] == [
        (p.name, p.read_bytes()) for p in sources
    ]
    assert not list(refined.rglob("labels.cache"))


def test_refined_data_yaml_points_at_the_refined_split(stage1_export: Path, refined: Path) -> None:
    """Absolute `path:`, since ultralytics honours it literally; classes kept from stage 1."""
    data = yaml.safe_load((refined / "data.yaml").read_text())
    source = yaml.safe_load((stage1_export / "data.yaml").read_text())

    assert data["path"] == str(refined.resolve())
    assert (data["train"], data["val"]) == ("val/images", "val/images")
    assert (data["nc"], data["names"]) == (source["nc"], source["names"])


def test_faithfulness_pairs_the_refined_frames_with_the_visible_split(
    data_yaml: Path, refined: Path
) -> None:
    """The seam `t2o faithfulness` and `t2o completion` both read, with no change to either."""
    visible = sorted(DatasetManifest.load(data_yaml).val_images.iterdir())

    paired = _translated_counterparts(visible, refined / "val" / "images")

    assert [p.stem for p in paired] == [p.stem for p in visible]


def test_refine_snapshots_a_config_that_reloads_to_the_same_hash(refined: Path) -> None:
    snapshot = RefineConfig.load(refined / CONFIG_FILENAME)
    renamed = snapshot.model_copy(update={"runtime": RefineRuntimeConfig(name="elsewhere")})

    assert snapshot.runtime.path == refined
    assert renamed.config_hash() == snapshot.config_hash()


def test_refine_config_rejects_unknown_keys_and_requires_its_inputs() -> None:
    with pytest.raises(ConfigError) as error:
        RefineConfig.load(overrides={"source": {"export": "x", "seed": 0}})

    assert "source.seed" in str(error.value)
    assert "refiner: Field required" in str(error.value)


def test_the_dtype_is_part_of_the_experiment() -> None:
    base = {"source": {"export": "x"}, "refiner": {"checkpoint": "c.pkl"}}
    half = {**base, "refiner": {"checkpoint": "c.pkl", "dtype": "float16"}}

    assert RefineConfig.load(overrides=base).config_hash() != (
        RefineConfig.load(overrides=half).config_hash()
    )


def test_refine_flags_override_the_config_file(stage1_export: Path, tmp_path: Path) -> None:
    config = _refine_yaml(stage1_export, tmp_path)
    args = build_parser().parse_args(["refine", "--config", str(config), "--name", "second"])

    loaded = RefineConfig.load(args.config, overrides_from_args(args, _REFINE_OVERRIDES))

    assert loaded.runtime.path == tmp_path / "second"
    assert loaded.source.export == stage1_export


def _must_not_load(*_: object, **__: object) -> PiSASRComponents:
    raise AssertionError("weights were loaded before the paths were checked")


def test_refine_refuses_a_non_empty_output_before_loading_weights(
    stage1_export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointed at the stage-1 export itself, a refine must not write a single frame over it."""
    monkeypatch.setattr("t2o.translators.pisa_sr.load_pisa_sr", _must_not_load)
    config = _refine_yaml(stage1_export, tmp_path)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in stage1_export.rglob("*"))
    argv = ["refine", "--config", str(config)]

    with pytest.raises(RefineError, match="is not empty"):
        main([*argv, "--run-dir", str(stage1_export.parent), "--name", stage1_export.name])

    assert sorted((p.name, p.stat().st_mtime_ns) for p in stage1_export.rglob("*")) == before


def test_refine_lists_every_missing_input_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("t2o.translators.pisa_sr.load_pisa_sr", _must_not_load)
    argv = ["refine", "--source", str(tmp_path / "nowhere"), "--checkpoint", "missing.pkl"]

    with pytest.raises(RefineError) as error:
        main([*argv, "--run-dir", str(tmp_path), "--name", "refined"])

    message = str(error.value)
    assert "no data.yaml" in message
    assert "no images" in message
    assert "no PiSA-SR checkpoint" in message


# -------------------------------------------------------------------------------- slow


def _real_checkpoint() -> tuple[Path, str] | None:
    from huggingface_hub import try_to_load_from_cache

    path = os.environ.get("T2O_PISA_SR_CKPT")
    config = try_to_load_from_cache(SD21_BASE, "unet/config.json", revision=SD21_BASE_REVISION)
    if path is None or not Path(path).is_file() or not isinstance(config, str):
        return None
    return Path(path), config


@pytest.mark.slow
@pytest.mark.skipif(
    _real_checkpoint() is None,
    reason="needs T2O_PISA_SR_CKPT and SD-2.1-base's unet/config.json in the HF cache",
)
def test_the_real_checkpoint_strict_loads_into_an_sd21_unet() -> None:
    """ADR-01's "believed, not verified": `pisa_sr.pkl` loads under `weights_only=True` and
    its keys fit a current diffusers SD-2.1 UNet exactly.

    The UNet is built from its config alone, with random weights: the key scheme is what is
    under test, and that needs a ~1 KB download rather than the 3.5 GB of weights.
    """
    found = _real_checkpoint()
    assert found is not None
    path, config = found
    unet = cast(
        UNet2DConditionModel,
        UNet2DConditionModel.from_config(cast(Any, UNet2DConditionModel.load_config(config))),
    )
    merge_pisa_sr_lora(unet, torch.load(path, map_location="cpu", weights_only=True))
    assert not any("lora" in key for key in unet.state_dict())
