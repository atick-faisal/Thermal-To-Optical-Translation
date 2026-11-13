"""Uniform wrapper per backbone over `third_party/`. See `Translator` for the contract."""

from __future__ import annotations

import math

import torch
from torch import nn

from t2o.config import (
    AmpDtype,
    Config,
    ConfigError,
    CycleGANTranslatorConfig,
    ModTrTranslatorConfig,
    Pix2PixTranslatorConfig,
    Pix2PixTurboTranslatorConfig,
    StubTranslatorConfig,
)
from t2o.data.dataset import TranslationPairDataset
from t2o.data.manifest import DatasetManifest
from t2o.seeding import seed_everything
from t2o.translators.cyclegan import CycleGANTranslator
from t2o.translators.modtr import ModTrTranslator
from t2o.translators.pix2pix import Pix2PixTranslator
from t2o.translators.pix2pix_turbo import Pix2PixTurboTranslator, load_sd_turbo
from t2o.translators.protocol import Translator
from t2o.translators.stub import StubTranslator

__all__ = [
    "CycleGANTranslator",
    "ModTrTranslator",
    "Pix2PixTranslator",
    "Pix2PixTurboTranslator",
    "StubTranslator",
    "Translator",
    "build_translator",
]

# `TrainConfig.amp_dtype` is a config-layer enum; only the backbones that actually autocast
# need the torch dtype behind it, so the mapping lives at the one place they are constructed.
_AMP_DTYPES = {
    AmpDtype.BFLOAT16: torch.bfloat16,
    AmpDtype.FLOAT16: torch.float16,
    AmpDtype.FLOAT32: torch.float32,
}


def build_translator(config: Config) -> nn.Module:
    """Construct the backbone named by `config.translator.backbone`.

    One `isinstance` branch per `TranslatorConfig` union member (PLAN.md invariant 3): a new
    backbone adds one branch here and nothing else in the dispatch changes -- M2a's
    pix2pix-turbo cost exactly that. The final `raise` is what keeps the next addition
    (M2b's LBBDM) a one-branch diff rather than a silent fallthrough.

    Seeds every RNG (`t2o.seeding.seed_everything`: torch, CUDA, numpy, `random`) from
    `config.train.seed` before constructing the backbone. `train.seed` is part of
    `config_hash()` on the premise that a seed is scientific (M0.2); that premise only holds
    if it actually determines the translator's initial weights, and this is the one place
    every caller (cli.py, tests) already goes through to get one. No other choke point
    exists: `Trainer` never constructs a translator, it only trains one.
    """
    seed_everything(config.train.seed)
    if isinstance(config.translator, CycleGANTranslatorConfig):
        _require_unpaired_objective(config)
        return CycleGANTranslator(
            total_batches=_count_training_batches(config),
            infrared_channels=config.data.infrared_channels,
            net_g=config.translator.net_g,
            net_d=config.translator.net_d,
            ngf=config.translator.ngf,
            ndf=config.translator.ndf,
            lr=config.train.lr,
            loss_gan=config.loss.gan,
            lambda_cycle=config.translator.lambda_cycle,
            lambda_identity=config.translator.lambda_identity,
        )
    _require_single_channel_infrared(config)
    if isinstance(config.translator, StubTranslatorConfig):
        return StubTranslator(hidden_channels=config.translator.hidden_channels)
    if isinstance(config.translator, Pix2PixTranslatorConfig):
        return Pix2PixTranslator(
            net_g=config.translator.net_g,
            net_d=config.translator.net_d,
            ngf=config.translator.ngf,
            ndf=config.translator.ndf,
            gan_mode=config.translator.gan_mode,
            lr=config.train.lr,
            loss_l2=config.loss.l2,
            loss_lpips=config.loss.lpips,
            loss_gan=config.loss.gan,
        )
    if isinstance(config.translator, Pix2PixTurboTranslatorConfig):
        return Pix2PixTurboTranslator(
            components=load_sd_turbo(config.translator.pretrained),
            prompt=config.translator.prompt,
            lora_rank_unet=config.translator.lora_rank_unet,
            lora_rank_vae=config.translator.lora_rank_vae,
            net_d=config.translator.net_d,
            ndf=config.translator.ndf,
            gan_mode=config.translator.gan_mode,
            lr=config.train.lr,
            loss_l2=config.loss.l2,
            loss_lpips=config.loss.lpips,
            loss_gan=config.loss.gan,
            amp=config.train.amp,
            amp_dtype=_AMP_DTYPES[config.train.amp_dtype],
        )
    if isinstance(config.translator, ModTrTranslatorConfig):
        _require_detection_only(config)
        return ModTrTranslator(
            total_batches=_count_training_batches(config),
            lr=config.train.lr,
            pretrained=config.translator.pretrained,
        )
    raise ConfigError(f"no translator wired up for backbone {config.translator.backbone!r}")


def _require_single_channel_infrared(config: Config) -> None:
    """Refuse RGB infrared for every backbone wired up here (cyclegan-cascade ADR-01).

    Each one fixes its input at one channel: pix2pix's `define_G(_INFRARED_CHANNELS, ...)`, the
    stub's first conv, ModTr repeating one channel to three, pix2pix-turbo expanding one. Fed
    three, they crash on the first batch or, worse, run on a reinterpreted input. Only
    `cyclegan` takes 3; its branch goes above this check.
    """
    if config.data.infrared_channels != 1:
        raise ConfigError(
            f"backbone {config.translator.backbone!r} reads a 1-channel infrared input, but "
            f"data.infrared_channels is {config.data.infrared_channels}; only 'cyclegan' "
            "accepts 3 -- set data.infrared_channels to 1 or switch the backbone"
        )


def _require_unpaired_objective(config: Config) -> None:
    """CycleGAN's objective is adversarial plus cycle; refuse a config that says otherwise.

    `loss.l2`/`loss.lpips` compare the output to its own visible frame, which `data.
    unpaired_visible` replaces with a random one, so they would train toward mush.
    `LossConfig` defaults to `lpips: 5.0` and `gan: 0.0`, so a config that forgets the `loss`
    section gets exactly that. The identity term feeds a visible image to G_AB, so it needs
    three input channels (upstream's own assert, `models/cycle_gan_model.py:87-88`).
    """
    assert isinstance(config.translator, CycleGANTranslatorConfig)
    if config.loss.l2 != 0.0 or config.loss.lpips != 0.0 or config.loss.gan <= 0.0:
        raise ConfigError(
            f"backbone 'cyclegan' trains on unpaired losses, but loss is "
            f"{config.loss.model_dump()}; set loss.l2 and loss.lpips to 0 and loss.gan above 0"
        )
    if config.translator.lambda_identity > 0.0 and config.data.infrared_channels != 3:
        raise ConfigError(
            "backbone 'cyclegan' with translator.lambda_identity "
            f"{config.translator.lambda_identity} feeds visible images to the infrared side, "
            f"which needs data.infrared_channels 3, not {config.data.infrared_channels}; "
            "set it to 3 or lambda_identity to 0"
        )


def _require_detection_only(config: Config) -> None:
    """ModTr's objective is the detector's loss alone; refuse a config that says otherwise.

    `LossConfig` defaults to `lpips: 5.0`, so a ModTr config that forgets the `loss` section
    would carry weights that change its hash and nothing else. A stage at weight 0 would leave
    `fit` with no objective at all.
    """
    weights = config.loss.model_dump()
    if any(weight != 0.0 for weight in weights.values()):
        raise ConfigError(
            f"backbone 'modtr' trains on detection loss only, but loss is {weights}; "
            "set loss.l2, loss.lpips and loss.gan to 0"
        )
    if any(weight <= 0.0 for weight in config.coupling.task_weights):
        raise ConfigError(
            "backbone 'modtr' needs a detection term in every stage, but "
            f"coupling.task_weights is {list(config.coupling.task_weights)}"
        )


def _count_training_batches(config: Config) -> int:
    """Every training batch the run will see: ModTr's cosine `T_max` (modtr-baseline Q8).

    Counted through `TranslationPairDataset` itself, with the same cap `Trainer` applies, so the
    count follows the real corpus: the calibration run's 480 images and the campaign's 600 each
    get their own `T_max` without a number to keep in sync by hand. The `DataLoader` keeps a
    short last batch, hence the ceiling.
    """
    manifest = DatasetManifest.load(config.data.manifest)
    corpus = TranslationPairDataset(
        manifest.train_images,
        pairing=manifest.pairing,
        max_images=config.data.max_train_images,
        subset_seed=config.data.subset_seed,
    )
    per_epoch = math.ceil(len(corpus) / config.train.batch_size)
    return per_epoch * config.train.epochs_per_stage * len(config.coupling.task_weights)
