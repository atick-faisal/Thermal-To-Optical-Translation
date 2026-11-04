"""The cascade's stage 2: PiSA-SR's default setting as a frozen RGB -> RGB refiner.

cascade-gate ADR-01 picked PiSA-SR (`csslc/PiSA-SR@9c76e12`) in its default setting: the
pixel and semantic LoRAs both merged at weight 1.0 into the SD-2.1-base UNet, and one UNet
pass (`pisasr.py:333-335`, `_predict_no_tiling`'s `default` branch at `:509-510`). Nothing is
vendored. Upstream's `unet_2d_condition.py` and `autoencoder_kl.py` are diffusers 0.25's
stock classes and no longer import under the pinned diffusers, so this module uses diffusers'
own `UNet2DConditionModel` and `AutoencoderKL`, and peft's own merge (ADR-01, Decision).

This is **not** a `Translator`. `translate()` must stay on the gradient graph
(`protocol.py:46-53`); this module never is. Plan Q1 keeps the protocol unchanged, so
`build_translator` has no branch for it and `t2o refine` constructs it directly.

The recipe, reimplemented:

1. RGB in [0, 1] -> [-1, 1] (`test_pisasr.py:54`), reflect-padded to a multiple of 64.
2. VAE-encode, scale by `scaling_factor` (`pisasr.py:481`).
3. `z' = z - UNet(z, t=1, emb(""))` (`:341`, `:487`; the empty prompt is `test_pisasr.py:50`).
4. Decode, clamp to [-1, 1], slice back to the input size, map to [0, 1]
   (`pisasr.py:488`, `test_pisasr.py:60-61`).
5. AdaIN colour matching to the input: per image, per channel mean and std
   (`test_pisasr.py:64-65`, `src/my_utils/wavelet_color_fix.py:44-71`).

The checkpoint, `pisa_sr.pkl`, holds six adapters: pixel and semantic, each split into encoder,
decoder and "others" module lists with one rank per family, and one `state_dict_unet` of every
LoRA tensor (`pisasr.py:296-305`). Upstream's training model carries all six at once
(`:243-249`), so injecting all six here reproduces the saved key names exactly. peft's
`LoraConfig` defaults (`lora_alpha=8`, no rsLoRA, no DoRA) set the scaling, unchanged since the
peft 0.9 upstream trained under.

Five deliberate deviations from upstream (ADR-01), plus one implementation note:

1. **No resizing.** Upstream resizes a 640x480 frame to 680x512 and back
   (`test_pisasr.py:38-46,69-70`). Padding keeps the input's own pixels, so the thin
   components the gate scores are not blurred by two resamplings.
2. **The VAE posterior's mode replaces `latent_dist.sample()`** (`pisasr.py:481`), so a
   refined image is a deterministic function of its input.
3. **The empty prompt is encoded once and the text encoder dropped**, as `pix2pix_turbo.py`'s
   deviation 3.
4. **Components are injected, not loaded inside the wrapper**, as `pix2pix_turbo.py`'s
   deviation 1, so CPU tests use tiny local models.
5. **No device or precision is hardcoded.** Upstream says `"cuda"` throughout (`pisasr.py:311`)
   and defaults to fp16 (`test_pisasr.py:97`). Here the caller moves the module with
   `.to(device, dtype)`; the output is float32 either way.
- *Note:* AdaIN runs on float tensors before any 8-bit quantisation. Upstream runs it on PIL
  images (`test_pisasr.py:62-65`), so its statistics see a uint8 round trip first.

**LoRA activation.** diffusers' `add_adapter` leaves only the adapter it just added active
(`diffusers/loaders/peft.py:544`). Upstream works around that with
`set_weights_and_activate_adapters` before each merge (`pisasr.py:334,431`). Here every adapter
is merged *by name* through `fuse_lora(adapter_names=...)`, which calls peft's own
`BaseTunerLayer.merge`, so which adapter happens to be active never matters.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

# Imported from their defining modules, not the `diffusers` root: the root re-exports lazily
# and pyright rejects it as a private import.
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
from torch import Tensor, nn
from transformers import AutoTokenizer, CLIPTextModel

from t2o.translators.pix2pix_turbo import _encode_prompt, _pad_to_multiple

logger = logging.getLogger(__name__)

# ADR-01: StabilityAI's own repo returns 401 to an anonymous request; this mirror is the one
# OSEDiff's README now points to, pinned by revision (Open RAIL++-M).
SD21_BASE = "Manojb/stable-diffusion-2-1-base"
SD21_BASE_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"

# `test_pisasr.py:50`. A constant, so its embedding is one too.
_PROMPT = ""
# `pisasr.py:341`: the residual step runs the UNet at t = 1, not upstream's training t = 999.
_TIMESTEP = 1
# `wavelet_color_fix.py:44`, added to the variance before the square root.
_ADAIN_EPS = 1.0e-5

# (adapter name, module-list field, rank field), in upstream's own names and order
# (`pisasr.py:413-424,439-441`). The adapter names are baked into every LoRA key in
# `state_dict_unet`, so they have to match character for character.
_ADAPTERS = (
    ("default_encoder_pix", "unet_lora_encoder_modules_pix", "lora_rank_unet_pix"),
    ("default_decoder_pix", "unet_lora_decoder_modules_pix", "lora_rank_unet_pix"),
    ("default_others_pix", "unet_lora_others_modules_pix", "lora_rank_unet_pix"),
    ("default_encoder_sem", "unet_lora_encoder_modules_sem", "lora_rank_unet_sem"),
    ("default_decoder_sem", "unet_lora_decoder_modules_sem", "lora_rank_unet_sem"),
    ("default_others_sem", "unet_lora_others_modules_sem", "lora_rank_unet_sem"),
)
_WEIGHTS_FIELD = "state_dict_unet"
_CHECKPOINT_FIELDS = (
    *dict.fromkeys(field for _, modules, rank in _ADAPTERS for field in (modules, rank)),
    _WEIGHTS_FIELD,
)


class PiSASRCheckpointError(RuntimeError):
    """`pisa_sr.pkl` does not fill the UNet's LoRA adapters exactly (ADR-02)."""


@dataclass(frozen=True, slots=True)
class PiSASRComponents:
    """The pretrained pieces plus the LoRA checkpoint, kept separate from the refiner.

    Injected so the test suite can build a tiny VAE/UNet and a synthetic checkpoint locally and
    still drive the real adapter injection, strict load and merge without the network.
    """

    tokenizer: Any
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    checkpoint: Mapping[str, Any]


def load_pisa_sr(
    checkpoint: Path, pretrained: str = SD21_BASE, revision: str = SD21_BASE_REVISION
) -> PiSASRComponents:
    """Fetch SD-2.1-base (HuggingFace cache, never the repo) and read `pisa_sr.pkl`.

    The only function here that touches the network or the disk. `weights_only=True`: the
    checkpoint holds only name lists, integer ranks and tensors (`pisasr.py:296-305`).
    """
    logger.info("loading %s at %s, LoRA checkpoint %s", pretrained, revision, checkpoint)
    return PiSASRComponents(
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained, subfolder="tokenizer", revision=revision
        ),
        text_encoder=CLIPTextModel.from_pretrained(
            pretrained, subfolder="text_encoder", revision=revision
        ),
        vae=AutoencoderKL.from_pretrained(pretrained, subfolder="vae", revision=revision),
        unet=UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", revision=revision),
        checkpoint=torch.load(checkpoint, map_location="cpu", weights_only=True),
    )


def merge_pisa_sr_lora(unet: UNet2DConditionModel, checkpoint: Mapping[str, Any]) -> None:
    """Inject the six adapters, fill them strictly from the checkpoint, merge them all at 1.0.

    Strict both ways (ADR-01, Consequences; ADR-02): every checkpoint tensor must land on a
    UNet key, and every LoRA parameter the injection created must be filled. Otherwise a
    checkpoint that silently fails to load leaves peft's zero-initialised `lora_B`, so stage 2
    becomes stock SD-2.1 at t = 1 -- close to an identity, and a trivial gate pass.

    Checked against the UNet's *full* `state_dict()`, not only its LoRA keys, so a non-LoRA
    tensor the checkpoint might carry is loaded rather than dropped.
    """
    absent = [field for field in _CHECKPOINT_FIELDS if field not in checkpoint]
    if absent:
        raise PiSASRCheckpointError(
            f"checkpoint lacks {absent}; a PiSA-SR checkpoint holds {list(_CHECKPOINT_FIELDS)}"
        )
    for adapter, modules, rank in _ADAPTERS:
        unet.add_adapter(
            LoraConfig(
                r=checkpoint[rank],
                init_lora_weights="gaussian",
                target_modules=list(checkpoint[modules]),
            ),
            adapter_name=adapter,
        )

    weights: Mapping[str, Tensor] = checkpoint[_WEIGHTS_FIELD]
    model_keys = set(unet.state_dict())
    unused = sorted(set(weights) - model_keys)
    unfilled = sorted(key for key in model_keys - set(weights) if "lora" in key)
    if unused or unfilled:
        raise PiSASRCheckpointError(
            f"checkpoint does not match the UNet's adapters: {len(unused)} tensor(s) fit no key "
            f"(first {unused[:1]}), {len(unfilled)} LoRA parameter(s) left unfilled "
            f"(first {unfilled[:1]})"
        )
    unet.load_state_dict(dict(weights), strict=False)

    # Merged by name, so the active-adapter state `add_adapter` leaves behind is irrelevant.
    unet.fuse_lora(adapter_names=[adapter for adapter, _, _ in _ADAPTERS])
    unet.unload_lora()
    logger.info("merged %d PiSA-SR adapters (%d tensors)", len(_ADAPTERS), len(weights))


def adain_color_fix(target: Tensor, source: Tensor) -> Tensor:
    """Give each image of `target` the per-channel mean and std of the same image in `source`.

    `wavelet_color_fix.py:44-71`: unbiased variance plus eps, then the square root. Both are
    `(B, C, H, W)`; the statistics are per image and per channel, never pooled over the batch.
    """
    target_mean, target_std = _mean_std(target)
    source_mean, source_std = _mean_std(source)
    return (target - target_mean) / target_std * source_std + source_mean


def _mean_std(image: Tensor) -> tuple[Tensor, Tensor]:
    flat = image.flatten(start_dim=2)
    mean = flat.mean(dim=2)[..., None, None]
    std = (flat.var(dim=2) + _ADAIN_EPS).sqrt()[..., None, None]
    return mean, std


class PiSASRRefiner(nn.Module):
    """Frozen, gradient-free RGB `(B, 3, H, W)` [0, 1] -> the same shape, float32, [0, 1].

    The LoRAs are merged at construction, so what remains is a plain VAE and UNet. Every
    parameter is frozen and the module is put in eval mode; the forward runs under
    `torch.no_grad`, so a refined image never carries a graph.
    """

    def __init__(self, components: PiSASRComponents) -> None:
        super().__init__()
        merge_pisa_sr_lora(components.unet, components.checkpoint)
        self.vae = components.vae
        self.unet = components.unet
        self.scaling_factor = float(components.vae.config["scaling_factor"])
        # Encoded once: the text encoder and tokenizer are not kept (deviation 3).
        self.register_buffer(
            "caption",
            _encode_prompt(components.tokenizer, components.text_encoder, _PROMPT),
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        parameter = next(self.unet.parameters())
        height, width = image.shape[-2:]
        source = image.to(device=parameter.device, dtype=torch.float32)
        control = _pad_to_multiple(source * 2.0 - 1.0).to(parameter.dtype)

        # `Any` on the diffusers call sites: each returns a dataclass-or-tuple union keyed on
        # `return_dict`, which no annotation narrows for us.
        posterior = cast(Any, self.vae.encode(control)).latent_dist
        latents = posterior.mode() * self.scaling_factor
        caption = self.get_buffer("caption").expand(control.shape[0], -1, -1)
        predicted: Tensor = self.unet(latents, _TIMESTEP, encoder_hidden_states=caption).sample
        decoded = cast(Any, self.vae.decode((latents - predicted) / self.scaling_factor)).sample

        # Sliced back *before* AdaIN, so the reflect padding never biases its statistics.
        refined = (cast(Tensor, decoded).float().clamp(-1.0, 1.0)[..., :height, :width] + 1.0) / 2
        return adain_color_fix(refined, source).clamp(0.0, 1.0)
