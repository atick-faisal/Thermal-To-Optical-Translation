"""M2a's one-step diffusion backbone (`PLAN.md` §4, §10 Phase 2a) -- the strong arm of E3.

SD-turbo with LoRA adapters on the UNet and VAE, plus skip connections from the VAE encoder
into its decoder. One UNet evaluation, no sampling loop, so the detection-consistency loss
back-props through the *entire* generator exactly -- which is the whole reason this backbone
replaced LBBDM-from-scratch as the Phase 2 primary.

Vendored from `GaParmar/img2img-turbo` at `463b2d3`: `third_party/pix2pix_turbo/model.py`,
verbatim, for `my_vae_encoder_fwd`/`my_vae_decoder_fwd` -- the skip-connection forwards that
are upstream's actual contribution. Its `make_1step_sched` is vendored with the file but
never called (it does `set_timesteps(1, device="cuda")` and a network fetch); this wrapper
configures the scheduler it is handed instead.

`src/pix2pix_turbo.py::Pix2Pix_Turbo` is deliberately **not** vendored, unlike PLAN.md §7's
table. It cannot be: the file opens with `sys.path.append("src/")` + `from model import ...`
and so does not import outside upstream's own cwd, and 100 of its 229 lines are
`requests`/`tqdm` checkpoint downloads for two pretrained tasks we do not use. What is left
is the random-init branch and a 12-line forward, with `.cuda()` hardcoded in six places.
Reimplementing it here is the precedent M1 already set -- `models/networks.py` was vendored,
`Pix2PixModel`/`BaseModel` were reimplemented as `Pix2PixTranslator`. The LoRA target-module
lists, the part that is a recipe rather than plumbing, are copied below with their citation.

Six deliberate deviations from upstream, each cited where it happens:

1. Components are injected, not constructed. `load_sd_turbo` is the only thing that touches
   HuggingFace, so tests can drive the real LoRA wiring and the real skip forwards with a
   small locally-built VAE/UNet and no network.
2. Skip-conv widths are derived from `vae.config.block_out_channels`, not the four literal
   `Conv2d(512, 512)`-style calls upstream writes for SD's own geometry.
3. The caption is encoded once at construction and the text encoder is then dropped. The
   prompt is a config constant, so its embedding is a constant; upstream re-encodes it every
   forward and keeps a frozen ~250MB text encoder resident for the whole run.
4. Input is reflect-padded to a multiple of 64 and sliced back after decode. The VAE is f8
   and the UNet downsamples 8x more; the dataset is 640x480 and 480 is not divisible by 64,
   so a full frame would shape-mismatch inside the UNet's skip concat.
5. `state_dict()`/`load_state_dict()` carry only what trains. `Trainer` writes a checkpoint
   twice per stage and `engine/loop.py` warm-starts from it -- unreduced that is ~2.5GB a
   time, hundreds of GB for a seeded campaign, all of it frozen base weights that
   `load_sd_turbo` reproduces exactly.
6. `loss.gan` uses the PatchGAN already vendored for M1 rather than upstream's
   `vision_aided_loss` CLIP discriminator. PLAN.md §8's loss is the same four knobs for every
   backbone; a different adversarial objective per backbone would confound E2/E3's backbone
   comparison with a change nobody asked for.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F

# Imported from their defining modules, not the `diffusers` root: the root re-exports lazily
# and pyright rejects it as a private import.
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from peft import LoraConfig
from third_party.pix2pix import networks
from third_party.pix2pix_turbo.model import my_vae_decoder_fwd, my_vae_encoder_fwd
from torch import Tensor, nn
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from transformers import AutoTokenizer, CLIPTextModel

from t2o.data.dataset import TranslationBatch

logger = logging.getLogger(__name__)

SD_TURBO = "stabilityai/sd-turbo"

# Our data contract is 1-channel infrared / 3-channel visible (PLAN.md §9); the VAE wants 3.
_INFRARED_CHANNELS = 1
_VISIBLE_CHANNELS = 3
# VAE downsamples by 8, the UNet by 8 more. Anything not divisible by this shape-mismatches
# inside the UNet's skip concat -- 480 is the dataset's own height and is not divisible.
_SIZE_MULTIPLE = 64
# The single fixed timestep upstream distils to (`pix2pix_turbo.py:162`).
_TIMESTEP = 999
# `train_pix2pix_turbo.py:98` -- AdamW, and `:190` clips before every step.
_MAX_GRAD_NORM = 10.0
# The discriminator half is pix2pix's, so it keeps pix2pix's own conventions.
_NORM = "batch"
_INIT_TYPE = "normal"
_INIT_GAIN = 0.02
_D_BETA1 = 0.5
_N_LAYERS_D = 3
_LPIPS_NET = "alex"

# `pix2pix_turbo.py:137-150`, verbatim. These are the recipe: which submodules get adapters.
_LORA_TARGETS_VAE = (
    "conv1",
    "conv2",
    "conv_in",
    "conv_shortcut",
    "conv",
    "conv_out",
    "skip_conv_1",
    "skip_conv_2",
    "skip_conv_3",
    "skip_conv_4",
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
)
_LORA_TARGETS_UNET = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "conv_out",
    "proj_in",
    "proj_out",
    "ff.net.2",
    "ff.net.0.proj",
)
# `pix2pix_turbo.py:133-136` -- near-zero so an untrained model starts as plain sd-turbo.
_SKIP_CONV_INIT = 1.0e-5


class TurboTranslatorError(RuntimeError):
    """A backbone could not be assembled from the components it was given."""


@dataclass(frozen=True, slots=True)
class SDTurboComponents:
    """The four pretrained pieces plus a scheduler, kept separate from the wrapper.

    Injecting these rather than loading them inside ``__init__`` is what lets the test suite
    build a small VAE/UNet locally and still exercise the real adapter wiring, the real
    vendored skip forwards and real gradient flow without touching the network.
    """

    tokenizer: Any
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    scheduler: DDPMScheduler


def load_sd_turbo(pretrained: str = SD_TURBO) -> SDTurboComponents:
    """Fetch the pretrained components (HuggingFace cache, never the repo).

    The only function here that touches the network. `build_translator` is its one caller.
    """
    logger.info("loading %s", pretrained)
    return SDTurboComponents(
        tokenizer=AutoTokenizer.from_pretrained(pretrained, subfolder="tokenizer"),
        text_encoder=CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder"),
        vae=AutoencoderKL.from_pretrained(pretrained, subfolder="vae"),
        unet=UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet"),
        scheduler=DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler"),
    )


def skip_connection_channels(block_out_channels: list[int]) -> tuple[tuple[int, int], ...]:
    """Derive the four skip convs' (in, out) widths from the VAE's own block widths.

    Upstream hardcodes `(512, 512)`, `(256, 512)`, `(128, 512)`, `(128, 256)` -- SD's VAE
    geometry spelled as literals. `my_vae_encoder_fwd` appends its activation *before* each
    down block, so for widths `[c0, c1, c2, c3]` the stored activations are
    `[c0, c0, c1, c2]`; `my_vae_decoder_fwd` consumes them reversed and adds each to the
    decoder's running sample, whose widths entering the up blocks are `[c3, c3, c2, c1]`.
    """
    if len(block_out_channels) != 4:
        raise TurboTranslatorError(
            "the vendored skip forwards assume a 4-block VAE (sd-turbo's own shape); "
            f"got {len(block_out_channels)} blocks"
        )
    first, second, third, fourth = block_out_channels
    encoder_widths = (third, second, first, first)
    decoder_widths = (fourth, fourth, third, second)
    return tuple(zip(encoder_widths, decoder_widths, strict=True))


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _pad_to_multiple(image: Tensor, multiple: int = _SIZE_MULTIPLE) -> Tensor:
    height, width = image.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return image
    # Reflect rather than zero: a black band at the frame edge is a structure the VAE has
    # never seen, and it would leak into the decoded image well inside the crop-back region.
    return F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")


class Pix2PixTurboTranslator(nn.Module):
    """Implements the `Translator` protocol over sd-turbo with LoRA adapters.

    Owns its optimizers internally, same as every other backbone (`engine/trainer.py` never
    touches an optimizer). Only LoRA parameters, the UNet's `conv_in` and the four decoder
    skip convs are trainable; everything else is frozen, which is also the lever DRaFT
    identifies as the most effective guard against reward hacking (PLAN.md §4).
    """

    def __init__(
        self,
        components: SDTurboComponents,
        prompt: str,
        lora_rank_unet: int = 8,
        lora_rank_vae: int = 4,
        net_d: str = "basic",
        ndf: int = 64,
        gan_mode: str = "vanilla",
        lr: float = 1.0e-4,
        loss_l2: float = 1.0,
        loss_lpips: float = 5.0,
        loss_gan: float = 0.0,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.loss_l2 = loss_l2
        self.loss_lpips = loss_lpips
        self.loss_gan = loss_gan
        self.amp = amp
        self.amp_dtype = amp_dtype

        self.vae = _prepare_vae(components.vae, lora_rank_vae)
        self.scaling_factor = float(components.vae.config["scaling_factor"])
        self.unet = components.unet
        self.unet.add_adapter(
            LoraConfig(
                r=lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=list(_LORA_TARGETS_UNET),
            )
        )
        self.scheduler = components.scheduler
        # `set_timesteps(timesteps=[999])`, not `set_timesteps(1)`. Upstream calls the latter
        # and gets [999] only because sd-turbo's scheduler config happens to say
        # `timestep_spacing: "trailing"`; under the "leading" default the same call yields
        # [0], and `step()` would then denoise from the wrong end of the chain without
        # erroring. Naming the timestep is the thing the distillation actually fixes.
        self.scheduler.set_timesteps(timesteps=[_TIMESTEP])

        _set_requires_grad(self.unet, False)
        _set_requires_grad(self.vae, False)
        for name, param in self.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        _set_requires_grad(self.unet.conv_in, True)
        for index in range(1, 5):
            _set_requires_grad(getattr(self.vae.decoder, f"skip_conv_{index}"), True)

        # Encoded once: the prompt is a config constant, so its embedding is one too. The
        # text encoder and tokenizer are dropped immediately afterwards -- keeping them would
        # cost ~250MB of device memory for a value that never changes again.
        self.register_buffer("caption", _encode_prompt(components, prompt), persistent=False)

        self.optimizer_g = torch.optim.AdamW(
            [param for param in self.parameters() if param.requires_grad], lr=lr
        )

        self.net_d: nn.Module | None = None
        self.criterion_gan: networks.GANLoss | None = None
        self.optimizer_d: torch.optim.Optimizer | None = None
        if loss_gan > 0.0:
            self.net_d = networks.define_D(
                _INFRARED_CHANNELS + _VISIBLE_CHANNELS, ndf, net_d, _N_LAYERS_D, _NORM
            )
            networks.init_weights(self.net_d, _INIT_TYPE, _INIT_GAIN)
            self.criterion_gan = networks.GANLoss(gan_mode)
            self.optimizer_d = torch.optim.AdamW(
                self.net_d.parameters(), lr=lr, betas=(_D_BETA1, 0.999)
            )

        self.lpips: LearnedPerceptualImagePatchSimilarity | None = None
        if loss_lpips > 0.0:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=_LPIPS_NET, normalize=True)
            _set_requires_grad(self.lpips, False)

        # Fixed at construction, not read from `requires_grad` later: this is what
        # `state_dict()` filters on, and a checkpoint's contents must not depend on whatever
        # mode the module happened to be in when it was written.
        self._trainable_keys = frozenset(
            name for name, param in self.named_parameters() if param.requires_grad
        )

    def translate(self, batch: TranslationBatch) -> Tensor:
        device = next(self.unet.parameters()).device
        return self._generate(batch["infrared"].to(device))

    def fit(
        self,
        batch: TranslationBatch,
        task_loss: Callable[[Tensor, TranslationBatch], Tensor] | None = None,
        task_weight: float = 0.0,
    ) -> dict[str, float]:
        device = next(self.unet.parameters()).device
        real_a = batch["infrared"].to(device)
        real_b = batch["visible"].to(device)
        fake_b = self._generate(real_a)

        stats: dict[str, float] = {}

        if self.net_d is not None and self.criterion_gan is not None:
            assert self.optimizer_d is not None
            _set_requires_grad(self.net_d, True)
            self.optimizer_d.zero_grad()
            fake_ab = torch.cat((real_a, fake_b), dim=1).detach()
            loss_d_fake = self.criterion_gan(self.net_d(fake_ab), False)
            real_ab = torch.cat((real_a, real_b), dim=1)
            loss_d_real = self.criterion_gan(self.net_d(real_ab), True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), _MAX_GRAD_NORM)
            self.optimizer_d.step()
            _set_requires_grad(self.net_d, False)
            stats["loss_d"] = float(loss_d.detach())

        self.optimizer_g.zero_grad()
        loss_l2 = F.mse_loss(fake_b, real_b) * self.loss_l2
        total = loss_l2
        stats["loss_l2"] = float(loss_l2.detach())

        if self.lpips is not None:
            loss_lpips = self.lpips(fake_b, real_b) * self.loss_lpips
            total = total + loss_lpips
            stats["loss_lpips"] = float(loss_lpips.detach())

        if self.net_d is not None and self.criterion_gan is not None:
            fake_ab = torch.cat((real_a, fake_b), dim=1)
            loss_gan = self.criterion_gan(self.net_d(fake_ab), True) * self.loss_gan
            total = total + loss_gan
            stats["loss_gan"] = float(loss_gan.detach())

        if task_loss is not None and task_weight > 0.0:
            detection = task_loss(fake_b, batch)
            total = total + task_weight * detection
            stats["loss_det"] = float(detection.detach())

        total.backward()
        trainable = [param for param in self.parameters() if param.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, _MAX_GRAD_NORM)
        self.optimizer_g.step()
        stats["loss_total"] = float(total.detach())
        return stats

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Only the parameters that train -- see deviation 5 in the module docstring."""
        full = super().state_dict(*args, **kwargs)
        return {key: value for key, value in full.items() if key in self._trainable_keys}

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        """Accept the reduced checkpoint `state_dict()` writes.

        Missing keys are expected and ignored -- they are the frozen base weights, which
        `load_sd_turbo` has already restored bit for bit. *Unexpected* keys are not ignored:
        a checkpoint holding something this model does not have is a wrong checkpoint, and
        `strict=False` alone would accept it silently.
        """
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        if result.unexpected_keys:
            raise TurboTranslatorError(
                f"checkpoint has {len(result.unexpected_keys)} key(s) this translator does "
                f"not define, first is {result.unexpected_keys[0]!r}"
            )
        return result

    def _generate(self, infrared: Tensor) -> Tensor:
        """Thermal `(B, 1, H, W)` in [0, 1] -> visible `(B, 3, H, W)` in [0, 1]."""
        device = infrared.device
        height, width = infrared.shape[-2:]
        control = infrared.expand(-1, _VISIBLE_CHANNELS, -1, -1) * 2.0 - 1.0
        control = _pad_to_multiple(control)
        self._align_scheduler(device)

        # `Any` on the three diffusers call sites: each returns a dataclass-or-tuple union
        # keyed on `return_dict`, which no annotation narrows for us.
        with torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=self._use_amp):
            latents = self._encode(control) * self.scaling_factor
            predicted: Tensor = self.unet(
                latents, _TIMESTEP, encoder_hidden_states=self._caption_for(control)
            ).sample
            stepped: Any = self.scheduler.step(predicted, _TIMESTEP, latents)
            denoised = stepped.prev_sample.to(predicted.dtype)
            # The vendored decoder forward reads these off the encoder's last pass, which is
            # the whole skip connection: it must be set between encode and decode, not once.
            decoder = cast(Any, self.vae.decoder)
            decoder.incoming_skip_acts = cast(Any, self.vae.encoder).current_down_blocks
            decoded = cast(Any, self.vae.decode(denoised / self.scaling_factor)).sample

        image = cast(Tensor, decoded).float().clamp(-1.0, 1.0)
        return ((image + 1.0) / 2.0)[..., :height, :width]

    def _encode(self, control: Tensor) -> Tensor:
        distribution = cast(Any, self.vae.encode(control)).latent_dist
        # Upstream samples the posterior in both training and inference. Sampling at eval
        # would make an exported image depend on RNG state, so two evaluations of the same
        # checkpoint would not agree -- unacceptable in a measuring instrument. Training
        # keeps upstream's stochasticity, which is where it was doing work.
        return distribution.sample() if self.training else distribution.mode()

    def _align_scheduler(self, device: torch.device) -> None:
        """`DDPMScheduler` is not an `nn.Module`, so `.to(device)` never reaches its tensors.

        `step()` indexes `alphas_cumprod` with the timestep, and indexing a CPU tensor with a
        CUDA index raises. Upstream solved this by hardcoding `.cuda()` at import time.
        """
        if self.scheduler.alphas_cumprod.device != device:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

    def _caption_for(self, control: Tensor) -> Tensor:
        caption = self.get_buffer("caption")
        return caption.expand(control.shape[0], -1, -1)

    @property
    def _use_amp(self) -> bool:
        # CUDA only: on CPU autocast buys nothing and makes the test suite's numbers depend
        # on which kernels happen to have bfloat16 paths.
        return self.amp and next(self.unet.parameters()).device.type == "cuda"


def _prepare_vae(vae: AutoencoderKL, lora_rank: int) -> AutoencoderKL:
    """Patch in the vendored skip forwards, add the four skip convs, then the adapters."""
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    for index, (in_channels, out_channels) in enumerate(
        skip_connection_channels(list(vae.config["block_out_channels"])), start=1
    ):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.constant_(conv.weight, _SKIP_CONV_INIT)
        setattr(vae.decoder, f"skip_conv_{index}", conv)
    # Plain flags the vendored decoder forward reads; `nn.Module.__setattr__` is typed for
    # tensors and submodules only, which is what the cast is for.
    decoder = cast(Any, vae.decoder)
    decoder.ignore_skip = False
    decoder.gamma = 1
    vae.add_adapter(
        LoraConfig(
            r=lora_rank, init_lora_weights="gaussian", target_modules=list(_LORA_TARGETS_VAE)
        ),
        adapter_name="vae_skip",
    )
    return vae


@torch.no_grad()
def _encode_prompt(components: SDTurboComponents, prompt: str) -> Tensor:
    tokens = components.tokenizer(
        prompt,
        max_length=components.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return components.text_encoder(tokens)[0].detach().clone()
