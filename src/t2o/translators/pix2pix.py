"""M1's GAN backbone (`PLAN.md` §10 Phase 1) -- the go/no-go gate before diffusion.

Vendored from `junyanz/pytorch-CycleGAN-and-pix2pix` at `2a7afba` (`third_party/pix2pix/
networks.py`, `define_G`/`define_D`/`GANLoss`, verbatim, never edited in place --
invariant 4). This module is the wrapper that owns everything the upstream repo used to put
in `BaseModel`/`Pix2PixModel`/`train.py`: device placement, optimizers, and the
`backward_D`/`backward_G` composition (`models/pix2pix_model.py` in the vendor source,
not itself vendored -- reimplemented here against our own `LossConfig`/`Translator`
contracts instead).

Three deliberate deviations from upstream, each cited where it happens below:

1. `define_G`/`define_D` accept but never *apply* `init_type`/`init_gain` at this pinned
   commit -- that used to be `init_net`'s job, called from `BaseModel`, which is out of
   vendor scope. `init_net` also hardcodes CUDA placement (PLAN.md §7: "our wrapper owns
   device placement"), so this wrapper calls `networks.init_weights` directly instead and
   does its own `.to(device)`.
2. Generator defaults to `resnet_9blocks`, not the paper's `unet_256` -- `UnetGenerator`
   needs input divisible by 2**8=256; the custom dataset is 640x480 and any `train.crop`
   only guarantees stride-32 divisibility (PLAN.md §8). `ResnetGenerator` only needs
   divisible-by-4.
3. Reconstruction/adversarial loss weights come from the shared `LossConfig.{l2,lpips,gan}`
   (`StubTranslator`'s own convention), not pix2pix's own `lambda_L1=100` -- every backbone
   is meant to be comparable through the same three knobs (PLAN.md §8).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from third_party.pix2pix import networks
from torch import Tensor, nn
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from t2o.data.dataset import TranslationBatch

# Our data contract always produces 1-channel infrared / 3-channel visible (PLAN.md §9) --
# hardcoded here exactly like StubTranslator hardcodes its own conv channel counts, not a
# config field.
_INFRARED_CHANNELS = 1
_VISIBLE_CHANNELS = 3
_NORM = "batch"  # pix2pix's own paper default
_INIT_TYPE = "normal"
_INIT_GAIN = 0.02
_BETA1 = 0.5
_N_LAYERS_D = 3  # only consulted when net_d == "n_layers"
_LPIPS_NET = "alex"


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


class Pix2PixTranslator(nn.Module):
    """Implements the `Translator` protocol over a vendored pix2pix generator/discriminator.

    Owns `optimizer_g`/`optimizer_d` internally, same as `StubTranslator` -- `engine/
    trainer.py` never touches optimizers (M0.8 decision). `net_d`/the LPIPS network are only
    built when their loss weight is positive, the same "clean no-op at weight 0" discipline
    `CouplingConfig` already uses for the frozen in-loop detector -- by default
    `loss_gan=0.0` (`LossConfig`'s own docstring: "the cheaper and more stable starting
    point"), so no discriminator exists unless a caller actually raises the weight.
    """

    def __init__(
        self,
        net_g: str = "resnet_9blocks",
        net_d: str = "basic",
        ngf: int = 64,
        ndf: int = 64,
        gan_mode: str = "vanilla",
        lr: float = 1.0e-4,
        loss_l2: float = 1.0,
        loss_lpips: float = 5.0,
        loss_gan: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss_l2 = loss_l2
        self.loss_lpips = loss_lpips
        self.loss_gan = loss_gan

        self.net_g = networks.define_G(_INFRARED_CHANNELS, _VISIBLE_CHANNELS, ngf, net_g, _NORM)
        networks.init_weights(self.net_g, _INIT_TYPE, _INIT_GAIN)
        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, betas=(_BETA1, 0.999))

        self.net_d: nn.Module | None = None
        self.criterion_gan: networks.GANLoss | None = None
        self.optimizer_d: torch.optim.Optimizer | None = None
        if loss_gan > 0.0:
            self.net_d = networks.define_D(
                _INFRARED_CHANNELS + _VISIBLE_CHANNELS, ndf, net_d, _N_LAYERS_D, _NORM
            )
            networks.init_weights(self.net_d, _INIT_TYPE, _INIT_GAIN)
            self.criterion_gan = networks.GANLoss(gan_mode)
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(), lr=lr, betas=(_BETA1, 0.999)
            )

        self.lpips: LearnedPerceptualImagePatchSimilarity | None = None
        if loss_lpips > 0.0:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=_LPIPS_NET, normalize=True)
            _set_requires_grad(self.lpips, False)

    def translate(self, batch: TranslationBatch) -> Tensor:
        device = next(self.net_g.parameters()).device
        raw = self.net_g(batch["infrared"].to(device))  # Tanh output, [-1, 1]
        return (raw + 1.0) / 2.0  # Translator protocol's [0, 1] contract

    def fit(
        self,
        batch: TranslationBatch,
        task_loss: Callable[[Tensor, TranslationBatch], Tensor] | None = None,
        task_weight: float = 0.0,
    ) -> dict[str, float]:
        # Everything below stays in the Translator protocol's [0, 1] convention -- the
        # [-1, 1] remap only ever exists transiently inside net_g's Tanh output. The affine
        # map back to [0, 1] is linear and differentiable, so nothing here is any less
        # un-detached than upstream's own raw-Tanh convention would be; staying in one
        # convention throughout also means the discriminator's two input channels (infrared,
        # visible) are always on the same scale, real pair or fake pair alike.
        device = next(self.net_g.parameters()).device
        real_a = batch["infrared"].to(device)
        real_b = batch["visible"].to(device)
        fake_b = (self.net_g(real_a) + 1.0) / 2.0

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
        self.optimizer_g.step()
        stats["loss_total"] = float(total.detach())
        return stats
