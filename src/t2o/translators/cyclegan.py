"""Stage 2 of the paired->unpaired cascade: plain CycleGAN (cyclegan-cascade Q7).

Networks, `GANLoss` and `ImagePool` are vendored verbatim from
`junyanz/pytorch-CycleGAN-and-pix2pix` at `2a7afba` (`third_party/pix2pix/networks.py`,
`third_party/pix2pix/image_pool.py`), the same pin stage 1's pix2pix uses. This module
reimplements what upstream keeps in `models/cycle_gan_model.py` against our own `Translator`
contract: two generators, two discriminators, one Adam over both generators and one over both
discriminators, generators stepped first (`optimize_parameters`, `:182-196`).

Domain A is stage 1's RGB output (or thermal at `infrared_channels: 1`); domain B is visible.
D_X judges domain X, as in the paper. That is the reverse of upstream's naming, whose `netD_A`
judges B.

The detection term (`task_weight * task_loss`) goes on G_AB's output only. It is added to the
generator step, during which both discriminators have `requires_grad` off, and the
discriminator step reads detached images from the pools, so it never reaches a discriminator.

Deviations from upstream, each cited where it happens:

1. `init_net` is skipped for `networks.init_weights` plus our own device placement, as in
   `translators/pix2pix.py` (its deviation 1).
2. The learning-rate schedule (plan Q10) is upstream's `linear` policy -- constant for the
   first half, then linear to zero -- but stepped per batch, with the step count from
   `build_translator`'s corpus count, rather than per epoch (`networks.py:57-60`). The same
   pattern as `translators/modtr.py`.
3. The adversarial weight is the shared `loss.gan` (upstream: an implicit 1), so backbones stay
   comparable through one knob.
4. Upstream's image tensors are in [-1, 1]. They are here too, inside this module, so
   `lambda_cycle: 10` weighs the same L1 it weighs upstream. Only `translate()` and the
   detection term see the `Translator` protocol's [0, 1].
5. The image pools are not checkpointed. A resumed run starts them empty and refills them
   within 50 images.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from third_party.pix2pix import networks
from third_party.pix2pix.image_pool import ImagePool
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

from t2o.data.dataset import TranslationBatch

_VISIBLE_CHANNELS = 3
_NORM = "instance"  # upstream CycleGAN's default (`--norm instance`)
_GAN_MODE = "lsgan"  # upstream CycleGAN's default (`--gan_mode lsgan`)
_INIT_TYPE = "normal"
_INIT_GAIN = 0.02
_BETA1 = 0.5
_N_LAYERS_D = 3  # only consulted when net_d == "n_layers"
_POOL_SIZE = 50  # upstream's `--pool_size`
# Upstream trains `n_epochs` 100 at a constant rate and `n_epochs_decay` 100 decaying (plan Q10).
_CONSTANT_FRACTION = 0.5


def _set_requires_grad(modules: tuple[nn.Module, ...], requires_grad: bool) -> None:
    for module in modules:
        for param in module.parameters():
            param.requires_grad = requires_grad


def _linear_decay(total_batches: int) -> Callable[[int], float]:
    """LR factor per optimiser step: 1.0 for the first half, then linear to zero at the end."""
    constant = int(total_batches * _CONSTANT_FRACTION)
    decay = max(total_batches - constant, 1)

    def factor(step: int) -> float:
        return max(0.0, min(1.0, (total_batches - step) / decay))

    return factor


def _to_signed(image: Tensor) -> Tensor:
    return image * 2.0 - 1.0


def _to_unit(image: Tensor) -> Tensor:
    return (image + 1.0) / 2.0


class CycleGANTranslator(nn.Module):
    """Implements the `Translator` protocol over two vendored generators and discriminators.

    Owns both optimizers and both schedules, as every backbone does (`engine/trainer.py` never
    touches them). `total_batches` is every training batch this translator will see, across
    every stage: the length of the learning-rate schedule.
    """

    def __init__(
        self,
        total_batches: int,
        infrared_channels: int = 3,
        net_g: str = "resnet_9blocks",
        net_d: str = "basic",
        ngf: int = 64,
        ndf: int = 64,
        lr: float = 2.0e-4,
        loss_gan: float = 1.0,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
    ) -> None:
        super().__init__()
        self.loss_gan = loss_gan
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        a, b = infrared_channels, _VISIBLE_CHANNELS
        self.net_g_ab = networks.define_G(a, b, ngf, net_g, _NORM)
        self.net_g_ba = networks.define_G(b, a, ngf, net_g, _NORM)
        self.net_d_a = networks.define_D(a, ndf, net_d, _N_LAYERS_D, _NORM)
        self.net_d_b = networks.define_D(b, ndf, net_d, _N_LAYERS_D, _NORM)
        for net in (self.net_g_ab, self.net_g_ba, self.net_d_a, self.net_d_b):
            networks.init_weights(net, _INIT_TYPE, _INIT_GAIN)
        self.criterion_gan = networks.GANLoss(_GAN_MODE)

        self.pool_a = ImagePool(_POOL_SIZE)
        self.pool_b = ImagePool(_POOL_SIZE)

        self.optimizer_g = torch.optim.Adam(
            itertools.chain(self.net_g_ab.parameters(), self.net_g_ba.parameters()),
            lr=lr,
            betas=(_BETA1, 0.999),
        )
        self.optimizer_d = torch.optim.Adam(
            itertools.chain(self.net_d_a.parameters(), self.net_d_b.parameters()),
            lr=lr,
            betas=(_BETA1, 0.999),
        )
        self.scheduler_g = LambdaLR(self.optimizer_g, _linear_decay(total_batches))
        self.scheduler_d = LambdaLR(self.optimizer_d, _linear_decay(total_batches))

    def translate(self, batch: TranslationBatch) -> Tensor:
        device = next(self.net_g_ab.parameters()).device
        return _to_unit(self.net_g_ab(_to_signed(batch["infrared"].to(device))))

    def fit(
        self,
        batch: TranslationBatch,
        task_loss: Callable[[Tensor, TranslationBatch], Tensor] | None = None,
        task_weight: float = 0.0,
    ) -> dict[str, float]:
        device = next(self.net_g_ab.parameters()).device
        real_a = _to_signed(batch["infrared"].to(device))
        real_b = _to_signed(batch["visible"].to(device))
        fake_b = self.net_g_ab(real_a)
        fake_a = self.net_g_ba(real_b)
        discriminators = (self.net_d_a, self.net_d_b)
        stats: dict[str, float] = {}

        # Generators first, discriminators frozen (upstream `optimize_parameters`, `:187-190`).
        _set_requires_grad(discriminators, False)
        self.optimizer_g.zero_grad()
        loss_gan = (
            self.criterion_gan(self.net_d_b(fake_b), True)
            + self.criterion_gan(self.net_d_a(fake_a), True)
        ) * self.loss_gan
        loss_cycle = (
            F.l1_loss(self.net_g_ba(fake_b), real_a) + F.l1_loss(self.net_g_ab(fake_a), real_b)
        ) * self.lambda_cycle
        total = loss_gan + loss_cycle
        stats["loss_gan"] = float(loss_gan.detach())
        stats["loss_cycle"] = float(loss_cycle.detach())

        if self.lambda_identity > 0.0:
            # The identity weight multiplies the cycle weight (upstream `:162,165`).
            loss_idt = (
                F.l1_loss(self.net_g_ab(real_b), real_b) + F.l1_loss(self.net_g_ba(real_a), real_a)
            ) * (self.lambda_cycle * self.lambda_identity)
            total = total + loss_idt
            stats["loss_idt"] = float(loss_idt.detach())

        if task_loss is not None and task_weight > 0.0:
            detection = task_loss(_to_unit(fake_b), batch)
            total = total + task_weight * detection
            stats["loss_det"] = float(detection.detach())

        total.backward()
        self.optimizer_g.step()
        self.scheduler_g.step()
        stats["loss_total"] = float(total.detach())

        # Then the discriminators, on pooled history of detached fakes (upstream `:192-196`).
        _set_requires_grad(discriminators, True)
        self.optimizer_d.zero_grad()
        loss_d = self._discriminator_loss(
            self.net_d_b, real_b, self.pool_b.query(fake_b)
        ) + self._discriminator_loss(self.net_d_a, real_a, self.pool_a.query(fake_a))
        loss_d.backward()
        self.optimizer_d.step()
        self.scheduler_d.step()
        stats["loss_d"] = float(loss_d.detach())
        return stats

    def get_extra_state(self) -> dict[str, Any]:
        """Both Adams and both schedules' positions, so a resume continues the decay.

        Without them a resumed 200-epoch run would restart at the full learning rate, however
        far into the decay it had got (plan Q10).
        """
        return {
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.optimizer_g.load_state_dict(state["optimizer_g"])
        self.optimizer_d.load_state_dict(state["optimizer_d"])
        self.scheduler_g.load_state_dict(state["scheduler_g"])
        self.scheduler_d.load_state_dict(state["scheduler_d"])

    def _discriminator_loss(self, net_d: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
        """Upstream `backward_D_basic` (`:121-141`), without the `backward()` call."""
        loss_real = self.criterion_gan(net_d(real), True)
        loss_fake = self.criterion_gan(net_d(fake.detach()), False)
        return (loss_real + loss_fake) * 0.5
