"""The `Translator` interface every backbone implements (PLAN.md invariant 3).

A translator is anything that can `fit()` on a batch and `translate()` a batch into an
image. That is the whole contract: swapping pix2pix for pix2pix-turbo for LBBDM is a
config change (`config.translator.backbone`), never a change to the code that drives
training or evaluation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from t2o.data.dataset import TranslationBatch


@runtime_checkable
class Translator(Protocol):
    """Structural interface: any object with these two methods qualifies.

    Each backbone owns its own optimizer(s) and training procedure internally -- a GAN's
    discriminator-then-generator steps, a single step for a distilled diffusion model --
    so the interface stays backbone-agnostic rather than assuming one optimizer or one
    loss term.
    """

    def fit(self, batch: TranslationBatch) -> dict[str, float]:
        """Run one full optimisation step. Returns scalar losses for logging."""
        ...

    def translate(self, batch: TranslationBatch) -> Tensor:
        """Generate visible-domain images from `batch["infrared"]`.

        Returns `(B, 3, H, W)` float32 in `[0, 1]` -- `TranslationSample`'s native
        convention, matching `batch["visible"]`. Must stay grad-connected to the
        translator's own parameters: the detection-consistency coupling loss (M0.7)
        backprops through this call.
        """
        ...
