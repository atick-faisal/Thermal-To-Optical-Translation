"""A tiny CPU stand-in translator.

Load-bearing, not a test fixture (PLAN.md §9, `StubTranslatorConfig`'s own docstring):
`Pix2Pix_Turbo` hardcodes `.cuda()` in `__init__` and cannot even be imported on the Mac, so
this is the only way the data -> translate -> export -> eval path stays locally testable on
a CPU dev machine before anything reaches the server.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from t2o.data.dataset import TranslationBatch


class StubTranslator(nn.Module):
    """Two convs, infrared -> visible. Implements the `Translator` protocol.

    `Sigmoid` bounds the output to `[0, 1]` natively -- `TranslationSample`'s convention --
    so, unlike a real diffusion backbone, there is no `[-1, 1]` bridge to own here. Owns a
    private Adam optimizer so `fit()` is a complete, self-contained step regardless of what
    the outer training loop looks like.
    """

    def __init__(self, hidden_channels: int = 16, lr: float = 1e-2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def translate(self, batch: TranslationBatch) -> Tensor:
        device = next(self.parameters()).device
        return self.net(batch["infrared"].to(device))

    def fit(self, batch: TranslationBatch) -> dict[str, float]:
        self._optimizer.zero_grad()
        pred = self.translate(batch)
        target = batch["visible"].to(pred.device)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        self._optimizer.step()
        return {"loss_l2": float(loss.detach())}
