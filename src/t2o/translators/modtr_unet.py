"""ModTr's transformation network h(.), reimplemented on torchvision (modtr-baseline ADR-01).

ModTr (Medeiros et al., ECCV 2024) ships no network of its own. Upstream at
`heitorrapela/ModTr@cace11e` builds `segmentation_models_pytorch.Unet` in
`src/translator.py:25-30`: encoder `resnet34`, depth 5, `encoder_weights="imagenet"`, and the
segmentation head's last layer swapped for a sigmoid. The paper says the same: Sec. 4.1(b) names
"PyTorch Segmentation Models", Table 3 gives 24.4M parameters for this ResNet34 default, and
Sec. 3(b) says the framework "is not limited by the translation architecture". smp is not a `t2o`
dependency and would bring in `timm`. Its files cannot be vendored standalone without editing them
(invariant 4), so this module rebuilds `smp.Unet`'s defaults (smp 0.3.3) layer for layer:

- **Encoder** -- torchvision `resnet34`, cut into smp's five stages
  (`encoders/resnet.py:47-55`, `ResNetEncoder.get_stages`): conv1-bn1-relu (/2, 64ch),
  maxpool-layer1 (/4, 64ch), layer2 (/8, 128ch), layer3 (/16, 256ch), layer4 (/32, 512ch).
  `fc` and `avgpool` are dropped, as smp drops them (`:44-45`).
- **Decoder** -- five blocks of decoder channels (256, 128, 64, 32, 16), no centre block
  (`decoders/unet/model.py:62, :83`). Each block upsamples x2 by nearest neighbour, concatenates
  the encoder skip, then runs two 3x3 conv-BN-ReLU layers with bias-free convs
  (`decoders/unet/decoder.py:35-43`, `base/modules.py:27-47`). Skips are the stage outputs from
  /16 down to /2. The last block has none (`decoder.py:108-121`).
- **Head** -- a 3x3 conv to 3 channels, then a sigmoid (`base/heads.py:5-10`; upstream
  `translator.py:25-30` for the sigmoid).
- **Initialisation** -- smp's `initialize_decoder` and `initialize_head` (`base/initialization.py`):
  Kaiming-uniform decoder convs, BN at (1, 0), Xavier-uniform head with zero bias. The encoder keeps
  torchvision's own initialisation, or its ImageNet weights.

The input is a 3-channel image of stride-32 size, as for `smp.Unet(in_channels=3)`. Repeating
1-channel thermal to 3 channels (upstream `src/engine/utils/utils.py:96-97`) and the output fusion
`clip(x * h(x), 0, 1)` (the paper's Eq. 3, ModTr-dot; upstream `utils.py:507-516`) are ModTr's
method, not its network. They live in the `Translator` wrapper, so this module is exactly the
network whose 24.4M parameters Table 3 counts.

Deviations from upstream:

1. The network is rebuilt from torchvision modules instead of imported from smp. The parameter
   count test (`tests/test_modtr_unet.py`) guards structural drift. It cannot catch every detail,
   such as the upsample mode, so those details are cited line by line above.
2. `pretrained=True` loads torchvision's `ResNet34_Weights.IMAGENET1K_V1`
   (`resnet34-b627a593.pth`). smp 0.3.3 loads `"imagenet"` from `pretrainedmodels`' settings,
   which point at `resnet34-333f7ec4.pth`. That these are the same weights under torchvision's
   renamed files is believed, not verified (ADR-01, Consequences).
3. State-dict keys are this module's own (`stem`, `stage1`, ..., `decoder`, `head`), not smp's.
   No upstream checkpoint is ever loaded, so nothing depends on smp's key names.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet34_Weights, resnet34

# smp.Unet's defaults (`decoders/unet/model.py:62`) and the ResNet34 stage widths
# (`encoders/resnet.py:140`), deepest stage first.
_DECODER_CHANNELS = (256, 128, 64, 32, 16)
_SKIP_CHANNELS = (256, 128, 64, 64, 0)
_HEAD_CHANNELS = 512
_IN_CHANNELS = 3
_OUT_CHANNELS = 3
# Five /2 stages: a side not divisible by 2**5 makes an upsampled map one pixel off its skip.
_OUTPUT_STRIDE = 32


class ModTrShapeError(ValueError):
    """The input is not 3-channel, or its height or width is not a multiple of the stride."""


def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    # smp's `Conv2dReLU` with `use_batchnorm=True`: the conv has no bias because BN follows it.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = _conv_bn_relu(in_channels + skip_channels, out_channels)
        self.conv2 = _conv_bn_relu(out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor | None) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class ModTrUNet(nn.Module):
    """`smp.Unet("resnet34", classes=3, activation="sigmoid")` on torchvision.

    Maps (B, 3, H, W) to (B, 3, H, W) in [0, 1]. `pretrained=False` exists for tests: it builds
    the same graph with no weight download.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stage1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4

        in_channels = (_HEAD_CHANNELS, *_DECODER_CHANNELS[:-1])
        self.decoder = nn.ModuleList(
            _DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(
                in_channels, _SKIP_CHANNELS, _DECODER_CHANNELS, strict=True
            )
        )
        self.head = nn.Conv2d(_DECODER_CHANNELS[-1], _OUT_CHANNELS, kernel_size=3, padding=1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.head.weight)
        assert self.head.bias is not None
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if x.shape[1] != _IN_CHANNELS or height % _OUTPUT_STRIDE or width % _OUTPUT_STRIDE:
            raise ModTrShapeError(
                f"ModTrUNet needs (B, {_IN_CHANNELS}, H, W) with H and W divisible by "
                f"{_OUTPUT_STRIDE}, got {tuple(x.shape)}"
            )
        f1 = self.stem(x)
        f2 = self.stage1(f1)
        f3 = self.stage2(f2)
        f4 = self.stage3(f3)
        y = self.stage4(f4)
        for block, skip in zip(self.decoder, (f4, f3, f2, f1, None), strict=True):
            y = block(y, skip)
        return torch.sigmoid(self.head(y))
