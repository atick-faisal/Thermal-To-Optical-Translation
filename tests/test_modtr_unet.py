"""ModTr's U-Net (modtr-baseline ADR-01): the network alone, before any `Translator` wrapper.

Built with `pretrained=False` throughout, so no ImageNet weights are downloaded. The graph is the
same either way.
"""

from __future__ import annotations

import pytest
import torch

from t2o.translators.modtr_unet import ModTrShapeError, ModTrUNet


@pytest.fixture(scope="module")
def network() -> ModTrUNet:
    torch.manual_seed(0)
    return ModTrUNet(pretrained=False)


def test_parameter_count_matches_the_paper(network: ModTrUNet) -> None:
    # ModTr's Table 3 reports 24.4M for the ResNet34 U-Net, and a shape-equivalent smp stand-in
    # measured 24.44M in the ADR-01 spike. Drift here means the arm is no longer ModTr's network.
    n_params = sum(p.numel() for p in network.parameters())
    assert round(n_params / 1e6, 2) == 24.44


@pytest.mark.parametrize(("height", "width"), [(64, 64), (96, 128)])
def test_forward_keeps_the_shape_and_stays_in_unit_range(
    network: ModTrUNet, height: int, width: int
) -> None:
    x = torch.rand((2, 3, height, width), generator=torch.Generator().manual_seed(0))
    y = network(x)

    assert y.shape == (2, 3, height, width)
    assert y.dtype == torch.float32
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_forward_is_grad_connected_to_encoder_and_decoder(network: ModTrUNet) -> None:
    network.zero_grad()
    network(torch.rand((1, 3, 64, 64))).sum().backward()

    for part in (network.stem, network.stage4, network.decoder, network.head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())


@pytest.mark.parametrize("shape", [(1, 3, 64, 48), (1, 1, 64, 64)])
def test_rejects_a_non_stride_32_or_non_rgb_input(
    network: ModTrUNet, shape: tuple[int, int, int, int]
) -> None:
    with pytest.raises(ModTrShapeError, match="divisible by 32"):
        network(torch.rand(shape))
