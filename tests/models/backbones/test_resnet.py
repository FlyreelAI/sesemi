import pytest
import torch

from sesemi.models.backbones.resnet import CIFARResNet, WRN
from ...utils import count_parameters, round_nth_digit, check_model_frozen_state


@pytest.mark.parametrize("n", list(range(1, 31)))
def test_cifar_resnet(n):
    net = CIFARResNet(n=n)
    outputs = net(torch.ones((2, 3, 32, 32), dtype=torch.float32))
    assert outputs.shape == (2, net.out_features)

    conv_layers = [x for x in net.modules() if isinstance(x, torch.nn.Conv2d)]
    assert len(conv_layers) == (6 * n + 1)

    for conv in conv_layers:
        assert conv.kernel_size == (3, 3)
        assert conv.padding == (1, 1)


@pytest.mark.parametrize("freeze", [False, True])
def test_cifar_resnet_freeze(freeze):
    check_model_frozen_state(CIFARResNet(n=18, freeze=freeze), freeze)


@pytest.mark.parametrize(
    "depth,k,expected_param_count",
    [
        (40, 1, 600_000),
        (40, 2, 2_200_000),
        (40, 4, 8_900_000),
        (40, 8, 35_700_000),
        (28, 10, 36_500_000),
        (28, 12, 52_500_000),
        (22, 8, 17_200_000),
        (22, 10, 26_800_000),
        (16, 8, 11_000_000),
        (16, 10, 17_100_000),
    ],
)
def test_wrn_param_count(depth, k, expected_param_count):
    net = WRN(depth, k)
    outputs = net(torch.ones((2, 3, 32, 32), dtype=torch.float32))
    assert outputs.shape == (2, net.out_features)

    conv_layers = [x for x in net.modules() if isinstance(x, torch.nn.Conv2d)]

    if k == 1:
        assert len(conv_layers) == (depth - 1)
    else:
        assert len(conv_layers) == depth

    net_param_count = count_parameters(net)

    rounded_param_count = round_nth_digit(net_param_count, 5)
    assert rounded_param_count == expected_param_count


@pytest.mark.parametrize("freeze", [False, True])
def test_wrn_freeze(freeze):
    check_model_frozen_state(WRN(28, 10, freeze=freeze), freeze)
