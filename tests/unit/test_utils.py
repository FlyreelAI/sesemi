import pytest
import torch

from sesemi.utils import reduce_tensor, compute_num_devices


@pytest.mark.parametrize(
    "tensor,weights,reduction,expected",
    [
        (torch.tensor([1.0, 2.0, 3.0]), None, "mean", torch.tensor(2.0)),
        (torch.tensor([1.0, 2.0, 3.0]), None, "sum", torch.tensor(6.0)),
        (torch.tensor([1.0, 2.0, 3.0]), None, "weighted_mean", torch.tensor(2.0)),
        (torch.tensor([1.0, 2.0, 3.0]), None, "none", torch.tensor([1.0, 2.0, 3.0])),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 4.0, 0.0]),
            "mean",
            torch.tensor(3.0),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 4.0, 0.0]),
            "sum",
            torch.tensor(9.0),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 4.0, 0.0]),
            "weighted_mean",
            torch.tensor(1.8),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 4.0, 0.0]),
            "none",
            torch.tensor([1.0, 8.0, 0.0]),
        ),
    ],
)
def test_reduce_tensor(tensor, weights, reduction, expected):
    assert torch.allclose(
        reduce_tensor(tensor, weights=weights, reduction=reduction), expected
    )


@pytest.mark.parametrize(
    "accelerator,devices,expected",
    [
        ("gpu", 5, 5),
        ("gpu", None, 1),
    ],
)
def test_compute_num_devices(accelerator, devices, expected, mocker):
    mocker.patch(
        "pytorch_lightning.accelerators.gpu.GPUAccelerator.auto_device_count",
        side_effect=lambda *args: 1,
    )
    mocker.patch(
        "pytorch_lightning.utilities.device_parser._get_all_available_gpus",
        side_effect=lambda *args: list(range(8)),
    )
    assert compute_num_devices(accelerator, devices) == expected
