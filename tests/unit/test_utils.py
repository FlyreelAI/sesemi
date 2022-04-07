import pytest
import torch

from sesemi.utils import reduce_tensor


@pytest.mark.parametrize(
    "tensor,weights,reduction,expected",
    [
        (torch.tensor([1.0, 1.0, 1.0]), None, "mean", torch.tensor(1.0)),
        (torch.tensor([1.0, 1.0, 1.0]), None, "sum", torch.tensor(3.0)),
        (torch.tensor([1.0, 1.0, 1.0]), None, "weighted_mean", torch.tensor(1.0)),
        (torch.tensor([1.0, 1.0, 1.0]), None, "none", torch.tensor([1.0, 1.0, 1.0])),
    ],
)
def test_reduce_tensor(tensor, weights, reduction, expected):
    assert torch.allclose(
        reduce_tensor(tensor, weights=weights, reduction=reduction), expected
    )
