import pytest
import torch

from sesemi.losses import softmax_mse_loss, kl_div_loss


@pytest.mark.parametrize(
    "inputs,targets",
    [
        (
            torch.ones((1, 10), dtype=torch.float32),
            torch.ones((1, 10), dtype=torch.float32),
        ),
        (
            torch.zeros((1, 10), dtype=torch.float32),
            torch.zeros((1, 10), dtype=torch.float32),
        ),
    ],
)
def test_softmax_mse_loss(inputs, targets):
    outputs = softmax_mse_loss(inputs, targets)
    assert outputs.shape == (inputs.shape[0],)


@pytest.mark.parametrize(
    "inputs,targets",
    [
        (
            torch.ones((1, 10), dtype=torch.float32),
            torch.ones((1, 10), dtype=torch.float32),
        ),
        (
            torch.zeros((1, 10), dtype=torch.float32),
            torch.zeros((1, 10), dtype=torch.float32),
        ),
    ],
)
def test_kl_div_loss(inputs, targets):
    outputs = kl_div_loss(inputs, targets)
    assert outputs.shape == (inputs.shape[0],)
