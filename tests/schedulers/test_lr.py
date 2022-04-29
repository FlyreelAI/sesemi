import numpy as np
import pytest
import torch
import torch.nn as nn

from torch.optim import SGD

from sesemi.schedulers.lr import PolynomialLR


@pytest.mark.parametrize(
    "lr,warmup_lr,lr_pow,max_iters,warmup_iters,warmup_epochs,iters_per_epoch,step,expected",
    [
        (0.1, 0.001, 0.5, 1000, 100, None, None, 0, 0.001),
        (0.1, 0.001, 0.5, 1000, 100, None, None, 100, 0.1),
        (0.1, 0.001, 0.5, 1000, 100, None, None, 1000, 0.0),
        (0.1, 0.001, 0.5, 1000, 100, None, None, 550, (0.5**0.5) * 0.1),
        (0.1, 0.001, 0.5, 1000, None, 10, 10, 0, 0.001),
        (0.1, 0.001, 0.5, 1000, None, 10, 10, 100, 0.1),
        (0.1, 0.001, 0.5, 1000, None, 10, 10, 1000, 0.0),
        (0.1, 0.001, 0.5, 1000, None, 10, 10, 550, (0.5**0.5) * 0.1),
        (0.1, 0.001, 0.8, 1000, None, 10, 6, 0, 0.001),
        (0.1, 0.001, 0.8, 1000, None, 10, 6, 60, 0.1),
        (0.1, 0.001, 0.8, 1000, None, 10, 6, 1000, 0.0),
        (0.1, 0.001, 0.8, 1000, None, 10, 6, 530, (0.5**0.8) * 0.1),
    ],
)
def test_polynomial_lr(
    lr,
    warmup_lr,
    lr_pow,
    max_iters,
    warmup_iters,
    warmup_epochs,
    iters_per_epoch,
    step,
    expected,
):
    module = nn.Linear(100, 10)
    optimizer = SGD(module.parameters(), lr)
    scheduler = PolynomialLR(
        optimizer,
        warmup_lr=warmup_lr,
        lr_pow=lr_pow,
        max_iters=max_iters,
        warmup_iters=warmup_iters,
        warmup_epochs=warmup_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    for _ in range(step):
        scheduler.step()

    curr_lr = scheduler.get_lr()
    assert len(curr_lr) == 1
    assert np.isclose(curr_lr[0], expected)
