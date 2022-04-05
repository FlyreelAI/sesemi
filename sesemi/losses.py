#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Loss functions and modules."""
from typing import Callable
from torch import Tensor

import torch.nn.functional as F


def softmax_mse_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes the mean squared error loss after softmax."""

    return F.mse_loss(
        F.softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction="none"
    )


def kl_div_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes the Kullback–Leibler divergence loss between two probability distributions."""

    return F.kl_div(
        F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction="none"
    )


def get_loss_fn(loss_fn: str) -> Callable[[Tensor, Tensor], Tensor]:
    """Gets a loss function by name.

    Valid loss functions include:
        * mse
        * kl_div

    Args:
        loss_fn: The name of the loss function to return.

    Returns:
        The loss callable.
    """
    if loss_fn == "mse":
        return softmax_mse_loss
    elif loss_fn == "kl_div":
        return kl_div_loss
    else:
        raise ValueError(
            loss_fn,
            "is not a supported consistency loss function. "
            "Choose between `mse` or `kl_div`. Default `mse`.",
        )