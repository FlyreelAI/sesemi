#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Loss functions and modules."""
from typing import Callable
from torch import Tensor

import torch.nn.functional as F

from .registries import CallableRegistry

LossRegistry = CallableRegistry[Tensor]()


@LossRegistry
def softmax_mse_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes the mean squared error loss after softmax."""

    return F.mse_loss(
        F.softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction="none"
    ).mean(dim=-1)


@LossRegistry
def kl_div_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes the Kullback–Leibler divergence loss between two probability distributions."""

    return F.kl_div(
        F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction="none"
    ).sum(dim=-1)
