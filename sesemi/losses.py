#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Loss functions and modules."""
import torch.nn.functional as F

from torch import Tensor


def softmax_mse_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes the mean squared error loss after softmax."""

    return F.mse_loss(F.softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction="none")


def kl_div_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes the Kullback–Leibler divergence loss between two probability distributions."""

    return F.kl_div(F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction="none")

