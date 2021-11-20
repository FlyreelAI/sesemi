#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Loss functions and modules."""
import torch.nn.functional as F

from torch import Tensor


def softmax_mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """Computes the mean squared error loss after softmax."""

    loss = F.mse_loss(
        F.softmax(input, dim=-1), F.softmax(target, dim=-1), reduction="none"
    )
    return loss


def softmax_kl_loss(input: Tensor, target: Tensor) -> Tensor:
    """Computes the Kullback–Leibler divergence loss between two probability distributions."""

    loss = F.kl_div(
        F.log_softmax(input, dim=-1), F.softmax(target, dim=-1), reduction="none"
    )
    return loss
