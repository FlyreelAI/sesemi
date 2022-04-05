#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Interface definition for backbones."""
import torch.nn as nn

from torch import Tensor

from sesemi.utils import freeze_module


class Head(nn.Module):
    """The interface for image classification heads.

    Attributes:
        in_features: The number of inputs features to the head.
        out_features: The number of output features from the head.
    """

    in_features: int
    out_features: int


class LinearHead(Head):
    def __init__(self, in_features: int, out_features: int, freeze: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features)
        if freeze:
            freeze_module(self)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.lin(inputs)
