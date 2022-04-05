from torch import nn
from torch import Tensor

from sesemi.utils import freeze_module

from .base import Head


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