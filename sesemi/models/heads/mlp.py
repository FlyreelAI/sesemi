"""Multi-layer perceptron heads."""
import torch

from torch import nn

from typing import List
from math import ceil


class SimCLRProjectionHead(nn.Module):
    """A SimCLR-based projection head."""

    def __init__(
        self,
        features_dim: int,
        projection_dim: int = 128,
        width_multiplier: float = 1.0,
        num_layers: int = 2,
        use_final_bn: bool = True,
        use_final_relu: bool = True,
    ):
        """Initializes the projection head.

        Args:
            features_dim: The dimensions of the input features.
            projection: The dimension of the projected features.
            width_multiplier: The multiplier to compute for the intermediate features.
            num_layers: How many total layers to use in this projection head.
            use_final_bn: Whether to use a batch norm layer at the end of the head.
            use_final_relu: Whether to apply a relu activation to the final outputs.
        """
        super().__init__()
        self.features_dim = features_dim
        self.projection_dim = projection_dim
        self.width_multiplier = width_multiplier
        self.num_layers = num_layers
        self.use_final_bn = use_final_bn
        self.use_final_relu = use_final_relu

        layers: List[nn.Module] = []
        if num_layers == 1:
            layers.append(
                nn.Linear(
                    self.features_dim, self.projection_dim, bias=not self.use_final_bn
                )
            )
            if self.use_final_bn:
                layers.append(nn.BatchNorm1d(self.projection_dim))
            if self.use_final_relu:
                layers.append(nn.ReLU(inplace=True))
        else:
            hidden_dim = ceil(self.features_dim * width_multiplier)

            layers.append(nn.Linear(self.features_dim, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))

            layers.append(
                nn.Linear(hidden_dim, self.projection_dim, bias=not self.use_final_bn)
            )
            if self.use_final_bn:
                layers.append(nn.BatchNorm1d(self.projection_dim))
            if self.use_final_relu:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) > 2:
            y = x.reshape(torch.tensor(torch.prod(x.shape[:-1])), x.shape[-1])
            o = self.layers(y)
            o = o.reshape(*x.shape[:-1], o.shape[-1])
            return o
        return self.layers(x)


class BYOLProjectionHead(SimCLRProjectionHead):
    def __init__(
        self,
        features_dim: int,
        projection_dim: int = 256,
        width_multiplier: float = 1.0,
        num_layers: int = 2,
        use_final_bn: bool = False,
        use_final_relu: bool = True,
    ):
        super().__init__(
            features_dim=features_dim,
            projection_dim=projection_dim,
            width_multiplier=width_multiplier,
            num_layers=num_layers,
            use_final_bn=use_final_bn,
            use_final_relu=use_final_relu,
        )


class BYOLPredictionHead(BYOLProjectionHead):
    ...


class SimSiamProjectionHead(SimCLRProjectionHead):
    def __init__(
        self,
        features_dim: int,
        projection_dim: int = 2048,
        width_multiplier: float = 0.25,
        num_layers: int = 3,
        use_final_bn: bool = True,
        use_final_relu: bool = False,
    ):
        super().__init__(
            features_dim=features_dim,
            projection_dim=projection_dim,
            width_multiplier=width_multiplier,
            num_layers=num_layers,
            use_final_bn=use_final_bn,
            use_final_relu=use_final_relu,
        )


class SimSiamPredictionHead(SimCLRProjectionHead):
    def __init__(
        self,
        features_dim: int,
        projection_dim: int = 2048,
        width_multiplier: float = 0.25,
        num_layers: int = 2,
        use_final_bn: bool = False,
        use_final_relu: bool = False,
    ):
        super().__init__(
            features_dim=features_dim,
            projection_dim=projection_dim,
            width_multiplier=width_multiplier,
            num_layers=num_layers,
            use_final_bn=use_final_bn,
            use_final_relu=use_final_relu,
        )
