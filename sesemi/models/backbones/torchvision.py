#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Backbones for the torchvision repository."""
import logging
import torch.nn as nn

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models

# from .base import Backbone

logger = logging.getLogger(__name__)


class TorchVisionBackbone(nn.Module):  # (Backbone):
    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        drop_rate: float = 0.0,
        freeze: bool = False,
        **kwargs,
    ):
        """Builds the pytorch-image-models backbone.

        Args:
            name: The name of the backbone.
            pretrained: Whether to load the default pretrained model weights.
            global_pool: The kind of pooling to use. Can be one of (avg, max, avgmax, catavgmax).
            drop_rate: The dropout rate.
            freeze: Whether to freeze the backbone's weights.
        """
        if name not in models.__dict__:
            logger.warn(f"backbone {name} supported by torchvision")

        super().__init__()
        self.encoder = models.__dict__[name](pretrained=pretrained)
        last_layer = list(self.encoder.modules())[-1]
        assert isinstance(last_layer, nn.Linear)

        self.encoder = create_feature_extractor(self.encoder, return_nodes=["flatten"])

        self.out_features = last_layer.in_features
        self.dropout = nn.Dropout(p=drop_rate)

        if freeze:
            self.freeze()

    def forward(self, x):
        return self.dropout(self.encoder(x)["flatten"])
