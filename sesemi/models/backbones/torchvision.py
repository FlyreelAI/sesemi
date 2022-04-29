#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Backbones for the torchvision repository."""
import logging
from typing import Optional
from torch import Tensor
import torch.nn as nn

from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from sesemi.models.backbones.base import Backbone
from sesemi.utils import freeze_module

from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import models

from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

SUPPORTED_MODELS = ()


class TorchVisionBackbone(Backbone):
    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        global_pool: Optional[str] = None,
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
            raise ValueError(f"backbone {name} supported by torchvision")

        if name not in SUPPORTED_MODELS:
            rank_zero_warn(f"backbone {name} is untested")

        super().__init__()

        encoder = models.__dict__[name](pretrained=pretrained)

        feature_vector_node_name = None
        feature_map_node_name = None

        feature_names, _ = get_graph_node_names(encoder)

        if name == "alexnet":
            feature_vector_node_name = "classifier.4"
        elif "vgg" in name:
            feature_vector_node_name = "classifier.4"
        elif "squeeze" in name:
            feature_map_node_name = [
                x
                for x in feature_names
                if x.startswith("features.") and x.endswith(".cat")
            ][-1]
        elif "resnet" in name or "resnext" in name or "regnet" in name:
            avgpool_index = feature_names.index("avgpool")
            feature_map_node_name = feature_names[avgpool_index - 1]
        elif "densenet" in name:
            feature_map_node_name = "relu"
        elif "inception" in name:
            feature_map_node_name = [x for x in feature_names if x.endswith(".cat_2")][
                -1
            ]
        elif name == "googlenet":
            feature_map_node_name = "inception5b.cat"
        elif "shufflenet" in name:
            feature_map_node_name = [x for x in feature_names if x.startswith("conv")][
                -1
            ]
        elif "mobilenet_v2" in name:
            feature_map_node_name = [
                x for x in feature_names if x.startswith("features.")
            ][-1]
        elif "mobilenet_v3" in name:
            feature_vector_node_name = "classifier.1"
        elif "mnasnet" in name:
            feature_map_node_name = [
                x for x in feature_names if x.startswith("layers.")
            ][-1]
        elif "efficientnet" in name:
            feature_map_node_name = [
                x for x in feature_names if x.startswith("features.")
            ][-1]
        elif "vit" in name:
            feature_vector_node_name = "getitem_5"
        elif "convnext" in name:
            feature_vector_node_name = "classifier.1"
        else:
            raise ValueError(f"unsupported backbone {name}")

        assert bool(feature_vector_node_name) ^ bool(
            feature_map_node_name
        ), "cannot specify both feature vector and feature map node names"

        self.feature_postprocessor = nn.Sequential()
        if feature_vector_node_name is not None:
            if global_pool:
                rank_zero_warn(f"feature map pooling not supported for backbone {name}")

            output_node_name = feature_vector_node_name
            self.feature_extractor = create_feature_extractor(
                encoder, return_nodes=[feature_vector_node_name]
            )
        else:
            assert feature_map_node_name is not None
            output_node_name = feature_map_node_name
            self.feature_extractor = create_feature_extractor(
                encoder, return_nodes=[feature_map_node_name]
            )

            self.feature_postprocessor.append(SelectAdaptivePool2d(1))
            self.feature_postprocessor.append(nn.Flatten(1))

        self.output_node_name = output_node_name
        self.out_features = self.get_out_features(name, encoder)

        self.dropout = nn.Dropout(p=drop_rate)

        if freeze:
            freeze_module(self)

    def get_out_features(self, name: str, encoder: nn.Module) -> int:
        """Computes the output feature size for the given encoder.

        Arg:
            name: The name of the encoder.
            encoder: The encoder module.

        Returns:
            The size of the extracted features.
        """
        if "squeezenet" in name:
            conv_layers = [x for x in encoder.modules() if isinstance(x, nn.Conv2d)]
            last_conv_layer = conv_layers[-1]
            return last_conv_layer.in_channels
        else:
            linear_layers = [x for x in encoder.modules() if isinstance(x, nn.Linear)]
            last_linear_layer = linear_layers[-1]
            return last_linear_layer.in_features

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(
            self.feature_postprocessor(self.feature_extractor(x)[self.output_node_name])
        )
