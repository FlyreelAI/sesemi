#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Loss heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Any, Optional
from pytorch_lightning.loggers.base import LightningLoggerBase

from ..backbones.base import Backbone
from ..heads.base import Head


class LossHead(nn.Module):
    """The interface for loss heads.

    Attributes:
        logger: An optional PyTorch Lightning logger.
    """

    def __init__(
        self,
        logger: Optional[LightningLoggerBase] = None,
    ):
        """Initializes the loss head.

        Args:
            logger: An optional PyTorch Lightning logger.
        """
        super().__init__()
        self.logger = logger

    def build(self, backbones: Dict[str, Backbone], heads: Dict[str, Head], **kwargs):
        """Builds the loss head.

        Args:
            backbones: A dictionary of shared backbones. During the build, new backbones can be
                added. As this is actually an `nn.ModuleDict` which tracks the parameters,
                these new backbones should not be saved to the loss head object to avoid
                double-tracking.
            heads: A dictionary of shared heads similar to the backbones
        """

    def forward(
        self,
        data: Dict[str, Any],
        backbones: Dict[str, Backbone],
        heads: Dict[str, Head],
        features: Dict[str, Any],
        step: int,
        **kwargs,
    ) -> Tensor:
        """Computes the loss.

        Args:
            data: A dictionary of data batches.
            backbones: A dictionary of shared backbones. This should not be altered.
            heads: A dictionary of shared heads. This should not be altered.
            features: A dictionary of shared features. Additional tensors can be added to this.
            step: The training step number.
            **kwargs: Placeholder for other arguments that may be added.
        """
        raise NotImplementedError()


class RotationPredictionLossHead(LossHead):
    """The rotation prediction loss head.
    https://arxiv.org/abs/1803.07728
    """

    def __init__(
        self,
        input_data: str,
        input_backbone: str = "backbone",
        num_pretext_classes: int = 4,
        logger: Optional[LightningLoggerBase] = None,
    ):
        """Initializes the loss head.

        Args:
            input_data: The key used to get the rotation prediction input data.
            input_backbone: The key used to get the backbone for feature extraction.
            num_pretext_classes: Number of pretext labels.
            logger: An optional PyTorch Lightning logger.
        """
        super().__init__(logger)
        self.input_data = input_data
        self.input_backbone = input_backbone
        self.num_pretext_classes = num_pretext_classes

    def build(self, backbones: Dict[str, Backbone], heads: Dict[str, Head], **kwargs):
        self.fc_unsupervised = nn.Linear(
            backbones[self.input_backbone].out_features, self.num_pretext_classes
        )

    def forward(
        self,
        data: Dict[str, Any],
        backbones: Dict[str, Backbone],
        heads: Dict[str, Head],
        features: Dict[str, Any],
        step: int,
        **kwargs,
    ) -> Tensor:
        inputs_u, targets_u = data[self.input_data]
        x_u = backbones[self.input_backbone](inputs_u)
        output_u = self.fc_unsupervised(x_u)
        loss_u = F.cross_entropy(output_u, targets_u, reduction="none")
        return loss_u


class JigsawPredictionLossHead(RotationPredictionLossHead):
    """The jigsaw prediction loss head.
    Idea and implementation are adapted from
    https://arxiv.org/abs/1903.06864
    """

    def __init__(
        self,
        input_data: str,
        input_backbone: str = "backbone",
        num_pretext_classes: int = 6,
        logger: Optional[LightningLoggerBase] = None,
    ):
        super().__init__(
            input_data,
            input_backbone,
            num_pretext_classes,
        )


class EntropyMinimizationLossHead(LossHead):
    """The entropy minimization loss head.
    https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf
    """

    def __init__(
        self,
        input_data: str,
        input_backbone: str = "backbone",
        predict_fn: str = "supervised",
        logger: Optional[LightningLoggerBase] = None,
    ):
        """Initializes the loss head.

        Args:
            input_data: The key used to get the unlabeled input data.
            input_backbone: The key used to get the backbone for feature extraction.
            predict_fn: The prediction function used to compute loss statistics.
            logger: An optional PyTorch Lightning logger.
        """
        super().__init__(logger)
        self.input_data = input_data
        self.input_backbone = input_backbone
        self.predict_fn = predict_fn

    def forward(
        self,
        data: Dict[str, Any],
        backbones: Dict[str, Backbone],
        heads: Dict[str, Head],
        features: Dict[str, Any],
        step: int,
        **kwargs,
    ) -> Tensor:
        inputs_u, _ = data[self.input_data]
        x_u = backbones[self.input_backbone](inputs_u)
        output_u = heads[self.predict_fn](x_u)
        loss_u = (-F.softmax(output_u, dim=-1) * F.log_softmax(output_u, dim=-1)).sum(1)
        return loss_u
