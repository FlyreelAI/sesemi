#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Loss heads."""
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Any, Optional

from sesemi.logger import LoggerWrapper

from ..backbones.base import Backbone
from ..heads.base import Head
from .base import LossHead


class RotationPredictionLossHead(LossHead):
    """The rotation prediction loss head.
    https://arxiv.org/abs/1803.07728
    """

    def __init__(
        self,
        data: str,
        backbone: str = "supervised_backbone",
        num_pretext_classes: int = 4,
    ):
        """Initializes the loss head.

        Args:
            input_data: The key used to get the rotation prediction input data.
            input_backbone: The key used to get the backbone for feature extraction.
            num_pretext_classes: Number of pretext labels.
        """
        super().__init__()
        self.data = data
        self.backbone = backbone
        self.num_pretext_classes = num_pretext_classes

    def build(self, backbones: Dict[str, Backbone], heads: Dict[str, Head], **kwargs):
        self.fc_unsupervised = nn.Linear(
            backbones[self.backbone].out_features, self.num_pretext_classes
        )

    def forward(
        self,
        data: Dict[str, Any],
        backbones: Dict[str, Backbone],
        heads: Dict[str, Head],
        features: Dict[str, Any],
        step: int,
        logger_wrapper: Optional[LoggerWrapper] = None,
        **kwargs,
    ) -> Tensor:
        inputs, targets = data[self.data]
        feats = backbones[self.backbone](inputs)
        logits = self.fc_unsupervised(feats)
        loss_u = F.cross_entropy(logits, targets, reduction="none")

        if logger_wrapper:
            logger_wrapper.log_images("rotation_prediction/images", inputs, step=step)

        return loss_u
