#
# Copyright 2021, Flyreel. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================#
"""Loss heads."""
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
    """The rotation prediction loss head."""

    def __init__(
        self,
        input_data: str = "rotation_prediction",
        input_backbone: str = "backbone",
        logger: Optional[LightningLoggerBase] = None,
    ):
        """Initializes the loss head.

        Args:
            input_data: The key used to get the rotation prediction input data.
            input_backbone: The key used to get the backbone for feature extraction.
            logger: An optional PyTorch Lightning logger.
        """
        super().__init__(logger)
        self.input_data = input_data
        self.input_backbone = input_backbone

    def build(self, backbones: Dict[str, Backbone], heads: Dict[str, Head], **kwargs):
        self.fc_unlabeled = nn.Linear(backbones[self.input_backbone].out_features, 4)

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
        x_unlabeled = backbones[self.input_backbone](inputs_u)
        output_unlabeled = self.fc_unlabeled(x_unlabeled)
        loss_u = F.cross_entropy(output_unlabeled, targets_u, reduction="none")
        return loss_u
