import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Any, Optional
from pytorch_lightning.loggers.base import LightningLoggerBase

from ..backbones.base import Backbone


class LossHead(nn.Module):
    def __init__(
        self,
        logger: Optional[LightningLoggerBase] = None,
    ):
        super().__init__()
        self.logger = logger

    def build(self, backbones: Dict[str, Backbone]):
        ...

    def forward(
        self,
        *,
        data: Dict[str, Any],
        backbones: Dict[str, Backbone],
        features: Dict[str, Any],
        global_step: int,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError()


class RotationPredictionLossHead(LossHead):
    def __init__(
        self,
        input_data: str = "rotation_prediction",
        input_backbone: str = "backbone",
        logger: Optional[LightningLoggerBase] = None,
    ):
        super().__init__(logger)
        self.input_data = input_data
        self.input_backbone = input_backbone

    def build(self, backbones: Dict[str, Backbone]):
        self.fc_unlabeled = nn.Linear(backbones[self.input_backbone].out_features, 4)

    def forward(
        self,
        *,
        data: Dict[str, Any],
        backbones: Dict[str, Backbone],
        features: Dict[str, Any],
        global_step: int,
        **kwargs,
    ) -> Tensor:
        inputs_u, targets_u = data[self.input_data]
        x_unlabeled = backbones[self.input_backbone](inputs_u)
        output_unlabeled = self.fc_unlabeled(x_unlabeled)
        loss_u = F.cross_entropy(output_unlabeled, targets_u, reduction="none")
        return loss_u
