#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Any, Optional

from sesemi.logger import LoggerWrapper

from ..backbones.base import Backbone
from ..heads.base import Head
from .base import LossHead, LossOutputs


class EntropyMinimizationLossHead(LossHead):
    """The entropy minimization loss head.
    https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf
    """

    def __init__(
        self,
        data: str,
        backbone: str = "supervised_backbone",
        head: str = "supervised_head",
    ):
        """Initializes the loss head.

        Args:
            data: The key used to get the unlabeled input data.
            backbone: The key used to get the backbone for feature extraction.
            head: The prediction module used to compute output logits.
        """
        super().__init__()
        self.data = data
        self.backbone = backbone
        self.head = head

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
        inputs, _ = data[self.data]
        feats = backbones[self.backbone](inputs)
        logits = heads[self.head](feats)
        loss_u = (-F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(
            dim=-1
        )

        if logger_wrapper:
            logger_wrapper.log_images("entropy_minimization/images", inputs, step=step)

        return LossOutputs(losses=loss_u)
