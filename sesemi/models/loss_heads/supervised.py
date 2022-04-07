#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Callable, Dict, Any, List, Optional

from sesemi.logger import LoggerWrapper

from ..backbones.base import Backbone
from ..heads.base import Head
from .base import LossHead, LossOutputs


class SupervisedLossHead(LossHead):
    """A supervised loss head."""

    def __init__(
        self,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        data: str = "supervised",
        logits: str = "supervised_head",
        class_weights: Optional[List[float]] = None,
    ):
        """Initializes the loss head.

        Args:
            loss_fn: The per-sample loss function.
            data: The supervised data key.
            logits: The supervised logits feature key.
            class_weights: Optional weights to apply per class label.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.data = data
        self.logits = logits
        self.class_weights = class_weights

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
        _, targets = data[self.data][:2]

        losses = self.loss_fn(features[self.logits], targets)

        weights: Optional[Tensor] = None
        if self.class_weights is not None:
            class_weights = torch.tensor(self.class_weights, dtype=losses.dtype, device=losses.device)
            weights = torch.gather(class_weights, dim=0, index=targets.to(torch.int64))

        assert losses.shape == targets.shape, "loss shape must match targets shape"

        return LossOutputs(losses=losses, weights=weights)