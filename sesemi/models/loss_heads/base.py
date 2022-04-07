#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import torch.nn as nn

from torch import Tensor
from typing import Dict, Any, Optional
from dataclasses import dataclass

from sesemi.logger import LoggerWrapper

from ..backbones.base import Backbone
from ..heads.base import Head


@dataclass
class LossOutputs:
    """The outputs of a loss head.

    Attributes:
        losses: The per-sample losses.
        weights: Optional per-sample weights.
    """

    losses: Tensor
    weights: Optional[Tensor] = None

    def asdict(self) -> Dict[str, Any]:
        """Converts this into a dictionary."""
        return dict(
            losses=self.losses,
            weights=self.weights,
        )


class LossHead(nn.Module):
    """The interface for loss heads."""

    def build(
        self,
        backbones: Dict[str, Backbone],
        heads: Dict[str, Head],
        **kwargs,
    ):
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
        logger_wrapper: Optional[LoggerWrapper] = None,
        **kwargs,
    ) -> LossOutputs:
        """Computes the loss.

        Args:
            data: A dictionary of data batches.
            backbones: A dictionary of shared backbones. This should not be altered.
            heads: A dictionary of shared heads. This should not be altered.
            features: A dictionary of shared features. Additional tensors can be added to this.
            logger_wrapper: An optional wrapper around the lightning logger.
            step: The training step number.
            **kwargs: Placeholder for other arguments that may be added.

        Returns:
            The losses and optional per-sample weights.
        """
        raise NotImplementedError()
