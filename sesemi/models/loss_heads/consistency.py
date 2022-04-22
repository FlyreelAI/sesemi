#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from torch import Tensor
from typing import Dict, Any, Optional

from sesemi.losses import LossRegistry
from sesemi.logger import LoggerWrapper

from ..backbones.base import Backbone
from ..heads.base import Head
from .base import LossHead, LossOutputs
from .entropy_minimization import EntropyMinimizationLossHead


class ConsistencyLossHead(EntropyMinimizationLossHead):
    """The consistency loss head following the Pi Model.
    https://arxiv.org/abs/1610.02242
    """

    def __init__(
        self,
        data: str,
        backbone: str = "supervised_backbone",
        head: str = "supervised_head",
        loss_fn: str = "softmax_mse_loss",
    ):
        """
        Args:
            input_data: The key used to get the unlabeled input data,
                which in this case returns two views of the same image.
            input_backbone: The key used to get the backbone for feature extraction.
            predict_head: The prediction module used to compute output logits.
            loss_fn: The loss function to compute the consistency between two views.
        """
        super().__init__(data=data, backbone=backbone, head=head)
        self.loss_fn = LossRegistry[loss_fn]

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
        (view1, view2), _ = data[self.data]
        feats1 = backbones[self.backbone](view1)
        feats2 = backbones[self.backbone](view2)
        logits1 = heads[self.head](feats1)
        logits2 = heads[self.head](feats2)
        loss_u = self.loss_fn(logits1, logits2)

        if logger_wrapper:
            logger_wrapper.log_images("consistency/images/view1", view1, step=step)
            logger_wrapper.log_images("consistency/images/view2", view2, step=step)

        return LossOutputs(losses=loss_u)


class EMAConsistencyLossHead(LossHead):
    """The EMA consistency loss head following Mean Teacher.
    https://arxiv.org/abs/1703.01780
    """

    def __init__(
        self,
        data: str,
        student_backbone: str = "supervised_backbone",
        teacher_backbone: str = "supervised_backbone_ema",
        student_head: str = "supervised_head",
        teacher_head: str = "supervised_head_ema",
        loss_fn: str = "softmax_mse_loss",
    ):
        """
        Args:
            input_data: The key used to get the unlabeled input data,
                which in this case returns two views of the same image.
            student_backbone: The student backbone for feature extraction.
            teacher_backbone: The teacher backbone for feature extraction.
            student_head: The student module used to compute output logits.
            teacher_head: The teacher module used to compute output logits.
            loss_fn: The loss function to compute the consistency between two views.
        """
        super().__init__()
        self.data = data
        self.student_backbone = student_backbone
        self.teacher_backbone = teacher_backbone
        self.student_head = student_head
        self.teacher_head = teacher_head
        self.loss_fn = LossRegistry[loss_fn]

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
        (view1, view2), _ = data[self.data]
        student_feats = backbones[self.student_backbone](view1)
        teacher_feats = backbones[self.teacher_backbone](view2)
        student_logits = heads[self.student_head](student_feats)
        teacher_logits = heads[self.teacher_head](teacher_feats)
        loss_u = self.loss_fn(student_logits, teacher_logits)

        if logger_wrapper:
            logger_wrapper.log_images("ema_consistency/images/view1", view1, step=step)
            logger_wrapper.log_images("ema_consistency/images/view2", view2, step=step)

        return LossOutputs(losses=loss_u)
