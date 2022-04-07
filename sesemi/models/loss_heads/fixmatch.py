#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Any, Optional

from sesemi.logger import LoggerWrapper

from ..backbones.base import Backbone
from ..heads.base import Head
from .base import LossHead, LossOutputs


class FixMatchLossHead(LossHead):
    """The FixMatch loss head.

    @article{Sohn2020FixMatchSS,
        title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
        author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin Dogus Cubuk and Alexey Kurakin and Han Zhang and Colin Raffel},
        journal={ArXiv},
        year={2020},
        volume={abs/2001.07685}
    }
    """

    def __init__(
        self,
        data: str,
        student_backbone: str = "supervised_backbone",
        teacher_backbone: Optional[str] = None,
        student_head: str = "supervised_head",
        teacher_head: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """Initializes the loss head.

        Args:
            data: The data key.
            student_backbone: The student's backbone key.
            teacher_backbone: The teacher's backbone key. Defaults to the student's.
            student_head: The student's head key.
            teacher_head: The teacher's head key. Defaults to the student's.
            threshold: The threshold used to filter low confidence predictions
                made by the teacher.
        """
        super().__init__()
        self.data = data
        self.student_backbone = student_backbone
        self.teacher_backbone = teacher_backbone or student_backbone
        self.student_head = student_head
        self.teacher_head = teacher_head or student_head
        self.threshold = threshold

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
        weakly_augmented, strongly_augmented = data[self.data]
        student_backbone = backbones[self.student_backbone]
        student_head = heads[self.student_head]
        teacher_backbone = backbones[self.teacher_backbone]
        teacher_head = heads[self.teacher_head]

        weakly_augmented_features = teacher_backbone(weakly_augmented)
        strongly_augmented_features = student_backbone(strongly_augmented)

        weakly_augmented_logits = teacher_head(weakly_augmented_features).detach()
        strongly_augmented_logits = student_head(strongly_augmented_features)

        weakly_augmented_probs = torch.softmax(weakly_augmented_logits, dim=-1)
        weakly_augmented_labels = torch.argmax(weakly_augmented_probs, dim=-1).to(
            torch.long
        )

        loss_weights = (weakly_augmented_probs.max(dim=-1)[0] >= self.threshold).to(
            torch.float32
        )

        losses = F.cross_entropy(
            strongly_augmented_logits,
            weakly_augmented_labels,
            reduction="none",
        )

        if logger_wrapper:
            logger_wrapper.log_images(
                "fixmatch/images/weak", weakly_augmented, step=step
            )
            logger_wrapper.log_images(
                "fixmatch/images/strong", strongly_augmented, step=step
            )

        return LossOutputs(losses=losses, weights=loss_weights)
