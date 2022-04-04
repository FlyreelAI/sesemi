#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Loss heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Any, Optional

from ..backbones.base import Backbone
from .base import Head
from ...losses import (
    softmax_mse_loss,
    kl_div_loss,
)
from ...logger import LoggerWrapper


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
    ) -> Tensor:
        """Computes the loss.

        Args:
            data: A dictionary of data batches.
            backbones: A dictionary of shared backbones. This should not be altered.
            heads: A dictionary of shared heads. This should not be altered.
            features: A dictionary of shared features. Additional tensors can be added to this.
            logger_wrapper: An optional wrapper around the lightning logger.
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
        input_backbone: str = "supervised_backbone",
        num_pretext_classes: int = 4,
    ):
        """Initializes the loss head.

        Args:
            input_data: The key used to get the rotation prediction input data.
            input_backbone: The key used to get the backbone for feature extraction.
            num_pretext_classes: Number of pretext labels.
        """
        super().__init__()
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
        logger_wrapper: Optional[LoggerWrapper] = None,
        **kwargs,
    ) -> Tensor:
        inputs, targets = data[self.input_data]
        feats = backbones[self.input_backbone](inputs)
        logits = self.fc_unsupervised(feats)
        loss_u = F.cross_entropy(logits, targets, reduction="none")

        if logger_wrapper:
            logger_wrapper.log_images("rotation_prediction/images", inputs, step=step)

        return loss_u


class JigsawPredictionLossHead(RotationPredictionLossHead):
    """The jigsaw prediction loss head. Idea and implementation are adapted from
    https://arxiv.org/abs/1903.06864
    """

    def __init__(
        self,
        input_data: str,
        input_backbone: str = "supervised_backbone",
        num_pretext_classes: int = 6,
    ):
        super(JigsawPredictionLossHead, self).__init__(
            input_data, num_pretext_classes=num_pretext_classes
        )


class EntropyMinimizationLossHead(LossHead):
    """The entropy minimization loss head.
    https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf
    """

    def __init__(
        self,
        input_data: str,
        input_backbone: str = "supervised_backbone",
        predict_head: str = "supervised_head",
    ):
        """Initializes the loss head.

        Args:
            input_data: The key used to get the unlabeled input data.
            input_backbone: The key used to get the backbone for feature extraction.
            predict_head: The prediction module used to compute output logits.
        """
        super().__init__()
        self.input_data = input_data
        self.input_backbone = input_backbone
        self.predict_head = predict_head

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
        inputs, _ = data[self.input_data]
        feats = backbones[self.input_backbone](inputs)
        logits = heads[self.predict_head](feats)
        loss_u = -F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)

        if logger_wrapper:
            logger_wrapper.log_images("entropy_minimization/images", inputs, step=step)

        return loss_u


class ConsistencyLossHead(EntropyMinimizationLossHead):
    """The consistency loss head following the Pi Model.
    https://arxiv.org/abs/1610.02242
    """

    def __init__(
        self,
        input_data: str,
        input_backbone: str = "supervised_backbone",
        predict_head: str = "supervised_head",
        loss_fn: str = "mse",
    ):
        """
        Args:
            input_data: The key used to get the unlabeled input data,
                which in this case returns two views of the same image.
            input_backbone: The key used to get the backbone for feature extraction.
            predict_head: The prediction module used to compute output logits.
            loss_fn: The loss function to compute the consistency between two views.
        """
        super(ConsistencyLossHead, self).__init__(input_data)
        if loss_fn == "mse":
            self.loss_fn = softmax_mse_loss
        elif loss_fn == "kl_div":
            self.loss_fn = kl_div_loss
        else:
            raise ValueError(
                loss_fn,
                "is not a supported consistency loss function. "
                "Choose between `mse` or `kl_div`. Default `mse`.",
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
        (view1, view2), _ = data[self.input_data]
        feats1 = backbones[self.input_backbone](view1)
        feats2 = backbones[self.input_backbone](view2)
        logits1 = heads[self.predict_head](feats1)
        logits2 = heads[self.predict_head](feats2)
        loss_u = self.loss_fn(logits1, logits2)

        if logger_wrapper:
            logger_wrapper.log_images("consistency/images/view1", view1, step=step)
            logger_wrapper.log_images("consistency/images/view2", view2, step=step)

        return loss_u


class EMAConsistencyLossHead(ConsistencyLossHead):
    """The EMA consistency loss head following Mean Teacher.
    https://arxiv.org/abs/1703.01780
    """

    def __init__(
        self,
        input_data: str,
        student_backbone: str = "supervised_backbone",
        teacher_backbone: str = "supervised_backbone_ema",
        student_head: str = "supervised_head",
        teacher_head: str = "supervised_head_ema",
        loss_fn: str = "mse",
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
        super(EMAConsistencyLossHead, self).__init__(input_data, loss_fn=loss_fn)
        self.student_backbone = student_backbone
        self.teacher_backbone = teacher_backbone
        self.student_head = student_head
        self.teacher_head = teacher_head

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
        (view1, view2), _ = data[self.input_data]
        student_feats = backbones[self.student_backbone](view1)
        teacher_feats = backbones[self.teacher_backbone](view2)
        student_logits = heads[self.student_head](student_feats)
        teacher_logits = heads[self.teacher_head](teacher_feats)
        loss_u = self.loss_fn(student_logits, teacher_logits)

        if logger_wrapper:
            logger_wrapper.log_images("ema_consistency/images/view1", view1, step=step)
            logger_wrapper.log_images("ema_consistency/images/view2", view2, step=step)

        return loss_u


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

        loss_weight = (weakly_augmented_probs.max(dim=-1)[0] >= self.threshold).to(
            torch.float32
        )

        loss = (
            F.cross_entropy(
                strongly_augmented_logits,
                weakly_augmented_labels,
                reduction="none",
            )
            * loss_weight
        )

        total_loss_weight = torch.sum(loss_weight)
        total_loss = torch.sum(loss)

        loss = total_loss / (total_loss_weight + 1e-8)

        if logger_wrapper:
            logger_wrapper.log_images(
                "fixmatch/images/weak", weakly_augmented, step=step
            )
            logger_wrapper.log_images(
                "fixmatch/images/strong", strongly_augmented, step=step
            )

        return loss
