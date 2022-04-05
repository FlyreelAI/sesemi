#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from .rotation_prediction import RotationPredictionLossHead


class JigsawPredictionLossHead(RotationPredictionLossHead):
    """The jigsaw prediction loss head. Idea and implementation are adapted from
    https://arxiv.org/abs/1903.06864
    """

    def __init__(
        self,
        data: str,
        backbone: str = "supervised_backbone",
        num_pretext_classes: int = 6,
    ):
        super(JigsawPredictionLossHead, self).__init__(
            data,
            backbone=backbone,
            num_pretext_classes=num_pretext_classes,
        )