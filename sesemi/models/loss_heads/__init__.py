#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from .base import LossHead
from .consistency import ConsistencyLossHead, EMAConsistencyLossHead
from .entropy_minimization import EntropyMinimizationLossHead
from .fixmatch import FixMatchLossHead
from .jigsaw_prediction import JigsawPredictionLossHead
from .rotation_prediction import RotationPredictionLossHead
from .supervised import SupervisedLossHead