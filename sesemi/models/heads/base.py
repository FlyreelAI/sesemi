#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Interface definition for backbones."""
import torch.nn as nn


class Head(nn.Module):
    """The interface for image classification heads.

    Attributes:
        in_features: The number of inputs features to the head.
        out_features: The number of output features from the head.
    """

    in_features: int
    out_features: int
