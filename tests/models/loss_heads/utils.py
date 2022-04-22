import torch

from copy import deepcopy
from typing import List, Optional

from sesemi.models.backbones import WRN
from sesemi.models.heads import LinearHead


def initialize_loss_head_mock_inputs(
    backbones: Optional[List[str]] = None,
    heads: Optional[List[str]] = None,
    num_classes: int = 5,
):
    backbone = WRN(10, 1)
    head = LinearHead(backbone.out_features, num_classes)

    backbones_set = set(backbones or [])
    backbones_set.update(["supervised_backbone", "supervised_backbone_ema"])

    heads_set = set(heads or [])
    heads_set.update(["supervised_head", "supervised_head_ema"])

    backbones = sorted(list(backbones_set))
    heads = sorted(list(heads_set))

    data = {}

    backbones_ = {name: deepcopy(backbone) for name in backbones}

    heads_ = {name: deepcopy(head) for name in heads}

    features = {}

    return dict(data=data, backbones=backbones_, heads=heads_, features=features)
