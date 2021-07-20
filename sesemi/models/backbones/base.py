"""Interface definition for backbones."""
from abc import ABC

import torch.nn as nn


class Backbone(nn.Module):
    """The interface for image classification backbones."""

    out_features: int

    def freeze(self):
        for m in self.modules():
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
