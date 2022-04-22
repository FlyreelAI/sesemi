#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Backbone models."""
from . import base, resnet, timm, torchvision
from .base import Backbone
from .resnet import CIFARResNet, WRN
from .timm import PyTorchImageModel
from .torchvision import TorchVisionBackbone
