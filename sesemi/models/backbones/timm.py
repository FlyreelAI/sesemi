# Copyright 2021, Flyreel. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import torch
import torch.nn as nn

from .base import Backbone

PYTORCH_IMAGE_MODELS_REPO = "rwightman/pytorch-image-models"


SUPPORTED_BACKBONES = (
    # The following backbones strike a balance between accuracy and model size, with optional
    # pretrained ImageNet weights. For a summary of their ImageNet performance, see
    # <https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv>.
    # Compared with the defaults, the "d" variants (e.g., resnet50d, resnest50d)
    # replace the 7x7 conv in the input stem with three 3x3 convs.
    # And in the downsampling block, a 2x2 avg_pool with stride 2 is added before conv,
    # whose stride is changed to 1. Described in `Bag of Tricks <https://arxiv.org/abs/1812.01187>`.
    # ResNet models.
    "resnet18",
    "resnet18d",
    "resnet34",
    "resnet34d",
    "resnet50",
    "resnet50d",
    "resnet101d",
    "resnet152d",
    # ResNeXt models.
    "resnext50_32x4d",
    "resnext50d_32x4d",
    "resnext101_32x8d",
    # Squeeze and Excite models.
    "seresnet50",
    "seresnet152d",
    "seresnext26d_32x4d",
    "seresnext26t_32x4d",
    "seresnext50_32x4d",
    # ResNeSt models.
    "resnest14d",
    "resnest26d",
    "resnest50d",
    "resnest101e",
    "resnest200e",
    "resnest269e",
    "resnest50d_1s4x24d",
    "resnest50d_4s2x40d",
    # ResNet-RS models.
    "resnetrs50",
    "resnetrs101",
    "resnetrs152",
    "resnetrs200",
    # DenseNet models.
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    # Inception models.
    "inception_v3",
    "inception_v4",
    "inception_resnet_v2",
    # Xception models.
    "xception",
    "xception41",
    "xception65",
    "xception71",
    # EfficientNet models.
    "tf_efficientnet_b0",
    "tf_efficientnet_b1",
    "tf_efficientnet_b2",
    "tf_efficientnet_b3",
    "tf_efficientnet_b4",
    "tf_efficientnet_b5",
    "tf_efficientnet_b6",
    "tf_efficientnet_b7",
    # EfficientNet models trained with noisy student.
    "tf_efficientnet_b0_ns",
    "tf_efficientnet_b1_ns",
    "tf_efficientnet_b2_ns",
    "tf_efficientnet_b3_ns",
    "tf_efficientnet_b4_ns",
    "tf_efficientnet_b5_ns",
    "tf_efficientnet_b6_ns",
    "tf_efficientnet_b7_ns",
)


class PyTorchImageModels(Backbone):
    def __init__(
        self,
        name: str = "resnet50d",
        pretrained: bool = True,
        global_pool: str = "avg",
        dropout_rate: float = 0.0,
        freeze: bool = False,
    ):
        assert (
            name in SUPPORTED_BACKBONES
        ), f"backbone {name} must be one of {SUPPORTED_BACKBONES}"
        super().__init__()
        try:
            self.encoder = torch.hub.load(
                PYTORCH_IMAGE_MODELS_REPO,
                name,
                pretrained,
                num_classes=0,
                global_pool=global_pool,
            )
        except RuntimeError:
            self.encoder = torch.hub.load(
                PYTORCH_IMAGE_MODELS_REPO,
                name,
                pretrained,
                num_classes=0,
                global_pool=global_pool,
                force_reload=True,
            )

        self.dropout = nn.Dropout(dropout_rate)
        self.out_features = self.encoder.num_features
        if global_pool == "catavgmax":
            self.out_features *= 2

        if freeze:
            self.freeze()

    def forward(self, x):
        return self.dropout(self.encoder(x))
