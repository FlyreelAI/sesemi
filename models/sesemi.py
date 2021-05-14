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
from .timm import PyTorchImageModels
import logging


SUPPORTED_BACKBONES = (
    # The following backbones strike a balance between accuracy and model size, with optional
    # pretrained ImageNet weights. For a summary of their ImageNet performance, see
    # <https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv>.

    # Compared with the defaults, the "d" variants (e.g., resnet50d, resnest50d)
    # replace the 7x7 conv in the input stem with three 3x3 convs.
    # And in the downsampling block, a 2x2 avg_pool with stride 2 is added before conv,
    # whose stride is changed to 1. Described in `Bag of Tricks <https://arxiv.org/abs/1812.01187>`.
    
    # ResNet models.
    'resnet18', 'resnet18d', 'resnet34', 'resnet34d', 'resnet50', 'resnet50d', 'resnet101d', 'resnet152d',
    
    # ResNeXt models.
    'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x8d',
    
    # Squeeze and Excite models.
    'seresnet50', 'seresnet152d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext50_32x4d',
    
    # ResNeSt models.
    'resnest14d', 'resnest26d', 'resnest50d', 'resnest101e', 'resnest200e', 'resnest269e',
    'resnest50d_1s4x24d', 'resnest50d_4s2x40d',
    
    # ResNet-RS models.
    'resnetrs50', 'resnetrs101', 'resnetrs152', 'resnetrs200',
    
    # DenseNet models.
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    
    # Inception models.
    'inception_v3', 'inception_v4', 'inception_resnet_v2',
    
    # Xception models.
    'xception', 'xception41', 'xception65', 'xception71',
    
    # EfficientNet models.
    'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2',
    'tf_efficientnet_b3', 'tf_efficientnet_b4', 'tf_efficientnet_b5',
    'tf_efficientnet_b6', 'tf_efficientnet_b7',

    # EfficientNet models trained with noisy student.
    'tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2_ns',
    'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns',
    'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns',
)


class SESEMI(nn.Module):
    def __init__(self, backbone, pretrained,
                 num_labeled_classes, num_unlabeled_classes,
                 dropout_rate=0.5, global_pool='avg'):
        super(SESEMI, self).__init__()
        assert backbone in SUPPORTED_BACKBONES, f'--backbone must be one of {SUPPORTED_BACKBONES}'
        self.backbone = backbone
        self.pretrained = pretrained
        self.num_labeled_classes = num_labeled_classes
        self.num_unlabeled_classes = num_unlabeled_classes
        self.dropout_rate = dropout_rate
        self.global_pool = global_pool
        if not global_pool:
            # If no global pooling method specified, fall back to "avg"
            self.global_pool = 'avg'

        self.feature_extractor = PyTorchImageModels(backbone, pretrained, self.global_pool)

        if self.pretrained:
            logging.info(f'Initialized with pretrained {backbone} backbone')
        self.in_features = self.feature_extractor.in_features
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_labeled = nn.Linear(self.in_features, self.num_labeled_classes)
        self.fc_unlabeled = nn.Linear(self.in_features, self.num_unlabeled_classes)
    
    def forward(self, x_labeled, x_unlabeled=None):
        # Compute output for labeled input
        x_labeled = self.feature_extractor(x_labeled)
        if self.dropout_rate > 0.0:
            x_labeled = self.dropout(x_labeled)
        output_labeled = self.fc_labeled(x_labeled)
        
        if x_unlabeled is not None:
            # Compute output for unlabeled input and return both outputs
            x_unlabeled = self.feature_extractor(x_unlabeled)
            output_unlabeled = self.fc_unlabeled(x_unlabeled)
            return output_labeled, output_unlabeled

        return output_labeled

