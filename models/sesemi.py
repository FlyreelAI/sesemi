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

from .resnet import Resnet
from .inception import Inception3
from .densenet import Densenet
from .efficientnet import EfficientNet

import logging


SUPPORTED_BACKBONES = (
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
    'inception_v3', 'densenet121', 'densenet169', 'densenet201', 'densenet161',

    # Standard EfficientNet models.
    'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2',
    'tf_efficientnet_b3', 'tf_efficientnet_b4', 'tf_efficientnet_b5',
    'tf_efficientnet_b6', 'tf_efficientnet_b7',

    # EfficientNet models pretrained using the noisy student algorithm.
    'tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2_ns',
    'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns',
    'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns',
)

class SESEMI(nn.Module):
    def __init__(self, backbone, pretrained, labeled_classes, unlabeled_classes):
        super(SESEMI, self).__init__()
        assert backbone in SUPPORTED_BACKBONES, f'--backbone must be one of {SUPPORTED_BACKBONES}'
        self.backbone = backbone
        self.pretrained = pretrained
        self.labeled_classes = labeled_classes
        self.unlabeled_classes = unlabeled_classes

        if 'resn' in backbone:
            self.feature_extractor = Resnet(
                backbone=backbone, pretrained=pretrained)
        elif 'inception' in backbone:
            self.feature_extractor = Inception3(
                backbone=backbone, pretrained=pretrained)
        elif 'densenet' in backbone:
            self.feature_extractor = Densenet(
                backbone=backbone, pretrained=pretrained)
        elif 'efficientnet' in backbone:
            self.feature_extractor = EfficientNet(
                backbone=backbone, pretrained=pretrained)
        else:
            raise NotImplementedError()

        if pretrained:
            logging.info(f'Initialized with pretrained {backbone} backbone')
        in_features = self.feature_extractor.in_features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc_labeled = nn.Linear(in_features, labeled_classes)
        self.fc_unlabeled = nn.Linear(in_features, unlabeled_classes)
    
    def forward(self, x_labeled, x_unlabeled=None):
        # Compute output for labeled input
        x_labeled = self.feature_extractor(x_labeled)
        x_labeled = self.avgpool(x_labeled)
        x_labeled = torch.flatten(x_labeled, start_dim=1)
        x_labeled = self.dropout(x_labeled)
        output_labeled = self.fc_labeled(x_labeled)
        
        if x_unlabeled is not None:
            # Compute output for unlabeled input and return both outputs
            x_unlabeled = self.feature_extractor(x_unlabeled)
            x_unlabeled = self.avgpool(x_unlabeled)
            x_unlabeled = torch.flatten(x_unlabeled, start_dim=1)
            output_unlabeled = self.fc_unlabeled(x_unlabeled)
            return output_labeled, output_unlabeled
        
        # Return predictions for only labeled input
        return output_labeled

