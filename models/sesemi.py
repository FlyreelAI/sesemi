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

import logging


class SESEMI(nn.Module):
    def __init__(self, backbone, pretrained, labeled_classes, unlabeled_classes):
        super(SESEMI, self).__init__()
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

