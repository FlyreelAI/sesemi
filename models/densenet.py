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
import torchvision.models as Models


class Densenet(nn.Module):
    def __init__(self, backbone='densenet169', pretrained=True):
        super(Densenet, self).__init__()
        self.encoder = getattr(Models, backbone)(pretrained=pretrained)
        self.final_relu = nn.ReLU(inplace=True)
        self.in_features = self.encoder.classifier.in_features

    def forward(self, x):
        x = self.encoder.features(x)
        return self.final_relu(x)

    def get_feature_blocks(self):
        blocks = [m for m in self.encoder.features.children()]
        block0 = blocks[:4]
        block1 = blocks[4:6]
        block2 = blocks[6:8]
        block3 = blocks[8:10]
        block4 = blocks[10:]
        return block0, block1, block2, block3, block4

