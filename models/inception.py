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


class Inception3(nn.Module):
    def __init__(
        self,
        backbone='inception_v3',
        pretrained=True,
        aux_logits=False
    ):
        super(Inception3, self).__init__()
        self.encoder = getattr(Models, backbone)(
            pretrained=pretrained, aux_logits=aux_logits)
        self.in_features = self.encoder.fc.in_features

    def forward(self, x):
        x = self.encoder.Conv2d_1a_3x3(x)
        x = self.encoder.Conv2d_2a_3x3(x)
        x = self.encoder.Conv2d_2b_3x3(x)
        x = self.encoder.maxpool1(x)
        x = self.encoder.Conv2d_3b_1x1(x)
        x = self.encoder.Conv2d_4a_3x3(x)
        x = self.encoder.maxpool2(x)
        
        x = self.encoder.Mixed_5b(x)
        x = self.encoder.Mixed_5c(x)
        x = self.encoder.Mixed_5d(x)
        x = self.encoder.Mixed_6a(x)
        x = self.encoder.Mixed_6b(x)
        x = self.encoder.Mixed_6c(x)
        x = self.encoder.Mixed_6d(x)
        x = self.encoder.Mixed_6e(x)
        x = self.encoder.Mixed_7a(x)
        x = self.encoder.Mixed_7b(x)
        x = self.encoder.Mixed_7c(x)
        return x

