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


PYTORCH_IMAGE_MODELS_REPO = 'rwightman/pytorch-image-models'


class EfficientNet(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0_ns', pretrained=True):
        super(EfficientNet, self).__init__()
        self.encoder = torch.hub.load(
            PYTORCH_IMAGE_MODELS_REPO,
            backbone, 
            pretrained=pretrained,
            num_classes=0,
            global_pool='')        
        self.final_gelu = nn.GELU()
        self.in_features = self.encoder.conv_head.out_channels

    def forward(self, x):
        x = self.encoder(x)
        return self.final_gelu(x)
