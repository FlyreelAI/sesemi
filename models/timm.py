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

class PyTorchImageModels(nn.Module):
    def __init__(self, backbone='resnet50d', pretrained=True):
        super(PyTorchImageModels, self).__init__()
        self.encoder = torch.hub.load(
            PYTORCH_IMAGE_MODELS_REPO,
            backbone, 
            pretrained,
        )
        for m in self.encoder.modules():
            if isinstance(m, (nn.Linear)):
                self.in_features = m.in_features

    def forward(self, x):
        return self.encoder.forward_features(x)
