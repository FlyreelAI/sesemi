import pytest
import torch

from types import FunctionType
from torchvision import models
from sesemi.models.backbones.torchvision import TorchVisionBackbone

from ...utils import check_model_frozen_state


@pytest.mark.parametrize(
    "name",
    [k for k, x in models.__dict__.items() if isinstance(x, FunctionType)],
)
@pytest.mark.parametrize("pretrained", [False])
def test_torchvision_backbone(name, pretrained):
    model = TorchVisionBackbone(name=name, pretrained=pretrained)
    outputs = model(torch.ones((2, 3, 224, 224), dtype=torch.float32))
    assert outputs.shape == (2, model.out_features)


@pytest.mark.parametrize("freeze", [False, True])
def test_torchvision_backbone_freeze(freeze):
    check_model_frozen_state(
        TorchVisionBackbone(name="resnet18", pretrained=False, freeze=freeze), freeze
    )
