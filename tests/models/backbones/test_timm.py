import pytest
import torch

from sesemi.models.backbones.timm import PyTorchImageModel

from ...utils import check_model_frozen_state


@pytest.mark.parametrize("name", ["resnet18", "resnet18d"])
@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("global_pool", ["avg", "max", "avgmax", "catavgmax"])
def test_pytorch_image_model(name, pretrained, global_pool):
    model = PyTorchImageModel(name=name, pretrained=pretrained, global_pool=global_pool)
    outputs = model(torch.ones((2, 3, 32, 32), dtype=torch.float32))
    assert outputs.shape == (2, model.out_features)


@pytest.mark.parametrize("freeze", [False, True])
def test_pytorch_image_model_freeze(freeze):
    check_model_frozen_state(
        PyTorchImageModel(name="resnet18", pretrained=False, freeze=freeze), freeze
    )
