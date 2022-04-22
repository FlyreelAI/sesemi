import torch
import torch.nn as nn

import pytest
import numpy as np

from collections import OrderedDict
from PIL import Image

from sesemi.tta import apply_model_to_test_time_augmentations, ListApply, MultiCropTTA


@pytest.mark.parametrize("batch_compatible_tensors", [True, False])
def test_apply_model_to_test_time_augmentations(batch_compatible_tensors, mocker):
    module = nn.Sequential(
        OrderedDict(
            {
                "linear": nn.Linear(10, 5),
                "gn": nn.GroupNorm(1, 5),
            }
        )
    )

    outputs = apply_model_to_test_time_augmentations(
        module,
        "cpu",
        [[torch.ones((10,)), torch.ones((10,))]],
        batch_compatible_tensors=batch_compatible_tensors,
    )

    assert len(outputs) == 1
    assert outputs[0].shape == (2, 5)

    outputs = apply_model_to_test_time_augmentations(
        module,
        "cpu",
        [[torch.ones((10,)), torch.ones((10,))], [torch.ones((10,))]],
        batch_compatible_tensors=batch_compatible_tensors,
    )

    assert len(outputs) == 2
    assert outputs[0].shape == (2, 5)
    assert outputs[1].shape == (1, 5)


def test_list_apply():
    a = ListApply(lambda x: 2 * x)
    assert a([1, 2]) == [2, 4]
    assert a([]) == []


@pytest.mark.parametrize("resize", [64])
@pytest.mark.parametrize("crop_dim", [48])
@pytest.mark.parametrize("num_crops", [5, 10])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bicubic"])
def test_multi_crop_tta(resize, crop_dim, num_crops, interpolation):
    t = MultiCropTTA(
        resize=resize,
        crop_dim=crop_dim,
        num_crops=num_crops,
        interpolation=interpolation,
    )

    image = Image.fromarray(np.ones((80, 80, 3), dtype=np.uint8))
    outputs = t(image)
    assert len(outputs) == num_crops
    for c in outputs:
        assert c.shape == (3, crop_dim, crop_dim)
