import torch
import torch.nn as nn

import pytest
import numpy as np
import albumentations.augmentations as A

from collections import OrderedDict
from PIL import Image

from sesemi.transforms import (
    AlbumentationTransform,
    GammaCorrection,
    GaussianBlur,
    TrainTransform,
    TwoViewsTransform,
    MultiViewTransform,
    CenterCropTransform,
    CIFARTrainTransform,
    CIFARTestTransform,
)

from .utils import check_image_transform


@pytest.mark.parametrize("gamma_range", [(0.5, 2.0), (0.1, 5.0)])
def test_gamma_correction(gamma_range):
    check_image_transform(GammaCorrection(gamma_range), (48, 48), (48, 48))


@pytest.mark.parametrize("sigma_range", [(0.5, 2.0), (0.1, 5.0)])
def test_gamma_correction(sigma_range):
    check_image_transform(GaussianBlur(sigma_range), (48, 48), (48, 48))


@pytest.mark.parametrize("random_resized_crop", [True, False])
@pytest.mark.parametrize("resize", [64])
@pytest.mark.parametrize("crop_dim", [48])
@pytest.mark.parametrize("scale", [(0.2, 1.0), [0.2, 1.0]])
@pytest.mark.parametrize("interpolation", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("gamma_range", [(0.5, 2.0), [0.5, 2.0]])
@pytest.mark.parametrize("sigma_range", [(0.5, 2.0), [0.5, 2.0]])
@pytest.mark.parametrize("p_blur", [0.0, 1.0])
@pytest.mark.parametrize("p_grayscale", [0.0, 1.0])
@pytest.mark.parametrize("p_hflip", [0.0, 1.0])
@pytest.mark.parametrize(
    "norms",
    [
        (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        (
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ),
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
    ],
)
@pytest.mark.parametrize("p_erase", [0.0, 0.5, 1.0])
def test_train_transform(
    random_resized_crop,
    resize,
    crop_dim,
    scale,
    interpolation,
    gamma_range,
    sigma_range,
    p_blur,
    p_grayscale,
    p_hflip,
    norms,
    p_erase,
):
    check_image_transform(
        TrainTransform(
            random_resized_crop=random_resized_crop,
            resize=resize,
            crop_dim=crop_dim,
            scale=scale,
            interpolation=interpolation,
            gamma_range=gamma_range,
            sigma_range=sigma_range,
            p_blur=p_blur,
            p_grayscale=p_grayscale,
            p_hflip=p_hflip,
            norms=norms,
            p_erase=p_erase,
        ),
        (64, 64),
        (48, 48),
    )


def test_two_views_transform():
    def multiply_by_2(x):
        return 2 * x

    t = TwoViewsTransform(multiply_by_2)
    outputs = t(torch.ones((), dtype=torch.int32))

    assert outputs == (
        2 * torch.ones((), dtype=torch.int32),
        2 * torch.ones((), dtype=torch.int32),
    )


def test_multi_view_transform():
    def multiply_by_2(x):
        return 2 * x

    for i in range(1, 4):
        t = MultiViewTransform(i, image_augmentations=multiply_by_2)
        outputs = t(torch.ones((), dtype=torch.int32))

        l = [2 * torch.ones((), dtype=torch.int32)] * i

        assert outputs == tuple(l)


@pytest.mark.parametrize("resize", [64])
@pytest.mark.parametrize("crop_dim", [48])
@pytest.mark.parametrize("interpolation", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize(
    "norms",
    [
        (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        (
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ),
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
    ],
)
def test_center_crop_transform(
    resize,
    crop_dim,
    interpolation,
    norms,
):
    check_image_transform(
        CenterCropTransform(
            resize=resize,
            crop_dim=crop_dim,
            interpolation=interpolation,
            norms=norms,
        ),
        (64, 64),
        (48, 48),
    )


def test_cifar_train_transform():
    check_image_transform(
        CIFARTrainTransform(),
        (32, 32),
        (32, 32),
    )


def test_cifar_test_transform():
    check_image_transform(
        CIFARTestTransform(),
        (32, 32),
        (32, 32),
    )


def test_albumentations_transform():
    check_image_transform(
        AlbumentationTransform(A.Cutout(num_holes=1)),
        (32, 32),
        (32, 32),
    )
