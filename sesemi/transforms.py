#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Image transforms."""
import os
import numpy as np

from PIL import ImageFilter
from typing import Any, Callable, List, Optional, Tuple, Union

from torch import Tensor
from PIL import ImageFilter
from typing import Callable, Tuple

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from omegaconf import ListConfig
from torchvision import datasets, transforms

from .collation import RotationCollator, JigsawCollator
from .utils import validate_paths


IMAGENET_CHANNEL_MEAN = (0.485, 0.456, 0.406)
IMAGENET_CHANNEL_STD = (0.229, 0.224, 0.225)

CIFAR_CHANNEL_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_CHANNEL_STD = (0.2023, 0.1994, 0.2010)

INTERPOLATION_MODE_BY_NAME = {
    "nearest": TF.InterpolationMode.NEAREST,
    "bilinear": TF.InterpolationMode.BILINEAR,
    "bicubic": TF.InterpolationMode.BICUBIC,
    "box": TF.InterpolationMode.BOX,
    "hamming": TF.InterpolationMode.HAMMING,
    "lanczos": TF.InterpolationMode.LANCZOS,
}


class GammaCorrection:
    def __init__(self, gamma_range: Tuple[float, float] = (0.5, 2.0)):
        """Initializes the gamma correction augmentation.

        Args:
            gamma_range: A tuple defining the lower and upper bound of the range.
        """
        self.gamma_range = gamma_range

    def __call__(self, x: Tensor) -> Tensor:
        """Applies random gamma correction to the input image.

        Args:
            x: The input PIL image or tensor.

        Returns:
            The gamma corrected image.
        """
        gamma = np.random.uniform(*self.gamma_range)
        return TF.adjust_gamma(x, gamma, gain=1)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(r={})".format(self.gamma_range)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR
    https://arxiv.org/abs/2002.05709
    """

    def __init__(self, sigma_range: Tuple[float, float] = (0.1, 2.0)):
        """Initializes the Gaussian blur augmentation.

        Args:
            sigma_range: A tuple defining the lower and upper bound of the range.
        """
        self.sigma_range = sigma_range

    def __call__(self, x):
        """Applies random Gaussian blur to the input image.

        Args:
            x: The input PIL image.

        Returns:
            The Gaussian blurred image.
        """
        sigma = np.random.uniform(*self.sigma_range)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(s={})".format(self.sigma_range)


def TrainTransform(
    random_resized_crop: bool = True,
    resize: int = 256,
    crop_dim: int = 224,
    scale: Tuple[float, float] = (0.2, 1.0),
    interpolation: str = "bilinear",
    gamma_range: Tuple[float, float] = (0.5, 1.5),
    sigma_range: Tuple[float, float] = (0.1, 2.0),
    p_blur: float = 0.0,
    p_grayscale: float = 0.0,
    p_hflip: float = 0.5,
    norms: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        IMAGENET_CHANNEL_MEAN,
        IMAGENET_CHANNEL_STD,
    ),
    p_erase: float = 0.0,
) -> Callable:
    """Builds a torchvision training transform.

    Args:
        random_resized_crop: Whether to apply random resized cropping.
        resize: The image size to resize to if random cropping is not applied.
        crop_dim: The output crop dimension.
        scale: The scale of the random crop.
        interpolation: The interpolation mode to use when resizing.
        gamma_range: The gamma correction range.
        sigma_range: The Gaussian blur range.
        p_blur: The probability of applying Gaussian blur.
        p_grayscale: The probability of applying random grayscale.
        p_hflip: The horiziontal random flip probability.
        norms: A tuple of the normalization mean and standard deviation.
        p_erase: The probability of random erasing.

    Returns:
        A torchvision transform.
    """
    interpolation = INTERPOLATION_MODE_BY_NAME[interpolation]
    augmentations = [
        GammaCorrection(gamma_range),
        transforms.RandomGrayscale(p_grayscale),
        transforms.RandomApply([GaussianBlur(sigma_range)], p_blur),
        transforms.RandomHorizontalFlip(p_hflip),
        transforms.ToTensor(),
        transforms.Normalize(*norms),
        transforms.RandomErasing(p=p_erase, value="random"),
    ]
    if random_resized_crop:
        return transforms.Compose(
            [transforms.RandomResizedCrop(crop_dim, scale, interpolation=interpolation)]
            + augmentations
        )
    else:
        return transforms.Compose(
            [transforms.Resize(resize, interpolation), transforms.RandomCrop(crop_dim)]
            + augmentations
        )


class TwoViewsTransform:
    """Randomly crops two views of the same image."""

    def __init__(
        self,
        transform: Callable,
    ):
        """Initializes the two-views transform.

        Args:
            transform: The base transform.
        """
        self.transform = transform

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Applies random two-views transform to the input image.

        Args:
            x: The input tensor.

        Returns:
            A tuple of two randomly cropped views with augmentations.
        """
        one = self.transform(x)
        two = self.transform(x)
        return (one, two)


class MultiViewTransform:
    """A multi-view image transform."""

    def __init__(
        self,
        num_views: int,
        image_augmentations: Optional[Union[Callable, List[Callable]]] = None,
    ):
        self.num_views = num_views

        if isinstance(image_augmentations, (list, ListConfig)):
            assert (
                len(image_augmentations) == num_views
            ), f"must provide {num_views} augmentations if multiple given"
            self.image_augmentations = image_augmentations
        else:
            self.image_augmentations = [image_augmentations] * self.num_views

    def __call__(self, image) -> Tuple[Any, ...]:
        """Generates a transformed data batch of rotation prediction data.

        Arguments:
            image: An input image.

        Returns:
            A tuple of augmented views.
        """

        views = []
        for i in range(self.num_views):
            if self.image_augmentations[i] is not None:
                view = self.image_augmentations[i](image)
            else:
                view = image
            views.append(view)
        return tuple(views)


def CenterCropTransform(
    resize: int = 256,
    crop_dim: int = 224,
    interpolation: str = "bilinear",
    norms: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        IMAGENET_CHANNEL_MEAN,
        IMAGENET_CHANNEL_STD,
    ),
) -> Callable:
    """Builds a center cropping transform.

    Args:
        resize: The image size to resize to if random cropping is not applied.
        crop_dim: The output crop dimension.
        interpolation: The interpolation mode to use when resizing.
        norms: A tuple of the normalization mean and standard deviation.

    Returns:
        A torchvision transform.
    """
    return transforms.Compose(
        [
            transforms.Resize(resize, INTERPOLATION_MODE_BY_NAME[interpolation]),
            transforms.CenterCrop(crop_dim),
            transforms.ToTensor(),
            transforms.Normalize(*norms),
        ]
    )


def CIFARTrainTransform() -> Callable:
    """Returns the standard CIFAR training transforms."""
    return T.Compose(
        [
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR_CHANNEL_MEAN, CIFAR_CHANNEL_STD),
        ]
    )


def CIFARTestTransform() -> Callable:
    """Returns the standard CIFAR test-time transform."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR_CHANNEL_MEAN, CIFAR_CHANNEL_STD),
        ]
    )


if __name__ == "__main__":
    import argparse
    from tqdm import trange

    parser = argparse.ArgumentParser(
        description="Dataset Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="`img_dir` must have one or more subdirs containing images",
    )
    parser.add_argument("--head", default=10, type=int, help="visualize first k images")
    parser.add_argument(
        "--hflip", action="store_true", help="apply random horizontal flip"
    )
    parser.add_argument("--erase", action="store_true", help="apply random erase")
    parser.add_argument(
        "--gamma",
        action="store_true",
        help="apply random luminance and gamma correction",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="apply channel-wise mean-std normalization",
    )
    parser.add_argument(
        "--visualization",
        choices=["none", "rotation", "jigsaw"],
        default="rotation",
        help="visualize transformations on unlabeled data",
    )
    parser.add_argument(
        "--out-dir",
        default="./sample_dataset_vis",
        help="directory to save images for visualization",
    )
    args = parser.parse_args()

    validate_paths([args.img_dir])
    os.makedirs(args.out_dir, exist_ok=True)

    print("args:")
    for key, val in args.__dict__.items():
        print("    {:20} {}".format(key, val))

    p_hflip = 0.5 if args.hflip else 0.0
    p_erase = 0.5 if args.erase else 0.0
    gamma_range = (0.5, 1.5) if args.gamma else (1.0, 1.0)
    (mean, std) = (
        (IMAGENET_CHANNEL_MEAN, IMAGENET_CHANNEL_STD)
        if args.normalize
        else ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    )

    transformations = TrainTransform(
        gamma_range=gamma_range, p_hflip=p_hflip, norms=(mean, std), p_erase=p_erase
    )
    dataset = datasets.ImageFolder(args.img_dir, transformations)
    print("transformations:\n", transformations)
    print("dataset size: {}".format(len(dataset)))
    to_pil_image = transforms.ToPILImage()
    rotate = RotationCollator()
    jigsaw = JigsawCollator(p_grayscale=0.5)
    for i in trange(args.head):
        fpath = dataset.imgs[i][0]
        fname = fpath.split("/")[-1]
        x, dummy_label = dataset[i]
        if args.visualization == "none":
            image = to_pil_image(x)
            image.save(os.path.join(args.out_dir, fname))
            continue
        if args.visualization == "rotation":
            tensors, indices = rotate([(x, dummy_label)])
        elif args.visualization == "jigsaw":
            tensors, indices = jigsaw([(x, dummy_label)])
        for x, ind in zip(*(tensors, indices)):
            image = to_pil_image(x)
            image.save(
                os.path.join(args.out_dir, f"vis_{args.visualization}_{ind}_" + fname)
            )
