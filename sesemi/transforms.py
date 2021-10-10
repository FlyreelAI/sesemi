#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Image transforms."""
import os
import numpy as np

import torch
import torchvision.transforms.functional as TF

from typing import Callable, List, Tuple
from torch import Tensor
from torchvision import datasets, transforms
import torchvision.transforms as T

import kornia.augmentation as K
import kornia.filters as KF

from .collation import RotationTransformer, JigsawTransformer
from .utils import validate_paths


IMAGENET_CHANNEL_MEAN = (0.485, 0.456, 0.406)
IMAGENET_CHANNEL_STD = (0.229, 0.224, 0.225)

CIFAR10_CHANNEL_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_CHANNEL_STD = (0.2023, 0.1994, 0.2010)

modes_mapping = {
    "nearest": TF.InterpolationMode.NEAREST,
    "bilinear": TF.InterpolationMode.BILINEAR,
    "bicubic": TF.InterpolationMode.BICUBIC,
    "box": TF.InterpolationMode.BOX,
    "hamming": TF.InterpolationMode.HAMMING,
    "lanczos": TF.InterpolationMode.LANCZOS,
}


class GammaCorrection:
    def __init__(self, gamma_range: Tuple[float, float] = (0.5, 2.0)):
        """Initializes the gamma correction transform.

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


def train_transforms(
    random_resized_crop: bool = True,
    resize: int = 256,
    crop_dim: int = 224,
    scale: Tuple[float, float] = (0.2, 1.0),
    interpolation: str = "bilinear",
    gamma_range: Tuple[float, float] = (0.5, 1.5),
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
        interpolation: The interpolation mode to use when resizing.
        gamma_range: The gamma correction range.
        p_hflip: The horiziontal random flip probability.
        norms: A tuple of the normalization mean and standard deviation.
        p_erase: The probability of random erasing.

    Returns:
        A torchvision transform.
    """
    interpolation = modes_mapping[interpolation]
    default_transforms = [
        GammaCorrection(gamma_range),
        transforms.RandomHorizontalFlip(p_hflip),
        transforms.ToTensor(),
        transforms.Normalize(*norms),
        transforms.RandomErasing(p=p_erase, value="random"),
    ]
    if random_resized_crop:
        return transforms.Compose(
            [transforms.RandomResizedCrop(crop_dim, scale, interpolation=interpolation)]
            + default_transforms
        )
    else:
        return transforms.Compose(
            [transforms.Resize(resize, interpolation), transforms.RandomCrop(crop_dim)]
            + default_transforms
        )


def center_crop_transforms(
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
            transforms.Resize(resize, modes_mapping[interpolation]),
            transforms.CenterCrop(crop_dim),
            transforms.ToTensor(),
            transforms.Normalize(*norms),
        ]
    )


def multi_crop_transforms(
    resize: int = 256,
    crop_dim: int = 224,
    num_crop: int = 5,
    interpolation: str = "bilinear",
    norms: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        IMAGENET_CHANNEL_MEAN,
        IMAGENET_CHANNEL_STD,
    ),
) -> Callable:
    """Builds a multi-crop transform.

    Args:
        resize: The image size to resize to if random cropping is not applied.
        crop_dim: The output crop dimension.
        num_crop: The number of crops to generate.
        interpolation: The interpolation mode to use when resizing.
        norms: A tuple of the normalization mean and standard deviation.

    Returns:
        A torchvision transform.
    """
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(*norms)
    Lambda = transforms.Lambda
    if num_crop == 5:
        multi_crop = transforms.FiveCrop
    elif num_crop == 10:
        multi_crop = transforms.TenCrop
    else:
        raise NotImplementedError("Number of crops should be integer of 5 or 10")

    return transforms.Compose(
        [
            transforms.Resize(resize, modes_mapping[interpolation]),
            multi_crop(crop_dim),  # this is a list of PIL Images
            Lambda(lambda crops: torch.stack([to_tensor(crop) for crop in crops])),
            Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ]
    )


def cifar_train_transform() -> Callable:
    """Returns the standard CIFAR training transforms."""
    return T.Compose(
        [
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR10_CHANNEL_MEAN, CIFAR10_CHANNEL_STD),
        ]
    )


def cifar_test_transform() -> Callable:
    """Returns the standard CIFAR test-time transform."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR10_CHANNEL_MEAN, CIFAR10_CHANNEL_STD),
        ]
    )


def simclr_image_augmentation(
    horizontal_flip: bool = True,
    strength: float = 0.5,
    size=(224, 224),
    interpolation: str = "BICUBIC",
    mean: Tuple[float, float, float] = IMAGENET_CHANNEL_MEAN,
    std: Tuple[float, float, float] = IMAGENET_CHANNEL_STD,
    enable_blurring: bool = True,
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.3333333333),
) -> Callable:
    """SimCLR image augmentations.

    Args:
        horizontal_flip: Whether to perform random horizontal flipping.
        strength: The strength of the color augmentations.
        size: The crop size.
        interpolation: The interpolation mode to use when rescaling the image.
        mean: The mean to subtract from the image.
        std: The standard deviation used to normalize the image.
        enable_blurring: Whether to perform gaussian blurring.
        scale: The scale range used when rescaling the image.
        ratio: The aspect ratio used when rescaling the image.
    
    Returns:
        A callable PyTorch transform.
    """
    size = tuple(size)
    transformations: List[Callable] = [
        T.ToTensor(),
        K.RandomResizedCrop(
            size=size,
            scale=scale,
            ratio=ratio,
            resample=interpolation,
        ),
        K.RandomHorizontalFlip(
            p=0.5 if horizontal_flip else 0.0,
        ),
        K.ColorJitter(
            brightness=0.8 * strength,
            contrast=0.8 * strength,
            saturation=0.8 * strength,
            hue=0.2 * strength,
        ),
        K.RandomGrayscale(
            p=0.2,
        ),
    ]

    if enable_blurring:
        transformations.append(
            KF.GaussianBlur2d(
                kernel_size=(23, 23),
                sigma=(0.1, 2.0),
            )
        )

    transformations.extend(
        [
            K.Normalize(list(mean), list(std)),
        ]
    )

    return T.Compose(transformations)


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

    transformations = train_transforms(
        gamma_range=gamma_range, p_hflip=p_hflip, norms=(mean, std), p_erase=p_erase
    )
    dataset = datasets.ImageFolder(args.img_dir, transformations)
    print("transformations:\n", transformations)
    print("dataset size: {}".format(len(dataset)))
    to_pil_image = transforms.ToPILImage()
    rotate = RotationTransformer()
    jigsaw = JigsawTransformer(p_grayscale=0.5)
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
