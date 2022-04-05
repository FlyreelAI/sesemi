#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import MNIST

from typing import List, Optional, Union
from .base import register_dataset, ImageTransform


@register_dataset
def mnist(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
    **kwargs,
) -> Dataset:
    """An MNIST dataset builder.

    Args:
        root: The path to the image folder dataset.
        subset: The subset(s) to use. Subets include (train, test).
        image_transform: The image transformations to apply.

    Returns:
        An `ImageFolder` dataset.
    """
    if isinstance(subset, str):
        assert subset in {
            "train",
            "test",
        }, f"invalid subset {subset}, only support {{train, test}}"
        train = subset == "train"
        return MNIST(
            root,
            train=train,
            download=True,
            transform=image_transform,
        )
    else:
        if subset is None:
            subsets = ["train", "test"]
        else:
            subsets = subset

        for s in subsets:
            assert s in {
                "train",
                "test",
            }, f"invalid subset {subset}, only support {{train, test}}"

        dsts = [
            MNIST(
                root,
                train=(s == "train"),
                download=True,
                transform=image_transform,
            )
            for s in subsets
        ]

        return ConcatDataset(dsts)
