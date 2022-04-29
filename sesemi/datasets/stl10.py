#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import STL10

from typing import List, Optional, Union
from .base import register_dataset, ImageTransform


@register_dataset
def stl10(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
    **kwargs,
) -> Dataset:
    """An STL10 dataset builder.

    Args:
        root: The path to the image folder dataset.
        subset: The subset(s) to use. Subets include
            (train, test, unlabeled, train+unlabeled).
        image_transform: The image transformations to apply.

    Returns:
        An `ImageFolder` dataset.
    """
    if isinstance(subset, str):
        assert subset in {
            "train",
            "test",
            "unlabeled",
            "train+unlabeled",
        }, f"invalid subset {subset}, only support {{train, test, unlabeled, train+unlabeled}}"
        return STL10(
            root,
            split=subset,
            download=True,
            transform=image_transform,
            **kwargs,
        )
    else:
        if subset is None:
            subsets = ["train", "test", "unlabeled"]
        else:
            subsets = subset

        for s in subsets:
            assert s in {
                "train",
                "test",
                "unlabeled",
                "train+unlabeled",
            }, f"invalid subset {subset}, only support {{train, test, unlabeled, train+unlabeled}}"

        dsts = [
            STL10(
                root,
                split=s,
                download=True,
                transform=image_transform,
                **kwargs,
            )
            for s in subsets
        ]

        return ConcatDataset(dsts)