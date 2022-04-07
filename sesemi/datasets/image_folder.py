#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import os

from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import (
    VisionDataset,
    default_loader,
    IMG_EXTENSIONS,
)

from PIL import Image
from typing import Any, Callable, List, Optional, Union
from .base import register_dataset, ImageTransform


class _ImageFolder(ImageFolder):
    """An ImageFolder dataset that adds metadata to loaded PIL images."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        classes: Optional[List[str]] = None,
    ):
        VisionDataset.__init__(
            self, root, transform=transform, target_transform=target_transform
        )

        if classes is None:
            classes, class_to_idx = self.find_classes(self.root)
        else:
            class_to_idx = {c: i for i, c in enumerate(classes)}

        assert classes is not None

        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        samples = self.make_dataset(
            self.root, class_to_idx, self.extensions, is_valid_file
        )

        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        if isinstance(sample, Image.Image):
            sample.info["filename"] = self.samples[index][0]
        return sample, target


@register_dataset
def image_folder(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
    **kwargs,
) -> Dataset:
    """An image folder dataset builder.

    Args:
        root: The path to the image folder dataset.
        subset: The subset(s) to use.
        image_transform: The image transformations to apply.

    Returns:
        An `ImageFolder` dataset.
    """
    if isinstance(subset, str):
        return _ImageFolder(
            os.path.join(root, subset), transform=image_transform, **kwargs
        )
    else:
        if subset is None:
            subsets = [
                x
                for x in os.listdir(root)
                if not x.startswith(".") and os.path.isdir(os.path.join(root, x))
            ]
        else:
            subsets = subset

        dsts = [
            _ImageFolder(os.path.join(root, s), transform=image_transform, **kwargs)
            for s in subsets
        ]

        return ConcatDataset(dsts)
