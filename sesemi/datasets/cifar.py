#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import torch
import numpy as np

from collections import defaultdict
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from PIL import Image
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from .base import DatasetRegistry

from sesemi.utils import random_indices


class _CIFAR10(CIFAR10):
    """An CIFAR10 dataset that supports random subset sampling."""

    def __init__(
        self,
        *args,
        unlabeled: bool = False,
        random_subset_size: Optional[int] = None,
        random_subset_seed: Any = None,
        **kwargs,
    ):
        """
        Initializes the dataset.

        Args:
            unlabeled: Whether to drop the supervised targets when yielding examples.
            random_subset_size: The size of the random subset.
            random_subset_seed: The seed to use to generate the random subset.
        """
        super().__init__(*args, **kwargs)
        self.unlabeled = unlabeled
        self.random_subset_size = random_subset_size
        self.random_subset_seed = random_subset_seed
        self.random_subset_indices = None
        self.length = len(self.data)
        if self.random_subset_size is not None:
            self.length = self.random_subset_size
            self.random_subset_indices = random_indices(
                self.random_subset_size,
                len(self.data),
                self.random_subset_seed,
                labels=self.targets,
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, index: int
    ) -> Union[Image.Image, Tuple[Image.Image, torch.Tensor]]:
        if self.random_subset_indices is None:
            item = super().__getitem__(index)
        else:
            item = super().__getitem__(self.random_subset_indices[index])

        if self.unlabeled:
            return item[0]
        else:
            return item


class _CIFAR100(CIFAR100):
    """An CIFAR100 dataset that supports random subset sampling."""

    def __init__(
        self,
        *args,
        unlabeled: bool = False,
        random_subset_size: Optional[int] = None,
        random_subset_seed: Any = None,
        **kwargs,
    ):
        """
        Initializes the dataset.

        Args:
            unlabeled: Whether to drop the supervised targets when yielding examples.
            random_subset_size: The size of the random subset.
            random_subset_seed: The seed to use to generate the random subset.
        """
        super().__init__(*args, **kwargs)
        self.unlabeled = unlabeled
        self.random_subset_size = random_subset_size
        self.random_subset_seed = random_subset_seed
        self.random_subset_indices = None
        self.length = len(self.data)
        if self.random_subset_size is not None:
            self.length = self.random_subset_size
            self.random_subset_indices = random_indices(
                self.random_subset_size,
                len(self.data),
                self.random_subset_seed,
                labels=self.targets,
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, index: int
    ) -> Union[Image.Image, Tuple[Image.Image, torch.Tensor]]:
        if self.random_subset_indices is None:
            item = super().__getitem__(index)
        else:
            item = super().__getitem__(self.random_subset_indices[index])

        if self.unlabeled:
            return item[0]
        else:
            return item


def _cifar(
    dataset_cls: Union[Type[_CIFAR10], Type[_CIFAR100]],
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[Callable] = None,
    **kwargs,
) -> Dataset:
    """A CIFAR dataset builder.

    Args:
        dataset_cls: One of (CIFAR10, CIFAR100).
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
        return dataset_cls(
            root,
            train=train,
            download=True,
            transform=image_transform,
            **kwargs,
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
            dataset_cls(
                root,
                train=(s == "train"),
                download=True,
                transform=image_transform,
                **kwargs,
            )
            for s in subsets
        ]

        return ConcatDataset(dsts)


@DatasetRegistry
def cifar10(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[Callable] = None,
    **kwargs,
) -> Dataset:
    """A CIFAR-10 dataset builder.

    Args:
        root: The path to the image folder dataset.
        subset: The subset(s) to use. Subets include (train, test).
        image_transform: The image transformations to apply.

    Returns:
        An `ImageFolder` dataset.
    """
    return _cifar(
        _CIFAR10, root, subset=subset, image_transform=image_transform, **kwargs
    )


@DatasetRegistry
def cifar100(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[Callable] = None,
    **kwargs,
) -> Dataset:
    """A CIFAR-100 dataset builder.

    Args:
        root: The path to the image folder dataset.
        subset: The subset(s) to use. Subets include (train, test).
        image_transform: The image transformations to apply.

    Returns:
        An `ImageFolder` dataset.
    """
    return _cifar(
        _CIFAR100, root, subset=subset, image_transform=image_transform, **kwargs
    )
