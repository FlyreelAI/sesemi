#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import torch

from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets import MNIST

from typing import Callable, List, Optional, Union, Any, Tuple
from PIL import Image
from .base import DatasetRegistry

from sesemi.utils import random_indices


class _MNIST(MNIST):
    """An MNIST dataset that supports random subset sampling."""

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


@DatasetRegistry
def mnist(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[Callable] = None,
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
        return _MNIST(
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
            _MNIST(
                root,
                train=(s == "train"),
                download=True,
                transform=image_transform,
                **kwargs,
            )
            for s in subsets
        ]

        return ConcatDataset(dsts)
