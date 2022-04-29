#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""SESEMI datasets and registry."""
import os
import yaml
import h5py
from sesemi.utils import get_distributed_rank
import torch
import numpy as np

from hydra.utils import to_absolute_path
from collections import defaultdict
from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, STL10, MNIST
from torchvision.datasets.folder import (
    VisionDataset,
    default_loader,
    has_file_allowed_extension,
    IMG_EXTENSIONS,
)

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

DatasetBuilder = Callable[..., Union[Dataset, IterableDataset]]
ImageTransform = Callable

DATASET_REGISTRY: Dict[str, DatasetBuilder] = {}


def register_dataset(builder: DatasetBuilder) -> DatasetBuilder:
    """A decorator to register a dataset builder.

    The lowercase name of the builder is used in the registry.

    Args:
        builder: A dataset builder.

    Returns:
        The input dataset builder.
    """
    DATASET_REGISTRY[builder.__name__.lower()] = builder
    return builder


def dataset(
    name: str,
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[Callable] = None,
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    """Builds a dataset.

    Args:
        name: The name of the dataset to build.
        root: The path to the image folder dataset.
        subset: The subset(s) to use.
        image_transform: The image transformations to apply.
        **kwargs: Any other arguments to forward to the underlying dataset builder.

    Returns:
        The dataset.
    """
    return DATASET_REGISTRY[name](
        to_absolute_path(root), subset=subset, image_transform=image_transform, **kwargs
    )
