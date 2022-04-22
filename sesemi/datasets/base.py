#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""SESEMI dataset and registry."""
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset, IterableDataset

from typing import Any, Callable, Dict, List, Optional, Union

from sesemi.registries import CallableRegistry

DatasetRegistry = CallableRegistry[Union[Dataset, IterableDataset]]()


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
    return DatasetRegistry[name](
        to_absolute_path(root), subset=subset, image_transform=image_transform, **kwargs
    )
