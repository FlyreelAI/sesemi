#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from torch.utils.data import ConcatDataset, Dataset

from typing import List, Optional, Union
from .base import register_dataset, ImageTransform


@register_dataset
def concat(
    root: str,
    *,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
    datasets: List[Dataset],
    **kwargs,
) -> Dataset:
    """An image folder dataset builder.

    Args:
        root: This is ignored for concat datasets.
        subset: The subset(s) to use.
        image_transform: The image transformations to apply.

    Returns:
        An `ImageFolder` dataset.
    """
    assert subset is None, "concat datasets don't support subsets"
    return ConcatDataset(datasets)