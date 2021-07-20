import os

from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from torchvision.datasets import ImageFolder

from typing import Callable, Dict, Optional, Union

DatasetBuilder = Callable[..., Union[Dataset, IterableDataset]]
ImageTransform = Callable

DATASET_REGISTRY: Dict[str, DatasetBuilder] = {}


def register_dataset(dst: DatasetBuilder) -> DatasetBuilder:
    DATASET_REGISTRY[dst.__name__.lower()] = dst
    return dst


@register_dataset
def image_folder(
    root: str,
    subset: Optional[str] = None,
    image_transform: Optional[ImageTransform] = None,
) -> Dataset:
    if subset is not None:
        return ImageFolder(os.path.join(root, subset), transform=image_transform)
    else:
        subsets = [
            x
            for x in os.listdir(root)
            if not x.startswith(".") and os.path.isdir(os.path.join(root, x))
        ]
        dsts = [
            ImageFolder(os.path.join(root, s), transform=image_transform)
            for s in subsets
        ]
        return ConcatDataset(dsts)


def dataset(
    name: str,
    root: str,
    subset: Optional[str] = None,
    image_transform: Optional[Callable] = None,
) -> Union[Dataset, IterableDataset]:
    return DATASET_REGISTRY[name](root, subset=subset, image_transform=image_transform)
