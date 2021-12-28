#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""SESEMI datasets and registry."""
import os

from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, STL10
from torchvision.datasets.folder import (
    default_loader,
    has_file_allowed_extension,
    IMG_EXTENSIONS,
)

from typing import Any, Callable, Dict, List, Optional, Type, Union

DatasetBuilder = Callable[..., Union[Dataset, IterableDataset]]
ImageTransform = Callable

DATASET_REGISTRY: Dict[str, DatasetBuilder] = {}


def get_image_files(directory: str, is_valid_file: Callable[[str], bool]) -> List[str]:
    directory = os.path.expanduser(directory)

    files: List[str] = []
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            if is_valid_file(fname):
                path = os.path.join(root, fname)
                files.append(path)

    return files


def default_is_vaild_file(path: str) -> bool:
    return has_file_allowed_extension(path, IMG_EXTENSIONS)


class ImageFile(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Callable[[str], bool] = default_is_vaild_file,
    ):
        super().__init__()
        self.transform = transform
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.image_files = get_image_files(root, self.is_valid_file)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        image = self.loader(self.image_files[index])
        if self.transform is not None:
            image = self.transform(image)
        return image


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
        return ImageFolder(os.path.join(root, subset), transform=image_transform)
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
            ImageFolder(os.path.join(root, s), transform=image_transform)
            for s in subsets
        ]

        return ConcatDataset(dsts)


def _cifar(
    dataset_cls: Union[Type[CIFAR10], Type[CIFAR100]],
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
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
            )
            for s in subsets
        ]

        return ConcatDataset(dsts)


@register_dataset
def cifar10(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
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
        CIFAR10, root, subset=subset, image_transform=image_transform, **kwargs
    )


@register_dataset
def cifar100(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
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
        CIFAR100, root, subset=subset, image_transform=image_transform, **kwargs
    )


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


@register_dataset
def image_file(
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
        return ImageFile(os.path.join(root, subset), transform=image_transform)
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
            ImageFile(os.path.join(root, s), transform=image_transform) for s in subsets
        ]

        return ConcatDataset(dsts)


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
        root, subset=subset, image_transform=image_transform, **kwargs
    )
