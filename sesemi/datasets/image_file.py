#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import os

from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets.folder import (
    default_loader,
    has_file_allowed_extension,
    IMG_EXTENSIONS,
)

from PIL import Image
from typing import Any, Callable, List, Optional, Union
from .base import DatasetRegistry


def default_is_vaild_file(path: str) -> bool:
    """The default callable to determine if a file is a valid image."""
    return has_file_allowed_extension(path, IMG_EXTENSIONS)


def get_image_files(directory: str, is_valid_file: Callable[[str], bool]) -> List[str]:
    """Finds the full list of image files recursively under a directory path.

    Args:
        directory: The root directory to search for image files.
        is_valid_file: A callable to determine if a file is a valid image.

    Returns:
        The list of paths to the image files.
    """
    directory = os.path.expanduser(directory)

    files: List[str] = []
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            if is_valid_file(fname):
                path = os.path.join(root, fname)
                files.append(path)

    return files


class ImageFile(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Callable[[str], bool] = default_is_vaild_file,
    ):
        """An image-file dataset.

        Args:
            root: The path to the dataset root.
            transform: An optional image transform.
            loader: The image loading callable.
            is_valid_file: A callable to determine if a file is a valid image.
        """
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
        if isinstance(image, Image.Image):
            image.info["filename"] = self.image_files[index]
        return image


@DatasetRegistry
def image_file(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[Callable] = None,
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
