#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import yaml
import os
import torch
import h5py
import numpy as np

from torch.utils.data import Dataset

from PIL import Image

from typing import List, Optional, Union, Callable
from .base import DatasetRegistry


class PseudoDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_probability_target: bool = False,
    ):
        """A pseudo-labeled dataset.

        Args:
            root: The path to the dataset root.
            transform: An optional image transform.
            target_transform: An optional target transform.
            use_probability_target: Whether to use the probabilities
                as learning targets rather than integer labels.
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.use_probability_target = use_probability_target

        with open(os.path.join(self.root, "metadata.yaml"), "r") as f:
            self.metadata = yaml.safe_load(f)

    def __len__(self) -> int:
        return len(self.metadata["ids"])

    def __getitem__(self, index: int):
        id_ = self.metadata["ids"][index]
        image_path = os.path.join(self.root, "images", f"{id_}.jpg")
        image = Image.open(image_path)
        image.info["filename"] = image_path

        if self.transform is not None:
            image = self.transform(image)

        prediction = h5py.File(os.path.join(self.root, "predictions", f"{id_}.h5"), "r")
        probabilities = np.array(prediction["probabilities"])
        label = probabilities.argmax()

        if self.use_probability_target:
            target = torch.tensor(probabilities)
        else:
            target = label

        return image, target


@DatasetRegistry
def pseudo(
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
    assert subset is None, "psuedo-labeled datasets don't have subsets"
    return PseudoDataset(
        root=root,
        transform=image_transform,
        **kwargs,
    )
