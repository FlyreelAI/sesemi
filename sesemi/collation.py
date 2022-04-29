#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""PyTorch collation callables."""
import torch
import torchvision
import torchvision.transforms.functional as TF

from torch import Tensor
from PIL import Image
from typing import Any, List, Optional, Tuple, Callable, TypeVar

T = TypeVar("T")


class RotationCollator:
    """A collation callable to transform a data batch for use with a rotation prediction task.

    Input images are rotated 0, 90, 180, and 270 degrees.
    """

    def __init__(self, return_supervised_labels: bool = False):
        """Initializes the rotation collation callable.

        Args:
            return_supervised_labels: Whether to return supervised class labels or pretext labels.
        """
        self.return_supervised_labels = return_supervised_labels

    def __call__(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Generates a transformed batch of rotation prediction data.

        Args:
            batch: A tuple containing a batch of input image tensors and supervised labels.

        Returns:
            A tuple of rotated image tensors and their associated supervised or pretext labels, where:

                0  corresponds to 0 degrees.
                1  corresponds to 90 degrees.
                2  corresponds to 180 degrees.
                3  corresponds to 270 degrees.
        """
        tensors, labels, rotation_labels = [], [], []
        for tensor, label in batch:
            for k in range(4):
                if k == 0:
                    t = tensor
                else:
                    t = torch.rot90(tensor, k, dims=[1, 2])
                tensors.append(t)
                labels.append(torch.LongTensor([label]))
                rotation_labels.append(torch.LongTensor([k]))
        x = torch.stack(tensors, dim=0)
        y = torch.cat(labels, dim=0)
        p = torch.cat(rotation_labels, dim=0)
        return (x, y) if self.return_supervised_labels else (x, p)


class JigsawCollator:
    """A collation callable to transform a data batch for use with a jigsaw prediction task.

    We select a set of P=5 patch permutations for the jigsaw task by using the maximal Hamming distance.
    https://github.com/bbrattoli/JigsawPuzzlePytorch
    """

    def __init__(
        self,
        p_grayscale: float = 0.1,
        return_supervised_labels: bool = False,
    ):
        """Initializes the jigsaw collation callable.

        Args:
            p_grayscale: probability of converting to grayscale.
            return_supervised_labels: Whether to return supervised class labels or pretext labels.
        """
        self.grid_size = 3
        self.num_grids = 3**2
        self.p_grayscale = p_grayscale
        self.return_supervised_labels = return_supervised_labels
        self.patch_permutations = torch.tensor(
            [
                [5, 7, 2, 6, 0, 3, 8, 1, 4],
                [0, 1, 3, 2, 4, 5, 6, 7, 8],
                [1, 0, 4, 3, 2, 6, 5, 8, 7],
                [2, 3, 0, 1, 5, 8, 7, 4, 6],
                [3, 2, 1, 0, 8, 7, 4, 6, 5],
            ]
        )
        # The number of jigsaw labels is P + 1, with the last label
        # indicating no patch permutation (no shuffling) applied.
        self.num_jigsaw_labels = self.patch_permutations.size(0) + 1

    def crop_patch(self, x: Tensor, idx: int) -> Tensor:
        """Crops a patch of equal height and width dimensions from an input tensor.

        Args:
            x: the input tensor.
            idx: the location index of the ith patch.

        Returns:
            A torchvision transformed tensor.
        """
        window = int(x.size(1) / self.grid_size)
        row = int(idx / self.grid_size)
        col = idx % self.grid_size
        patch = TF.crop(x, row * window, col * window, window, window)
        return self.transform()(patch)

    def transform(self) -> Callable:
        """Image transformations.

        Returns:
            Torchvision transform.
        """
        return torchvision.transforms.Compose(
            [torchvision.transforms.RandomGrayscale(self.p_grayscale)]
        )

    def __call__(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Generates a transformed batch of jigsaw prediction data.

        Args:
            batch: A tuple containing a batch of input image tensors and supervised labels.

        Returns:
            A tuple of jigsaw image tensors and their associated supervised or pretext labels.
        """
        b = len(batch) * (self.num_jigsaw_labels)
        c, h, w = batch[0][0].size()
        tensors, labels, jigsaw_labels = [], [], []
        for tensor, label in batch:
            patches = torch.stack(
                [self.crop_patch(tensor, idx) for idx in range(self.num_grids)], dim=0
            )
            for perm_idx in range(self.num_jigsaw_labels):
                if perm_idx == self.num_jigsaw_labels - 1:
                    data = patches
                else:
                    shuffle_order = self.patch_permutations[perm_idx]
                    data = patches[shuffle_order]
                data = torchvision.utils.make_grid(data, self.grid_size, padding=0)
                tensors.append(data)
                labels.append(torch.LongTensor([label]))
                jigsaw_labels.append(torch.LongTensor([perm_idx]))
        tensors = torch.stack(tensors, dim=0)
        x = torch.zeros(b, c, h, w)
        x[..., : tensors.size(2), : tensors.size(3)] = tensors
        y = torch.cat(labels, dim=0)
        p = torch.cat(jigsaw_labels, dim=0)
        return (x, y) if self.return_supervised_labels else (x, p)


def _identity(x: T) -> T:
    """The identity function used as a default transform."""
    return x


def default_test_time_augmentation(x: T) -> List[T]:
    """An identity test-time augmentation."""
    return [x]


class TestTimeAugmentationCollator:
    def __init__(
        self,
        preprocessing_transform: Optional[Callable],
        test_time_augmentation: Optional[Callable[[Image.Image], List[Image.Image]]],
        postaugmentation_transform: Optional[Callable],
    ):
        """Initializes the collator.

        Args:
            preprocessing_transform: The preprocessing transform.
            test_time_augmentation: The test-time augmentation that takes an image
                and returns a list of augmented versions of that image.
            postaugmentation_transform: A transform to apply after test-time
                augmentations and which should return a tensor.
        """
        self.preprocessing_transform = (
            _identity if preprocessing_transform is None else preprocessing_transform
        )
        self.test_time_augmentation = (
            default_test_time_augmentation
            if test_time_augmentation is None
            else test_time_augmentation
        )
        self.postaugmentation_transform = (
            _identity
            if postaugmentation_transform is None
            else postaugmentation_transform
        )

    def __call__(self, data_batch: List[Image.Image]) -> List[List[torch.Tensor]]:
        """Generates test-time augmented versions of the input images."""
        data_tensors: List[List[torch.Tensor]] = []
        images: List[Image.Image] = []
        for data in data_batch:
            images.append(data)

            augmentations = self.test_time_augmentation(
                self.preprocessing_transform(data)
            )
            tensors = [self.postaugmentation_transform(x) for x in augmentations]
            data_tensors.append(tensors)
        return images, data_tensors
