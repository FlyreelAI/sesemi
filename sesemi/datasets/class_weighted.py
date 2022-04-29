#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from sesemi.utils import get_distributed_rank
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset, IterableDataset

from typing import List, Optional, Union
from .base import register_dataset, ImageTransform


class ClassWeightedDataset(IterableDataset):
    """A class-weighted dataset.

    Takes a finite dataset and generates an iterable dataset
    which samples using the provided class weights.
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """Initializes the class-weighted dataset.

        Args:
            dataset: The dataset to sample from.
            weights: The optional weights to sample from the different classes.
                Defaults to a uniform distribution over the classes (balanced).
            seed: An optional random seed used in the iterable dataset.
                The rank of the data loading process is added to the seed in the
                distributed setting.
        """
        super().__init__()
        self._dataset = dataset
        self._labels = [x[1] for x in self._dataset]
        self._num_classes = max(self._labels) + 1
        self._weights = weights or ([1.0] * self._num_classes)
        s = sum(self._weights)
        self._weights = [x / s for x in self._weights]
        assert len(self._weights) == self._num_classes

        self._seed = seed if seed is None else seed + (get_distributed_rank() or 0)
        self._random_state = np.random.RandomState(self._seed)

        self._indices_by_label = defaultdict(list)
        for i, l in enumerate(self._labels):
            self._indices_by_label[l].append(i)

        self._class_labels = list(self._indices_by_label.keys())
        self._class_labels.sort()

        assert self._class_labels == list(range(self._num_classes))

        self._sampling_indices_by_label = {}
        self._sampling_index_by_label = {}
        for k, v in self._indices_by_label.items():
            self._sampling_indices_by_label[k] = list(v)
            self._random_state.shuffle(self._sampling_indices_by_label[k])
            self._sampling_index_by_label[k] = 0

    def __iter__(self):
        while True:
            class_label = self._random_state.choice(self._class_labels, p=self._weights)
            sampling_index = self._sampling_index_by_label[class_label]
            self._sampling_index_by_label[class_label] = sampling_index + 1
            dataset_index = self._sampling_indices_by_label[class_label][sampling_index]
            if self._sampling_index_by_label[class_label] == len(
                self._sampling_indices_by_label[class_label]
            ):
                self._random_state.shuffle(self._sampling_indices_by_label[class_label])
                self._sampling_index_by_label[class_label] = 0
            yield self._dataset[dataset_index]


@register_dataset
def class_weighted(
    root: str,
    *,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
    dataset: Dataset,
    weights: Optional[List[float]] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> IterableDataset:
    """A class-weighted dataset builder.

    Args:
        root: This is ignored.
        subset: The subset(s) to use.
        image_transform: The image transformations to apply.
        dataset: The dataset to sample from.
        weights: The optional weights to sample from the different classes.
            Defaults to a uniform distribution over the classes (balanced).
        seed: An optional random seed used in the iterable dataset.
            The rank of the data loading process is added to the seed in the
            distributed setting.

    Returns:
        A class-weighted iterable dataset dataset.
    """
    assert subset is None, "class-weighted datasets don't support subsets"
    return ClassWeightedDataset(dataset, weights=weights, seed=seed)
