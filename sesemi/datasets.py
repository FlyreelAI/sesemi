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


def _random_indices(
    n: int, length: int, seed: Any = None, labels: Optional[List[int]] = None
) -> List[int]:
    """Samples `n` random indices for a dataset of length `length.

    Args:
        n: The number of random indices to sample.
        length: The length of the original dataset.
        seed: An optional random seed to use.
        labels: An optional list of labels for each item in the dataset
            to use for stratification during sampling.

    Returns:
        A list of sampled random indices.
    """
    assert n <= length, (
        f"number of random indices ({n}) must be less than or "
        f"equal to the size of the total length ({length})"
    )

    rs = np.random.RandomState(seed)
    if labels is not None:
        assert (
            len(labels) == length
        ), "number of labels must match the provided dataset length"

        indices_by_labels = defaultdict(list)
        for i, l in enumerate(labels):
            indices_by_labels[l].append(i)

        counts_by_label = {l: len(v) for l, v in indices_by_labels.items()}

        num_samples_per_label = {
            l: (c * n) // length for l, c in counts_by_label.items()
        }
        remainders_per_label = {l: (c * n) % length for l, c in counts_by_label.items()}
        num_remaining_samples = sum(remainders_per_label.values())

        possible_label_classes = list(remainders_per_label.keys())
        if num_remaining_samples > 0:
            for i in range(num_remaining_samples):
                label_weights = [
                    remainders_per_label[l] / num_remaining_samples
                    for l in remainders_per_label
                ]
                s = rs.choice(possible_label_classes, p=label_weights)
                remainders_per_label[s] -= 1
                num_samples_per_label[s] += 1
                num_remaining_samples -= 1

        sample_indices_by_label = {
            l: rs.choice(
                indices_by_labels[l], num_samples_per_label[l], replace=False
            ).tolist()
            for l in indices_by_labels
        }

        print(sample_indices_by_label)

        return sum(sample_indices_by_label.values(), [])
    else:
        indices = rs.choice(length, size=n, replace=False)
        return indices.tolist()


class _ImageFolder(ImageFolder):
    """An ImageFolder dataset that adds metadata to loaded PIL images."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        classes: Optional[List[str]] = None,
    ):
        VisionDataset.__init__(
            self, root, transform=transform, target_transform=target_transform
        )

        if classes is None:
            classes, class_to_idx = self.find_classes(self.root)
        else:
            class_to_idx = {c: i for i, c in enumerate(classes)}

        assert classes is not None

        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        samples = self.make_dataset(
            self.root, class_to_idx, self.extensions, is_valid_file
        )

        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        if isinstance(sample, Image.Image):
            sample.info["filename"] = self.samples[index][0]
        return sample, target


class _CIFAR10(CIFAR10):
    """An CIFAR10 dataset that supports random subset sampling."""

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
            self.random_subset_indices = _random_indices(
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


class _CIFAR100(CIFAR100):
    """An CIFAR100 dataset that supports random subset sampling."""

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
            self.random_subset_indices = _random_indices(
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


def default_is_vaild_file(path: str) -> bool:
    """The default callable to determine if a file is a valid image."""
    return has_file_allowed_extension(path, IMG_EXTENSIONS)


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
        return _ImageFolder(
            os.path.join(root, subset), transform=image_transform, **kwargs
        )
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
            _ImageFolder(os.path.join(root, s), transform=image_transform, **kwargs)
            for s in subsets
        ]

        return ConcatDataset(dsts)


@register_dataset
def pseudo(
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
    assert subset is None, "psuedo-labeled datasets don't have subsets"
    return PseudoDataset(
        root=root,
        transform=image_transform,
        **kwargs,
    )


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


def _cifar(
    dataset_cls: Union[Type[_CIFAR10], Type[_CIFAR100]],
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
            dataset_cls(
                root,
                train=(s == "train"),
                download=True,
                transform=image_transform,
                **kwargs,
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
        _CIFAR10, root, subset=subset, image_transform=image_transform, **kwargs
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
        _CIFAR100, root, subset=subset, image_transform=image_transform, **kwargs
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


@register_dataset
def mnist(
    root: str,
    subset: Optional[Union[str, List[str]]] = None,
    image_transform: Optional[ImageTransform] = None,
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
        return MNIST(
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
            MNIST(
                root,
                train=(s == "train"),
                download=True,
                transform=image_transform,
            )
            for s in subsets
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
        to_absolute_path(root), subset=subset, image_transform=image_transform, **kwargs
    )
