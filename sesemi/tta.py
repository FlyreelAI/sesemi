#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
import torch
import torchvision.transforms as T

from collections import defaultdict
from typing import Any, Callable, List, Tuple

from .transforms import (
    IMAGENET_CHANNEL_MEAN,
    IMAGENET_CHANNEL_STD,
    INTERPOLATION_MODE_BY_NAME,
)


def apply_model_to_test_time_augmentations(
    model: Callable[[torch.Tensor], torch.Tensor],
    device: str,
    data: List[List[torch.Tensor]],
    batch_compatible_tensors: bool = True,
) -> List[torch.Tensor]:
    """Applies a model to test-time augmented images.

    Args:
        model: The model to apply.
        device: The device the model is on.
        data: The list of test-time-augmented images. Each
            child list corresponds to test-time augmented images
            from an original input image.
        batch_compatible_tensors: Whether to concatenate compatibly sized
            tensors.

    Returns:
        A list of batches of outputs corresponding to the inputs.
    """
    if batch_compatible_tensors:
        data_index_by_shape = defaultdict(list)
        for i, items in enumerate(data):
            for j, item in enumerate(items):
                data_index_by_shape[item.shape].append((i, j))

        results = []
        for indices in data_index_by_shape.values():
            tensors = [data[i][j] for i, j in indices]
            tensor_batch = torch.stack(tensors, dim=0)
            results_batch = model(tensor_batch.to(device))
            results.extend(zip(indices, list(results_batch)))
        results.sort()

        outputs = [[] for _ in range(len(data))]
        for (i, j), result in results:
            assert len(outputs[i]) == j
            outputs[i].append(result)

        return [torch.stack(x, dim=0) for x in outputs]
    else:
        return [
            torch.cat([model(tensor[None].to(device)) for tensor in tensors], dim=0)
            for tensors in data
        ]


class ListApply:
    """A callable to apply a transform to a list of inputs."""

    def __init__(self, transform: Callable[[List[Any]], List[Any]]):
        self.transform = transform

    def __call__(self, data: List[Any]) -> List[Any]:
        return [self.transform(x) for x in data]


def MultiCropTTA(
    resize: int = 256,
    crop_dim: int = 224,
    num_crops: int = 5,
    interpolation: str = "bilinear",
    norms: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        IMAGENET_CHANNEL_MEAN,
        IMAGENET_CHANNEL_STD,
    ),
) -> Callable:
    """Builds a multi-crop test-time augmentation.

    Args:
        resize: The image size to resize to if random cropping is not applied.
        crop_dim: The output crop dimension.
        num_crop: The number of crops to generate.
        interpolation: The interpolation mode to use when resizing.
        norms: A tuple of the normalization mean and standard deviation.

    Returns:
        A torchvision transform.
    """
    to_tensor = T.ToTensor()
    normalize = T.Normalize(*norms)

    if num_crops == 5:
        multi_crop = T.FiveCrop
    elif num_crops == 10:
        multi_crop = T.TenCrop
    else:
        raise NotImplementedError("Number of crops should be integer of 5 or 10")

    return T.Compose(
        [
            T.Resize(resize, INTERPOLATION_MODE_BY_NAME[interpolation]),
            multi_crop(crop_dim),  # this is a list of PIL Images
            ListApply(T.Compose([to_tensor, normalize])),
        ]
    )
