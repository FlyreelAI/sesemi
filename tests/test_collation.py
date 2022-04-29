import numpy as np
import pytest
import torch

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image

from sesemi.collation import (
    RotationCollator,
    JigsawCollator,
    TestTimeAugmentationCollator,
)


@pytest.mark.parametrize("return_supervised_labels", [False, True])
def test_rotation_collator(return_supervised_labels):
    collator = RotationCollator(return_supervised_labels=return_supervised_labels)

    num_images = 10

    image = torch.ones((3, 32, 32), dtype=torch.uint8)
    image[:, 0, :] = 0

    batch = [
        (image.clone(), torch.ones((), dtype=torch.long) * i) for i in range(num_images)
    ]

    rotations, labels = collator(batch)

    assert len(rotations) == len(labels)
    assert len(rotations) == (4 * len(batch))

    for i in range(num_images):
        for j in range(4):
            ix = i * 4 + j
            im, lb = rotations[ix], labels[ix]
            assert torch.all(im == torch.rot90(image, j, dims=[1, 2])).item()
            if return_supervised_labels:
                assert lb == i
            else:
                assert lb == j


@pytest.mark.parametrize("p_grayscale", [0.0, 1.0])
@pytest.mark.parametrize("return_supervised_labels", [False, True])
def test_jigsaw_collator(p_grayscale, return_supervised_labels):
    collator = JigsawCollator(
        p_grayscale=p_grayscale,
        return_supervised_labels=return_supervised_labels,
    )

    num_images = 10

    image = torch.ones((3, 32, 32), dtype=torch.uint8)
    image[:, 0, :] = 0

    batch = [
        (image.clone(), torch.ones((), dtype=torch.long) * i) for i in range(num_images)
    ]

    jigsaw, labels = collator(batch)

    assert len(jigsaw) == len(labels)
    assert len(jigsaw) == (num_images * collator.num_jigsaw_labels)

    for i in range(num_images):
        for j in range(collator.num_jigsaw_labels):
            ix = i * collator.num_jigsaw_labels + j
            _, lb = jigsaw[ix], labels[ix]
            if return_supervised_labels:
                assert lb == i
            else:
                assert lb == j


def test_tta_collator():
    data = [Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8)) for _ in range(100)]

    collator = TestTimeAugmentationCollator(
        preprocessing_transform=None,
        test_time_augmentation=None,
        postaugmentation_transform=None,
    )

    images, tensors = collator(data)
    assert len(images) == len(tensors)
    assert len(images) == len(data)
    for o, t, d in zip(images, tensors, data):
        assert o == d
        assert len(t) == 1
        assert t[0] == d

    collator = TestTimeAugmentationCollator(
        preprocessing_transform=None,
        test_time_augmentation=None,
        postaugmentation_transform=T.ToTensor(),
    )

    images, tensors = collator(data)
    assert len(images) == len(tensors)
    assert len(images) == len(data)
    for o, t, d in zip(images, tensors, data):
        assert o == d
        assert len(t) == 1
        assert torch.allclose(t[0], TF.to_tensor(d))

    collator = TestTimeAugmentationCollator(
        preprocessing_transform=lambda x: x,
        test_time_augmentation=None,
        postaugmentation_transform=None,
    )

    images, tensors = collator(data)
    assert len(images) == len(tensors)
    assert len(images) == len(data)
    for o, t, d in zip(images, tensors, data):
        assert o == d
        assert len(t) == 1
        assert t[0] == d

    collator = TestTimeAugmentationCollator(
        preprocessing_transform=None,
        test_time_augmentation=lambda x: [x, x],
        postaugmentation_transform=None,
    )

    images, tensors = collator(data)
    assert len(images) == len(tensors)
    assert len(images) == len(data)
    for o, t, d in zip(images, tensors, data):
        assert o == d
        assert len(t) == 2
        assert t[0] == d
        assert t[1] == d

    collator = TestTimeAugmentationCollator(
        preprocessing_transform=lambda x: x,
        test_time_augmentation=lambda x: [x, x],
        postaugmentation_transform=T.ToTensor(),
    )

    images, tensors = collator(data)
    assert len(images) == len(tensors)
    assert len(images) == len(data)
    for o, t, d in zip(images, tensors, data):
        assert o == d
        assert len(t) == 2
        assert torch.allclose(t[0], TF.to_tensor(d))
        assert torch.allclose(t[1], TF.to_tensor(d))
