import pytest
import os
import os.path as osp
import numpy as np

from PIL import Image
from sesemi.datasets import dataset


def test_image_file(tmp_path):
    images_per_subset = dict(train=100, val=20, test=10)
    total_num_images = sum(images_per_subset.values())
    for subset, num_images in images_per_subset.items():
        subset_dir = tmp_path / subset
        subset_dir.mkdir()
        for i in range(num_images):
            image = Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8))
            image.save(str(subset_dir / f"{i}.jpg"))

    entire_dataset = dataset(
        name="image_file",
        root=str(tmp_path),
        subset=None,
    )

    for x in entire_dataset:
        assert isinstance(x, Image.Image)
        assert x.info.get("filename") is not None
        assert osp.exists(x.info["filename"])

    assert len(entire_dataset) == total_num_images
    for subset, num_images in images_per_subset.items():
        data_subset = dataset(
            name="image_file",
            root=str(tmp_path),
            subset=subset,
        )

        assert len(data_subset) == num_images
