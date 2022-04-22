import pytest
import os.path as osp
import numpy as np

from PIL import Image
from sesemi.datasets import dataset


@pytest.mark.parametrize("num_classes", [1, 5, 10])
def test_image_folder(tmp_path, num_classes):
    images_per_subset = dict(train=100, val=20, test=10)
    total_num_images_per_class = sum(images_per_subset.values())
    for subset, num_images_per_class in images_per_subset.items():
        subset_dir = tmp_path / subset
        subset_dir.mkdir()
        for c in range(num_classes):
            image_dir = subset_dir / str(c)
            image_dir.mkdir()
            for i in range(num_images_per_class):
                image = Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8))
                image.save(str(image_dir / f"{i}.jpg"))

    entire_dataset = dataset(
        name="image_folder",
        root=str(tmp_path),
        subset=None,
    )

    for x in entire_dataset:
        assert isinstance(x, tuple)
        assert isinstance(x[0], Image.Image)
        assert x[0].info.get("filename") is not None
        assert osp.exists(x[0].info["filename"])

    assert len(entire_dataset) == (total_num_images_per_class * num_classes)
    for subset, num_images in images_per_subset.items():
        data_subset = dataset(
            name="image_folder",
            root=str(tmp_path),
            subset=subset,
        )

        assert len(data_subset) == (num_images * num_classes)
