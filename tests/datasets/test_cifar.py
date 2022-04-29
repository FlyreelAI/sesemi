from collections import defaultdict
import pytest

from PIL import Image

from sesemi.datasets.base import dataset


@pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100"])
@pytest.mark.parametrize("subset", ["train", "test"])
@pytest.mark.parametrize("unlabeled", [True, False])
@pytest.mark.parametrize("random_subset_size", [4000, None, 100, 1000])
@pytest.mark.parametrize("random_subset_seed", [42, None])
def test_cifar(
    tmp_path_factory,
    dataset_name,
    subset,
    unlabeled,
    random_subset_size,
    random_subset_seed,
):
    data_dir = tmp_path_factory.getbasetemp().joinpath(f"data/{dataset_name}")

    dst = dataset(
        name=dataset_name,
        root=str(data_dir),
        subset=subset,
        unlabeled=unlabeled,
        random_subset_size=random_subset_size,
        random_subset_seed=random_subset_seed,
    )

    if random_subset_size:
        assert len(dst) == random_subset_size
    elif subset == "train":
        assert len(dst) == 50000
    else:
        assert len(dst) == 10000

    if unlabeled:
        assert isinstance(dst[0], Image.Image)
    else:
        assert isinstance(dst[0], tuple)
