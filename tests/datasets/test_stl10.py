import pytest

from PIL import Image
from sesemi.datasets import dataset


@pytest.mark.parametrize("subset", ["train", "test", "unlabeled"])
@pytest.mark.parametrize("unlabeled", [True, False])
@pytest.mark.parametrize("random_subset_size", [4000, None, 100, 1000])
@pytest.mark.parametrize("random_subset_seed", [42, None])
def test_stl10(
    tmp_path_factory,
    subset,
    unlabeled,
    random_subset_size,
    random_subset_seed,
):
    data_dir = tmp_path_factory.getbasetemp().joinpath("data/stl10")

    dst = dataset(
        name="stl10",
        root=str(data_dir),
        subset=subset,
        unlabeled=unlabeled,
        random_subset_size=random_subset_size,
        random_subset_seed=random_subset_seed,
    )

    if random_subset_size:
        assert len(dst) == random_subset_size
    elif subset == "train":
        assert len(dst) == 5000
    elif subset == "test":
        assert len(dst) == 8000
    elif subset == "unlabeled":
        assert len(dst) == 97000
    elif subset is None:
        assert len(dst) == 100000

    if unlabeled:
        assert isinstance(dst[0], Image.Image)
    else:
        assert isinstance(dst[0], tuple)
