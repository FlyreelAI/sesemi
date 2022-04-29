import pytest

from sesemi.datasets import dataset


def test_concat():
    dst1 = list(range(100))
    dst2 = list(range(100, 200))

    concat_dataset = dataset(
        name="concat",
        root="",
        datasets=[
            dst1,
            dst2,
        ],
    )

    assert len(concat_dataset) == 200
    assert set(concat_dataset) == set(range(200))
