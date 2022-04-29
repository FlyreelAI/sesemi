import pytest

from torch.utils.data import IterableDataset

from sesemi.dataloaders import _DataLoader, RepeatableDataLoader


def test_data_loader():
    dl = _DataLoader(list(range(100)))
    assert len(dl) == 100
    assert len(list(dl)) == 100


@pytest.mark.parametrize("repeat", [1, 2])
def test_repeatable_dataloader_map(repeat):
    x = list(range(100))
    dl = RepeatableDataLoader(x, repeat=repeat)

    assert len(dl) == (100 * repeat)
    assert len(list(dl)) == (100 * repeat)


@pytest.mark.parametrize("repeat", [1, 2])
def test_repeatable_dataloader_iterable(repeat):
    class _Iterable(IterableDataset):
        def __iter__(self):
            yield from list(range(100))

        def __len__(self) -> int:
            return 100

    x = _Iterable()
    dl = RepeatableDataLoader(x, repeat=repeat)

    assert len(dl) == (100 * repeat)
    assert len(list(dl)) == (100 * repeat)
