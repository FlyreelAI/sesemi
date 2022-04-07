#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""SESEMI data loaders."""
import torch
import torch.utils.data.graph_settings

from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
    Sequence,
)
from itertools import chain, repeat

from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    IterableDataset,
    IterDataPipe,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data import _utils
from torch.utils.data.dataloader import (
    _DatasetKind,
    _InfiniteConstantSampler,
)

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class _DataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Union[Sampler, Iterable, None] = None,
        batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[T]], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        multiprocessing_context=None,
        generator=None,
        **kwargs
    ):
        major, minor, patch = torch.__version__.split(".")
        major_i, minor_i = int(major), int(minor)

        allowed_kwargs = set()
        if major_i > 1 or (major_i == 1 and minor_i >= 10):
            allowed_kwargs = {"prefetch_factor", "persistent_workers"}

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            **{k: v for k, v in kwargs.items() if k in allowed_kwargs}
        )


def _repeat_iterable(iterable: Iterable, n: int) -> Iterable:
    return chain.from_iterable(repeat(iterable, n))


class _RepeatedIterable:
    def __init__(self, iterable: Iterable, n: int):
        self._iterable = iterable
        self._n = n

    def __len__(self) -> int:
        return len(self._iterable) * self._n

    def __iter__(self) -> Iterator:
        it = iter(_repeat_iterable(self._iterable, self._n))
        return it


class RepeatableDataLoader(_DataLoader):
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Union[Sampler, Iterable, None] = None,
        batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[T]], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        repeat: Optional[int] = None,
        **kwargs
    ):
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )

        if timeout < 0:
            raise ValueError("timeout option should be non-negative")

        if num_workers == 0 and prefetch_factor != 2:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing."
                "let num_workers > 0 to enable multiprocessing."
            )
        assert prefetch_factor > 0

        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        self.repeat = repeat if repeat is not None else 1

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self.dataset = _RepeatedIterable(dataset, self.repeat)

            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and IterableDataset ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if isinstance(dataset, IterDataPipe):
                torch.utils.data.graph_settings.apply_shuffle_settings(
                    dataset, shuffle=shuffle
                )
            elif shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle)
                )

            if sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler)
                )
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(
                        batch_sampler
                    )
                )
        else:
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching "
                    "and is mutually exclusive with drop_last"
                )

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                else:
                    sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

        sampler = _RepeatedIterable(sampler, self.repeat)
        if batch_sampler is not None:
            batch_sampler = _RepeatedIterable(batch_sampler, self.repeat)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = (
            None  # See NOTE [ IterableDataset and __len__ ]
        )

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital("Dataloader", "enabled", "True")  # type: ignore[attr-defined]
