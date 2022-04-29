#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""PyTorch Lightning data modules."""
import pytorch_lightning as pl
import os.path as osp

from math import ceil
from torch.utils.data.dataset import Dataset
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader, DistributedSampler
from hydra.utils import instantiate, to_absolute_path

from sesemi.utils import copy_config

from .config.structs import DataConfig, DataLoaderConfig, DatasetConfig


class SESEMIDataModule(pl.LightningDataModule):
    """The main SESEMI data module.

    Attributes:
        train (Optional[Dict[str, Dataset]]): An optional dictionary of training datasets.
        val (Optional[Dataset]): An optional validation dataset.
        test (Optional[Dataset]): An optional test dataset.
        train_batch_sizes (Dict[str, int]): The batch size used for each training data loader.
        train_batch_sizes_per_device (Dict[str, int]): The batch size used per GPU for each training
            data loader.
        train_batch_sizes_per_iteration (Dict[str, int]): The batch size per iteration for each
            training data loader.
    """

    def __init__(
        self,
        config: DataConfig,
        strategy: Optional[str],
        num_devices: int,
        data_root: str,
        batch_size_per_device: Optional[int] = None,
        random_seed: int = 0,
    ):
        """Initializes the data module.

        Args:
            config: The data config.
            strategy: The strategy being used.
            num_devices: The number of devices being used per node.
            data_root: The data root to look for datasets with relative paths.
            batch_size_per_device: An optional default batch size per device to use with
                data loaders.
            random_seed: The random seed to initialize DDP data loaders.
        """
        super().__init__()
        self.config = config
        self.strategy = strategy
        self.num_devices = num_devices
        self.data_root = data_root
        self.random_seed = random_seed
        self.batch_size_per_device = batch_size_per_device
        self._build()

    def _build_dataset(self, config: DatasetConfig) -> Dataset:
        if config.root is None:
            dataset_root = to_absolute_path(self.data_root)
        elif osp.isabs(config.root):
            dataset_root = config.root
        else:
            dataset_root = to_absolute_path(osp.join(self.data_root, config.root))

        dataset_config = copy_config(config)
        dataset_config.pop("root")

        return instantiate(dataset_config, root=dataset_root)

    def _build(self):
        ignored_train = self.config.ignored.train or {}
        self.train, self.val, self.test = None, None, None
        self.train_batch_sizes = {}
        self.train_batch_sizes_per_device = {}
        self.train_batch_sizes_per_iteration = {}
        if self.config.train is not None:
            self.train = {
                key: self._build_dataset(value.dataset)
                for key, value in self.config.train.items()
                if not ignored_train.get(key, False)
            }

            for key in self.train:
                value = self.config.train[key]
                if value.batch_size is not None:
                    assert value.batch_size_per_device is None
                    if self.strategy == "dp":
                        batch_size_per_device = ceil(
                            value.batch_size / self.num_devices
                        )
                    else:
                        batch_size_per_device = value.batch_size
                elif value.batch_size_per_device is not None:
                    batch_size_per_device = value.batch_size_per_device
                elif self.batch_size_per_device is not None:
                    batch_size_per_device = self.batch_size_per_device
                else:
                    batch_size_per_device = 1

                self.train_batch_sizes_per_device[key] = batch_size_per_device

            self.train_batch_sizes_per_iteration = {
                key: self.train_batch_sizes_per_device[key] * max(self.num_devices, 1)
                for key in self.train.keys()
            }

            self.train_batch_sizes = {
                key: self.train_batch_sizes_per_device[key]
                if self.strategy == "ddp"
                else self.train_batch_sizes_per_iteration[key]
                for key in self.train.keys()
            }

        if self.config.val is not None:
            self.val = self._build_dataset(self.config.val.dataset)

        if self.config.test is not None:
            self.test = self._build_dataset(self.config.test.dataset)

        if self.config.extra is not None:
            self.extra = {
                k: self._build_dataset(v.dataset) for k, v in self.config.extra.items()
            }

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        assert self.train is not None
        assert self.config.train is not None
        assert self.train_batch_sizes is not None

        train_dataloaders = {}
        for key, dataset in self.train.items():
            dataloader_config = copy_config(self.config.train[key])
            dataloader_config.pop("dataset")
            dataloader_config.pop("batch_size")
            dataloader_config.pop("batch_size_per_device")

            if self.strategy == "ddp":
                dataloader_config.pop("shuffle")
                train_dataloaders[key] = instantiate(
                    dataloader_config,
                    dataset=dataset,
                    batch_size=self.train_batch_sizes[key],
                    sampler=DistributedSampler(
                        dataset,
                        shuffle=self.config.train[key].shuffle,
                        seed=self.random_seed,
                    ),
                )
            else:
                train_dataloaders[key] = instantiate(
                    dataloader_config,
                    dataset=dataset,
                    batch_size=self.train_batch_sizes[key],
                )

        return train_dataloaders

    def _build_evaluation_data_loader(
        self, config: DataLoaderConfig, dataset: Dataset
    ) -> DataLoader:
        dataloader_config = copy_config(config)
        dataloader_config.pop("dataset")
        dataloader_config.pop("batch_size")
        dataloader_config.pop("batch_size_per_device")

        assert (
            config.batch_size is None or config.batch_size_per_device is None
        ), "cannot set both batch_size and batch_size_per_device"

        if self.strategy == "dp":
            if config.batch_size is not None:
                batch_size = config.batch_size
            elif config.batch_size_per_device is not None:
                batch_size = config.batch_size_per_device * self.num_devices
            elif self.batch_size_per_device is not None:
                batch_size = self.batch_size_per_device * self.num_devices
            else:
                batch_size = 1
        else:
            batch_size = (
                config.batch_size
                or config.batch_size_per_device
                or self.batch_size_per_device
                or 1
            )

        return instantiate(dataloader_config, dataset=dataset, batch_size=batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        assert self.val is not None
        assert self.config.val is not None
        return self._build_evaluation_data_loader(self.config.val, self.val)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        assert self.test is not None
        assert self.config.test is not None
        return self._build_evaluation_data_loader(self.config.test, self.test)

    def extra_dataloader(
        self, name: str
    ) -> Optional[Union[DataLoader, List[DataLoader]]]:
        if self.config.extra is None or self.extra is None:
            return None

        ignored_extra = self.config.ignored.extra or {}
        if ignored_extra.get(name, False):
            return None

        if name not in self.config.extra or name not in self.extra:
            return None

        return self._build_evaluation_data_loader(
            self.config.extra[name], self.extra[name]
        )
