#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""PyTorch Lightning data modules."""
import pytorch_lightning as pl
import os.path as osp
import logging

from math import ceil
from torch.utils.data.dataset import Dataset
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader, DistributedSampler
from hydra.utils import instantiate, to_absolute_path

from .config.structs import DataConfig, DataLoaderConfig, DatasetConfig

logger = logging.getLogger(__name__)


class SESEMIDataModule(pl.LightningDataModule):
    """The main SESEMI data module.

    Attributes:
        train (Optional[Dict[str, Dataset]]): An optional dictionary of training datasets.
        val (Optional[Dataset]): An optional validation dataset.
        test (Optional[Dataset]): An optional test dataset.
        train_batch_sizes (Dict[str, int]): The batch size used for each training data loader.
        train_batch_sizes_per_gpu (Dict[str, int]): The batch size used per GPU for each training
            data loader.
        train_batch_sizes_per_iteration (Dict[str, int]): The batch size per iteration for each
            training data loader.
    """

    def __init__(
        self,
        config: DataConfig,
        accelerator: Optional[str],
        num_gpus: int,
        data_root: str,
        batch_size_per_gpu: Optional[int] = None,
        random_seed: int = 0,
    ):
        """Initializes the data module.

        Args:
            config: The data config.
            accelerator: The accelerator being used.
            num_gpus: The number of GPUs being used.
            data_root: The data root to look for datasets with relative paths.
            batch_size_per_gpu: An optional default batch size per GPU to use with data loaders.
            random_seed: The random seed to initialize DDP data loaders.
        """
        super().__init__()
        self.config = config
        self.accelerator = accelerator
        self.num_gpus = num_gpus
        self.data_root = data_root
        self.random_seed = random_seed
        self.batch_size_per_gpu = batch_size_per_gpu
        self._build()

    def _build_dataset(self, config: DatasetConfig) -> Dataset:
        if config.root is None:
            dataset_root = to_absolute_path(self.data_root)
        elif osp.isabs(config.root):
            dataset_root = config.root
        else:
            dataset_root = to_absolute_path(osp.join(self.data_root, config.root))

        dataset_kwargs = dict(config)  # type: ignore
        dataset_kwargs.pop("root")

        return instantiate(
            dataset_kwargs,
            root=dataset_root,
        )

    def _build(self):
        self.train, self.val, self.test = None, None, None
        self.train_batch_sizes = {}
        self.train_batch_sizes_per_gpu = {}
        self.train_batch_sizes_per_iteration = {}
        if self.config.train is not None:
            self.train = {
                key: self._build_dataset(value.dataset)
                for key, value in self.config.train.items()
            }

            for key, value in self.config.train.items():
                if value.batch_size is not None:
                    assert value.batch_size_per_gpu is None
                    if self.accelerator == "dp":
                        batch_size_per_gpu = ceil(value.batch_size / self.num_gpus)
                    else:
                        batch_size_per_gpu = value.batch_size
                elif value.batch_size_per_gpu is not None:
                    batch_size_per_gpu = value.batch_size_per_gpu
                elif self.batch_size_per_gpu is not None:
                    batch_size_per_gpu = self.batch_size_per_gpu
                else:
                    batch_size_per_gpu = 1

                self.train_batch_sizes_per_gpu[key] = batch_size_per_gpu

            self.train_batch_sizes_per_iteration = {
                key: self.train_batch_sizes_per_gpu[key] * max(self.num_gpus, 1)
                for key in self.config.train.keys()
            }

            self.train_batch_sizes = {
                key: self.train_batch_sizes_per_gpu[key]
                if self.accelerator == "ddp"
                else self.train_batch_sizes_per_iteration[key]
                for key in self.config.train.keys()
            }

        if self.config.val is not None:
            self.val = self._build_dataset(self.config.val.dataset)

        if self.config.test is not None:
            self.test = self._build_dataset(self.config.test.dataset)

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        assert self.train is not None
        assert self.config.train is not None
        assert self.train_batch_sizes is not None

        train_dataloaders = {}
        for key, dataset in self.train.items():
            dataloader_kwargs = dict(self.config.train[key])  # type: ignore
            dataloader_kwargs.pop("dataset")
            dataloader_kwargs.pop("batch_size")
            dataloader_kwargs.pop("batch_size_per_gpu")

            if self.accelerator == "ddp":
                dataloader_kwargs.pop("shuffle")
                train_dataloaders[key] = instantiate(
                    dataloader_kwargs,
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
                    dataloader_kwargs,
                    dataset=dataset,
                    batch_size=self.train_batch_sizes[key],
                )

        return train_dataloaders

    def _build_evaluation_data_loader(
        self, config: DataLoaderConfig, dataset: Dataset
    ) -> DataLoader:
        dataloader_kwargs = dict(config)  # type: ignore
        dataloader_kwargs.pop("dataset")
        dataloader_kwargs.pop("batch_size")
        dataloader_kwargs.pop("batch_size_per_gpu")

        assert (
            config.batch_size is None or config.batch_size_per_gpu is None
        ), "cannot set both batch_size and batch_size_per_gpu"

        if self.accelerator == "dp":
            if config.batch_size is not None:
                batch_size = config.batch_size
            elif config.batch_size_per_gpu is not None:
                batch_size = config.batch_size_per_gpu * self.num_gpus
            elif self.batch_size_per_gpu is not None:
                batch_size = self.batch_size_per_gpu * self.num_gpus
            else:
                batch_size = 1
        else:
            batch_size = (
                config.batch_size
                or config.batch_size_per_gpu
                or self.batch_size_per_gpu
                or 1
            )

        return instantiate(dataloader_kwargs, dataset=dataset, batch_size=batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        assert self.val is not None
        assert self.config.val is not None
        return self._build_evaluation_data_loader(self.config.val, self.val)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        assert self.test is not None
        assert self.config.test is not None
        return self._build_evaluation_data_loader(self.config.test, self.test)
