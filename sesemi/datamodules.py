import pytorch_lightning as pl
import os.path as osp

from math import ceil

from torch.utils.data.dataset import Dataset
from sesemi.config.structs import DataConfig, DataLoaderConfig, DatasetConfig
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader, DistributedSampler
from hydra.utils import instantiate, to_absolute_path


class SESEMIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
        accelerator: Optional[str],
        num_gpus: int,
        data_root: str = ".",
        batch_size_per_gpu: Optional[int] = None,
        random_seed: int = 0,
    ):
        super().__init__()
        self.config = config
        self.accelerator = accelerator
        self.num_gpus = num_gpus
        self.data_root = data_root
        self.random_seed = random_seed
        self.batch_size_per_gpu = batch_size_per_gpu
        self._build()

    def _build_dataset(self, config: DatasetConfig) -> Dataset:
        dataset_root = config.root
        if not osp.isabs(dataset_root):
            dataset_root = to_absolute_path(osp.join(self.data_root, dataset_root))

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
                key: self.train_batch_sizes_per_gpu[key] * self.num_gpus
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

        if self.accelerator == "dp":
            batch_size = config.batch_size or (
                self.batch_size_per_gpu * self.num_gpus
                if self.batch_size_per_gpu is not None
                else 1
            )
        else:
            batch_size = config.batch_size or self.batch_size_per_gpu or 1

        return instantiate(dataloader_kwargs, dataset=dataset, batch_size=batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        assert self.val is not None
        assert self.config.val is not None
        return self._build_evaluation_data_loader(self.config.val, self.val)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        assert self.test is not None
        assert self.config.test is not None
        return self._build_evaluation_data_loader(self.config.test, self.test)
