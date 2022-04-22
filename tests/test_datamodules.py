from omegaconf import DictConfig, OmegaConf
import pytest
import torch
import os
import numpy as np

from sesemi.config.structs import DataConfig, DataLoaderConfig, DatasetConfig
from sesemi.datamodules import SESEMIDataModule

from PIL import Image


@pytest.mark.parametrize("strategy", ["dp", "ddp"])
@pytest.mark.parametrize("num_devices", [1, 2])
def test_sesemi_datamodule(strategy, num_devices, tmp_path, mocker):
    mocker.patch(
        "sesemi.datamodules.DistributedSampler",
        side_effect=lambda *args, **kwargs: None,
    )

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    num_sample_images = 40

    sample = Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8))
    for i in range(num_sample_images):
        sample.save(os.path.join(str(image_dir), f"{i}.jpg"))

    datamodule = SESEMIDataModule(
        config=DataConfig(
            train=dict(
                a=OmegaConf.create(
                    DataLoaderConfig(
                        dataset=DatasetConfig(
                            DictConfig(
                                dict(
                                    _target_="sesemi.dataset",
                                    name="image_file",
                                    root=str(tmp_path),
                                    subset=None,
                                    image_transform=None,
                                )
                            )
                        ),
                        batch_size=2,
                    )
                ),
                b=OmegaConf.create(
                    DataLoaderConfig(
                        dataset=DictConfig(
                            dict(
                                _target_="sesemi.dataset",
                                name="image_file",
                                root=str(tmp_path),
                                subset=None,
                                image_transform=None,
                            )
                        ),
                        batch_size_per_device=4,
                    )
                ),
                c=OmegaConf.create(
                    DataLoaderConfig(
                        dataset=DictConfig(
                            dict(
                                _target_="sesemi.dataset",
                                name="image_file",
                                root=str(tmp_path),
                                subset=None,
                                image_transform=None,
                            )
                        ),
                        batch_size=num_devices,
                    )
                ),
            ),
            val=OmegaConf.create(
                DataLoaderConfig(
                    dataset=DatasetConfig(
                        DictConfig(
                            dict(
                                _target_="sesemi.dataset",
                                name="image_file",
                                root=str(tmp_path),
                                subset=None,
                                image_transform=None,
                            )
                        )
                    ),
                    batch_size=num_devices,
                ),
            ),
            test=OmegaConf.create(
                DataLoaderConfig(
                    dataset=DatasetConfig(
                        DictConfig(
                            dict(
                                _target_="sesemi.dataset",
                                name="image_file",
                                root=str(tmp_path),
                                subset=None,
                                image_transform=None,
                            )
                        )
                    ),
                    batch_size=num_devices,
                ),
            ),
            extra=dict(
                a=OmegaConf.create(
                    DataLoaderConfig(
                        dataset=DatasetConfig(
                            DictConfig(
                                dict(
                                    _target_="sesemi.dataset",
                                    name="image_file",
                                    root=str(tmp_path),
                                    subset=None,
                                    image_transform=None,
                                )
                            )
                        ),
                        batch_size=2,
                    )
                ),
                b=OmegaConf.create(
                    DataLoaderConfig(
                        dataset=DictConfig(
                            dict(
                                _target_="sesemi.dataset",
                                name="image_file",
                                root=str(tmp_path),
                                subset=None,
                                image_transform=None,
                            )
                        ),
                        batch_size_per_device=4,
                    )
                ),
                c=OmegaConf.create(
                    DataLoaderConfig(
                        dataset=DictConfig(
                            dict(
                                _target_="sesemi.dataset",
                                name="image_file",
                                root=str(tmp_path),
                                subset=None,
                                image_transform=None,
                            )
                        ),
                        batch_size=num_devices,
                    )
                ),
            ),
        ),
        strategy=strategy,
        num_devices=num_devices,
        data_root="",
    )

    train_dataloaders = datamodule.train_dataloader()

    assert set(train_dataloaders.keys()) == {"a", "b", "c"}
    if strategy == "dp":
        assert datamodule.train_batch_sizes["a"] == 2
        assert datamodule.train_batch_sizes["b"] == (num_devices * 4)
        assert datamodule.train_batch_sizes["c"] == num_devices
    else:
        assert datamodule.train_batch_sizes["a"] == 2
        assert datamodule.train_batch_sizes["b"] == 4
        assert datamodule.train_batch_sizes["c"] == num_devices

    val_dataloader = datamodule.val_dataloader()
    assert val_dataloader is not None

    test_dataloader = datamodule.val_dataloader()
    assert test_dataloader is not None

    extra_a = datamodule.extra_dataloader("a")
    extra_b = datamodule.extra_dataloader("b")
    extra_c = datamodule.extra_dataloader("c")
    extra_d = datamodule.extra_dataloader("d")

    assert extra_a is not None
    assert extra_b is not None
    assert extra_c is not None
    assert extra_d is None
