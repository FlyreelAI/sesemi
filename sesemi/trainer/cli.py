#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""The trainer CLI's main function and configuration."""
import hydra
import pytorch_lightning as pl

from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from math import ceil

from ..config.resolvers import SESEMIConfigAttributes
from ..config.structs import SESEMIBaseConfig, ClassifierConfig, RunMode
from ..datamodules import SESEMIDataModule
from ..utils import compute_num_devices, copy_config, load_checkpoint, has_length


config_store = ConfigStore.instance()
config_store.store(
    name="sesemi", node=SESEMIBaseConfig, group="base", package="_global_"
)
config_store.store(name="classifier", node=ClassifierConfig, group="learner")


@hydra.main(config_path="conf", config_name="/base/config")
def open_sesemi(config: SESEMIBaseConfig):
    """The trainer's main function.

    Args:
        config: The trainer config.
    """
    random_seed = pl.seed_everything(config.run.seed, workers=True)

    sesemi_config_attributes = SESEMIConfigAttributes()
    OmegaConf.register_new_resolver("sesemi", sesemi_config_attributes, replace=True)

    # Expose trainer-specific attributes to config
    num_devices = compute_num_devices(config.run.accelerator, config.run.devices)
    sesemi_config_attributes.num_devices = num_devices
    sesemi_config_attributes.num_nodes = config.run.num_nodes or 1

    # Build data loaders
    strategy = None
    if sesemi_config_attributes.num_devices > 0:
        strategy = config.run.strategy or "ddp"
        assert strategy in {"dp", "ddp"}, f"Unsupport strategy {strategy}"

    datamodule = SESEMIDataModule(
        config.data,
        strategy,
        num_devices,
        config.run.data_root,
        batch_size_per_device=config.run.batch_size_per_device,
        random_seed=random_seed,
    )

    # Expose data-specific attributes to config
    sesemi_config_attributes.iterations_per_epoch = None
    sesemi_config_attributes.max_iterations = None
    if config.data.train is not None:
        try:
            sesemi_config_attributes.iterations_per_epoch = max(
                (len(x) * (config.data.train[k].get("repeat") or 1))
                // datamodule.train_batch_sizes_per_iteration[k]
                if config.data.train[k].drop_last
                else ceil(
                    (len(x) * (config.data.train[k].get("repeat") or 1))
                    / datamodule.train_batch_sizes_per_iteration[k]
                )
                for k, x in datamodule.train.items()
                if has_length(x)
            )
        except ValueError:
            assert (
                config.run.num_iterations is not None
            ), "must use num_iterations if all training datasets are iterable"
            sesemi_config_attributes.iterations_per_epoch = config.run.num_iterations

        if config.run.num_iterations is not None:
            assert config.run.num_epochs is None
            sesemi_config_attributes.max_iterations = config.run.num_iterations
        elif config.run.num_epochs is not None:
            assert config.run.num_iterations is None
            sesemi_config_attributes.max_iterations = (
                config.run.num_epochs * sesemi_config_attributes.iterations_per_epoch
            )

    learner = instantiate(config.learner, sesemi_config=config, _recursive_=False)

    trainer_config = copy_config(config.trainer) if config.trainer is not None else {}
    trainer_config["callbacks"] = [
        instantiate(c) for c in trainer_config.get("callbacks", [])
    ]

    trainer = pl.Trainer(
        **trainer_config,
        max_epochs=config.run.num_epochs,
        max_steps=config.run.num_iterations,
        strategy=strategy,
        accelerator=config.run.accelerator,
        devices=config.run.devices,
        num_nodes=config.run.num_nodes,
        resume_from_checkpoint=to_absolute_path(config.run.resume_from_checkpoint)
        if config.run.resume_from_checkpoint
        else None,
        replace_sampler_ddp=False,
    )

    if not config.run.resume_from_checkpoint and config.run.pretrained_checkpoint_path:
        load_checkpoint(
            learner, to_absolute_path(config.run.pretrained_checkpoint_path)
        )

    if config.run.mode == RunMode.FIT:
        trainer.fit(learner, datamodule)
    elif config.run.mode == RunMode.VALIDATE:
        trainer.validate(learner, datamodule=datamodule)
    else:
        assert config.run.mode == RunMode.TEST
        trainer.test(learner, datamodule=datamodule)
