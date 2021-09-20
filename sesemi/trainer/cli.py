#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""The trainer CLI's main function and configuration."""
import hydra
import logging
import pytorch_lightning as pl

from typing import Any, Dict
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from math import ceil

from ..config.resolvers import SESEMIConfigAttributes
from ..config.structs import SESEMIBaseConfig, ClassifierConfig, RunMode
from ..datamodules import SESEMIDataModule
from ..utils import compute_num_gpus, load_checkpoint

logger = logging.getLogger(__name__)


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
    random_seed = pl.seed_everything(config.run.seed)

    sesemi_config_attributes = SESEMIConfigAttributes()
    OmegaConf.register_new_resolver("sesemi", sesemi_config_attributes, replace=True)

    # Expose trainer-specific attributes to config
    num_gpus = compute_num_gpus(config.run.gpus)
    sesemi_config_attributes.num_gpus = num_gpus
    sesemi_config_attributes.num_nodes = config.run.num_nodes or 1

    # Build data loaders
    accelerator = None
    if sesemi_config_attributes.num_gpus > 0:
        accelerator = config.run.accelerator or "ddp"
        assert accelerator in {"dp", "ddp"}, f"Unsupport accelerator {accelerator}"

    datamodule = SESEMIDataModule(
        config.data,
        accelerator,
        num_gpus,
        config.run.data_root,
        batch_size_per_gpu=config.run.batch_size_per_gpu,
        random_seed=random_seed,
    )

    # Expose data-specific attributes to config
    sesemi_config_attributes.iterations_per_epoch = None
    sesemi_config_attributes.max_iterations = None
    if config.data.train is not None:
        sesemi_config_attributes.iterations_per_epoch = max(
            len(x) // datamodule.train_batch_sizes_per_iteration[k]
            if config.data.train[k].drop_last
            else ceil(len(x) / datamodule.train_batch_sizes_per_iteration[k])
            for k, x in datamodule.train.items()
        )

        if config.run.num_iterations is not None:
            assert config.run.num_epochs is None
            sesemi_config_attributes.max_iterations = config.run.num_iterations
        elif config.run.num_epochs is not None:
            assert config.run.num_iterations is None
            sesemi_config_attributes.max_iterations = (
                config.run.num_epochs * sesemi_config_attributes.iterations_per_epoch
            )

    learner = instantiate(config.learner, _recursive_=False)

    trainer_config = config.trainer or {}
    callbacks = trainer_config.get("callbacks", [])
    callbacks = [instantiate(c) for c in callbacks]

    trainer_kwargs: Dict[str, Any] = dict(trainer_config)
    trainer_kwargs["callbacks"] = callbacks

    trainer = pl.Trainer(
        **trainer_kwargs,
        max_epochs=config.run.num_epochs,
        max_steps=config.run.num_iterations,
        accelerator=accelerator,
        gpus=config.run.gpus,
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
        trainer.validate(learner, datamodule.val_dataloader())
    else:
        assert config.run.mode == RunMode.TEST
        trainer.validate(learner, datamodule.test_dataloader())
