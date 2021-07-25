# Copyright 2021, Flyreel. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import os
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
from ..config.structs import SESEMIConfig, ClassifierConfig, RunMode
from ..datamodules import SESEMIDataModule
from ..utils import compute_num_gpus, load_checkpoint

logger = logging.getLogger(__name__)


config_store = ConfigStore.instance()
config_store.store(name="sesemi_config", node=SESEMIConfig)
config_store.store(name="classifier", node=ClassifierConfig, group="learner")


@hydra.main(config_path="conf", config_name="config")
def open_sesemi(config: SESEMIConfig):
    random_seed = pl.seed_everything(config.run.seed)

    sesemi_config_attributes = SESEMIConfigAttributes()
    OmegaConf.register_new_resolver("sesemi", sesemi_config_attributes)

    # Expose config trainer-specific attributes to config
    gpus = config.run.gpus if not config.run.no_cuda else 0
    num_gpus = compute_num_gpus(gpus) if not config.run.no_cuda else 0
    sesemi_config_attributes.num_gpus = num_gpus
    sesemi_config_attributes.num_nodes = config.trainer.get("num_nodes", 1)

    # Build data loaders
    accelerator = None
    if sesemi_config_attributes.num_gpus > 0:
        accelerator = config.run.accelerator

    datamodule = SESEMIDataModule(
        config.data,
        accelerator,
        num_gpus,
        batch_size_per_gpu=config.run.batch_size_per_gpu,
        random_seed=random_seed,
    )

    # Expose config data-specific attributes to config
    sesemi_config_attributes.iterations_per_epoch = None
    sesemi_config_attributes.max_iterations = None
    if config.data.train is not None:
        sesemi_config_attributes.iterations_per_epoch = max(
            ceil(len(x) / datamodule.train_batch_sizes_per_iteration[k])
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

    callbacks = config.trainer.get("callbacks", [])
    callbacks = [instantiate(c) for c in callbacks]

    trainer_kwargs: Dict[str, Any] = dict(config.trainer)
    trainer_kwargs["callbacks"] = callbacks

    trainer = pl.Trainer(
        **trainer_kwargs,
        max_epochs=config.run.num_epochs,
        max_steps=config.run.num_iterations,
        accelerator=accelerator,
        gpus=gpus,
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
