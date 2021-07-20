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
from typing import Any, Dict
import hydra
import logging
import pytorch_lightning as pl

from omegaconf import OmegaConf
from hydra.utils import instantiate
from sesemi.learners import Classifier
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from math import ceil
from torch.utils.data import DistributedSampler

from ..config.resolvers import SESEMIConfigAttributes
from ..config.structs import SESEMIConfig, ClassifierConfig, RunMode
from ..utils import compute_num_gpus, load_checkpoint

logger = logging.getLogger(__name__)


config_store = ConfigStore.instance()
config_store.store(name="sesemi_config", node=SESEMIConfig)
config_store.store(name="classifier", node=ClassifierConfig, group="learner")


@hydra.main(config_path="conf", config_name="config")
def open_sesemi(config: SESEMIConfig):
    random_seed = pl.seed_everything(config.run.seed)

    run_dir = os.path.join(config.run.log_dir, config.run.id)
    os.makedirs(run_dir, exist_ok=True)

    sesemi_config_attributes = SESEMIConfigAttributes()
    OmegaConf.register_new_resolver("sesemi", sesemi_config_attributes)

    # Expose config trainer-specific attributes to config
    gpus = config.trainer.get("gpus", None)
    sesemi_config_attributes.num_gpus = (
        compute_num_gpus(gpus) if not config.run.no_cuda else 0
    )
    sesemi_config_attributes.num_nodes = config.trainer.get("nodes", 1)

    # Build data loaders
    train_dataloaders = {
        key: instantiate(value) for key, value in config.data.train.items()
    }

    val_dataloaders = instantiate(config.data.val)

    # Expose config data-specific attributes to config
    sesemi_config_attributes.iterations_per_epoch = max(
        len(x) for x in train_dataloaders.values()
    )

    if sesemi_config_attributes.num_gpus > 0:
        sesemi_config_attributes.iterations_per_epoch = ceil(
            sesemi_config_attributes.iterations_per_epoch
            / sesemi_config_attributes.num_gpus
        )

    sesemi_config_attributes.max_iterations = (
        config.run.num_epochs * sesemi_config_attributes.iterations_per_epoch
    )

    learner = Classifier(config.learner)

    callbacks = config.trainer.get("callbacks", [])
    callbacks = [instantiate(c) for c in callbacks]

    trainer_kwargs: Dict[str, Any] = dict(config.trainer)
    trainer_kwargs["callbacks"] = callbacks

    trainer = pl.Trainer(
        **trainer_kwargs,
        max_epochs=config.run.num_epochs,
        default_root_dir=run_dir,
        resume_from_checkpoint=to_absolute_path(config.run.resume_from_checkpoint)
        if config.run.resume_from_checkpoint
        else None,
        replace_sampler_ddp=True,
    )

    if not config.run.resume_from_checkpoint and config.run.pretrained_checkpoint_path:
        load_checkpoint(
            learner, to_absolute_path(config.run.pretrained_checkpoint_path)
        )

    if config.run.mode == RunMode.EVALUATE_ONLY:
        # Evaluate model on validation set and exit
        trainer.validate(learner, val_dataloaders)
        return

    trainer.fit(learner, train_dataloaders, val_dataloaders)
