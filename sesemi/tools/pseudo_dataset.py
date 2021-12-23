#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""The trainer CLI's main function and configuration."""
import hydra
import logging
import pytorch_lightning as pl

from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from ..config.structs import (
    ClassifierConfig,
    SESEMIBaseConfig,
    SESEMIPseudoDatasetConfig,
)
from ..utils import load_checkpoint

logger = logging.getLogger(__name__)


config_store = ConfigStore.instance()
config_store.store(
    name="sesemi", node=SESEMIBaseConfig, group="base", package="_global_"
)
config_store.store(name="classifier", node=ClassifierConfig, group="learner")


@hydra.main(config_path="../trainer/conf", config_name=None)
def pseudo_dataset(config: SESEMIPseudoDatasetConfig):
    """The pseudo dataset generator.

    Args:
        config: The pseudo dataset generation config.
    """
    random_seed = pl.seed_everything(config.run.seed)
    learner = instantiate(config.learner, _recursive_=False)
    load_checkpoint(
        learner, to_absolute_path(config.pseudo_dataset.checkpoint_path), strict=True
    )


if __name__ == "__main__":
    pseudo_dataset()
