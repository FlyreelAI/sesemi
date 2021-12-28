#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""The trainer CLI's main function and configuration."""
import hydra
import torch
import logging
import pytorch_lightning as pl
import math
import os
import h5py
import yaml

from torch.utils.data import ConcatDataset
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from typing import Callable, List, Optional

from tqdm import tqdm
from PIL import Image

from ..config.structs import SESEMIPseudoDatasetConfig
from ..learners import Classifier

logger = logging.getLogger(__name__)


config_store = ConfigStore.instance()
config_store.store(
    name="pseudo_dataset",
    node=SESEMIPseudoDatasetConfig,
    group="cmd",
    package="_global_",
)


def _num_digits(x: int) -> int:
    return int(math.ceil(math.log10(x)))


@hydra.main(config_path="../trainer/conf", config_name=None)
def pseudo_dataset(config: SESEMIPseudoDatasetConfig):
    """The pseudo dataset generator.

    Args:
        config: The pseudo dataset generation config.
    """
    random_seed = pl.seed_everything(config.seed)

    device = "cpu"
    if config.gpu is not None:
        device = f"cuda:{config.gpu}"

    learner = Classifier.load_from_checkpoint(
        to_absolute_path(config.checkpoint_path), map_location=device
    )
    dataset = ConcatDataset(
        [
            instantiate(
                dataset_cfg,
                root=to_absolute_path(dataset_cfg.root),
                image_transform=None,
            )
            for dataset_cfg in config.datasets
        ]
    )
    if config.preprocessing_transform is not None:
        preprocessing_transform: Optional[Callable] = instantiate(
            config.preprocessing_transform
        )
    else:
        preprocessing_transform = lambda x: x

    if config.test_time_augmentation is not None:
        test_time_augmentation: Optional[
            Callable[[Image.Image], List[Image.Image]]
        ] = instantiate(config.test_time_augmentation)
    else:
        test_time_augmentation = None

    if config.postaugmentation_transform is not None:
        postaugmentation_transform: Optional[Callable] = instantiate(
            config.postaugmentation_transform
        )
    else:
        postaugmentation_transform = lambda x: x

    num_samples = len(dataset)
    num_digits = _num_digits(num_samples)

    learner.eval()
    ids: List[str] = []

    images_dir = os.path.join(config.output_dir, "images")
    predictions_dir = os.path.join(config.output_dir, "predictions")

    os.makedirs(config.output_dir, exist_ok=False)
    os.makedirs(images_dir, exist_ok=False)
    os.makedirs(predictions_dir, exist_ok=False)

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset)):
            item: Image.Image = data[0]
            item_l = [preprocessing_transform(item)]

            augmentations = (
                test_time_augmentation(item_l)
                if test_time_augmentation is not None
                else item_l
            )

            # If list or tuple assume that test-time augmented version.
            # Aggregate by averaging probabilities.
            logits_l = [
                learner(postaugmentation_transform(x)[None].to(device))
                for x in augmentations
            ]
            logits_s = torch.cat(logits_l, dim=0)
            logits = torch.mean(logits_s, dim=0)

            probabilities = torch.softmax(logits, dim=-1)

            data_id = str(i).zfill(num_digits)
            item.save(os.path.join(images_dir, f"{data_id}.jpg"))
            with h5py.File(
                os.path.join(predictions_dir, f"{data_id}.h5"), "w"
            ) as predictions:
                predictions.create_dataset("logits", data=logits.detach().cpu().numpy())
                predictions.create_dataset(
                    "probabilities", data=probabilities.detach().cpu().numpy()
                )

            ids.append(data_id)

    metadata = dict(ids=ids)
    with open(os.path.join(config.output_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)


if __name__ == "__main__":
    pseudo_dataset()
