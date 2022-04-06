#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""A pseudo-labeled dataset generation operation."""
import hydra
import torch
import logging
import pytorch_lightning as pl
import math
import os
import h5py
import yaml
import multiprocessing as mp

from torch.utils.data import DataLoader, Subset

from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log, setup_globals

from typing import Any, ChainMap, Dict

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait

from sesemi.config.structs import SESEMIPseudoDatasetConfig
from sesemi.learners import Classifier
from sesemi.collation import TestTimeAugmentationCollator
from sesemi.tta import apply_model_to_test_time_augmentations
from sesemi.utils import compute_device_names, compute_num_digits

logger = logging.getLogger(__name__)


config_store = ConfigStore.instance()
config_store.store(
    name="pseudo_dataset",
    node=SESEMIPseudoDatasetConfig,
    group="ops",
    package="_global_",
)


def task(
    config: SESEMIPseudoDatasetConfig,
    hydra_config: HydraConfig,
    task_id: int,
    num_tasks: int,
    device: str,
) -> Dict[str, Dict[str, Any]]:
    """A task to generate a subset of the pseudo-labeled dataset.

    Args:
        config: The operation's configuration.
        hydra_config: The global hydra configuration.
        task_id: The assigned task id from 0 to num_tasks-1.
        num_tasks: The number of tasks running.
        device: The device to use for inference.

    Returns:
        A dictionary of detailed metadata for each image by its assigned
        string identifier.
    """
    pl.seed_everything(config.seed)

    setup_globals()
    configure_log(hydra_config.hydra.job_logging, hydra_config.hydra.verbose)
    HydraConfig.instance().set_config(hydra_config)

    learner: Classifier = Classifier.load_from_checkpoint(
        to_absolute_path(config.checkpoint_path), map_location=device
    )
    learner.eval()
    learner.to(device)

    dataset = instantiate(
        config.dataset,
        image_transform=None,
    )

    collator = TestTimeAugmentationCollator(
        instantiate(config.preprocessing_transform),
        instantiate(config.test_time_augmentation),
        instantiate(config.postaugmentation_transform),
    )

    size = len(dataset)
    items_per_worker = math.ceil(size / num_tasks)
    start = items_per_worker * task_id
    end = min(items_per_worker * (task_id + 1), size)
    subset = Subset(dataset, list(range(start, end)))

    dataloader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        batch_sampler=None,
        collate_fn=collator,
    )

    output_dir = to_absolute_path(config.output_dir)
    images_dir = os.path.join(output_dir, "images")
    predictions_dir = os.path.join(output_dir, "predictions")

    num_samples = len(dataset)
    num_digits = compute_num_digits(num_samples)

    if learner.has_ema and config.use_ema:
        model_forward = learner.forward_ema
    else:
        model_forward = learner.forward

    details: Dict[str, Dict[str, Any]] = {}
    relative_index = 0
    with torch.no_grad():
        iterable = tqdm(dataloader) if task_id == 0 else dataloader
        for i, (images, data_tensors) in enumerate(iterable):
            logits = apply_model_to_test_time_augmentations(
                model_forward,
                device,
                data_tensors,
                batch_compatible_tensors=True,
            )
            probabilities = [torch.softmax(l, dim=-1) for l in logits]
            avg_logits = [torch.mean(x, dim=0) for x in logits]
            avg_probabilities = [torch.mean(x, dim=0) for x in probabilities]
            for j, (image, logits_, probs) in enumerate(
                zip(images, avg_logits, avg_probabilities)
            ):
                index = start + relative_index + j
                data_id = str(index).zfill(num_digits)

                src_image_filename = image.info.get("filename")
                dst_image_filename = os.path.join(images_dir, f"{data_id}.jpg")
                if config.symlink_images and src_image_filename is not None:
                    os.symlink(src_image_filename, dst_image_filename)
                else:
                    if config.symlink_images:
                        logger.warning(
                            f"filename metadata for image {data_id} does not exist"
                        )
                    image.save(dst_image_filename)

                with h5py.File(
                    os.path.join(predictions_dir, f"{data_id}.h5"), "w"
                ) as predictions:
                    predictions.create_dataset(
                        "logits", data=logits_.detach().cpu().numpy()
                    )
                    predictions.create_dataset(
                        "probabilities", data=probs.detach().cpu().numpy()
                    )

                details[data_id] = dict(id=data_id, filename=src_image_filename)
            relative_index += len(images)

    return details


@hydra.main(config_path="./conf", config_name="/ops/pseudo_dataset")
def pseudo_dataset(config: SESEMIPseudoDatasetConfig):
    """The pseudo dataset generator.

    Args:
        config: The pseudo dataset generation config.
    """
    mp.set_start_method("spawn")

    output_dir = to_absolute_path(config.output_dir)
    images_dir = os.path.join(output_dir, "images")
    predictions_dir = os.path.join(output_dir, "predictions")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=False)
    os.makedirs(predictions_dir, exist_ok=False)

    devices = compute_device_names(config.gpus)

    process_pool = ProcessPoolExecutor(max_workers=len(devices))

    task_futures = []
    hydra_config = HydraConfig.instance().cfg
    for task_id, device in enumerate(devices):
        task_futures.append(
            process_pool.submit(
                task,
                config=config,
                hydra_config=hydra_config,
                task_id=task_id,
                num_tasks=len(devices),
                device=device,
            )
        )

    done, not_done = wait(task_futures)
    if not_done:
        logger.error(f"Could not run inference successfully on all devices: {not_done}")
        return

    details = dict(ChainMap(*[x.result() for x in done]))
    ids = sorted(list(details.keys()))

    metadata = dict(ids=ids, details=details)
    with open(os.path.join(output_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


if __name__ == "__main__":
    pseudo_dataset()
