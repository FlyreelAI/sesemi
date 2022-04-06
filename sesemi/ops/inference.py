#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""An inference operation."""
import hydra
import torch
import logging
import pytorch_lightning as pl
import math
import os
import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp

from torch import Tensor
from torch.utils.data import DataLoader, Subset

from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log, setup_globals

from typing import Any, ChainMap, Dict, List

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait

from sesemi.config.structs import SESEMIInferenceConfig
from sesemi.learners import Classifier
from sesemi.collation import TestTimeAugmentationCollator
from sesemi.datasets.image_file import ImageFile
from sesemi.tta import apply_model_to_test_time_augmentations
from sesemi.utils import compute_device_names, compute_num_digits

logger = logging.getLogger(__name__)


config_store = ConfigStore.instance()
config_store.store(
    name="inference",
    node=SESEMIInferenceConfig,
    group="ops",
    package="_global_",
)


def task(
    config: SESEMIInferenceConfig,
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

    dataset = ImageFile(to_absolute_path(config.data_dir))

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
            probabilities: List[Tensor] = [torch.softmax(l, dim=-1) for l in logits]
            avg_logits: List[Tensor] = [torch.mean(x, dim=0) for x in logits]
            avg_probabilities: List[Tensor] = [
                torch.mean(x, dim=0) for x in probabilities
            ]
            for j, (image, logits_, probs) in enumerate(
                zip(images, avg_logits, avg_probabilities)
            ):
                index = start + relative_index + j
                data_id = str(index).zfill(num_digits)

                src_image_filename = image.info.get("filename")

                logits_np = logits_.detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy()
                if config.export_predictions:
                    with h5py.File(
                        os.path.join(predictions_dir, f"{data_id}.h5"), "w"
                    ) as predictions:
                        predictions.create_dataset("logits", data=logits_np)
                        predictions.create_dataset("probabilities", data=probs_np)

                label = int(np.argmax(probs_np))
                score = float(probs_np[label])
                details[data_id] = dict(
                    id=data_id, filename=src_image_filename, label=label, score=score
                )
            relative_index += len(images)

    return details


@hydra.main(config_path="./conf", config_name="/ops/inference")
def inference(config: SESEMIInferenceConfig):
    """The pseudo dataset generator.

    Args:
        config: The pseudo dataset generation config.
    """
    mp.set_start_method("spawn")

    output_dir = to_absolute_path(config.output_dir)
    predictions_dir = os.path.join(output_dir, "predictions")

    os.makedirs(output_dir, exist_ok=True)
    if config.export_predictions:
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
    df = pd.DataFrame(details.values(), columns=["id", "filename", "label", "score"])
    df.to_csv(os.path.join(config.output_dir, "labels.csv"), index=False)


if __name__ == "__main__":
    inference()
