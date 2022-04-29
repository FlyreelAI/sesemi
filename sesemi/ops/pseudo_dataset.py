#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""A pseudo-labeled dataset generation operation."""
import hydra
import torch
import torch.nn as nn
import logging
import pytorch_lightning as pl
import math
import os
import h5py
import yaml
import multiprocessing as mp

from torch.utils.data import DataLoader, Subset
from collections import defaultdict

from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log, setup_globals

from typing import Any, Callable, ChainMap, Dict, List, Optional, TypeVar

from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, wait

from sesemi.utils import compute_device_names

from ..config.structs import SESEMIPseudoDatasetConfig
from ..learners import Classifier

logger = logging.getLogger(__name__)


config_store = ConfigStore.instance()
config_store.store(
    name="pseudo_dataset",
    node=SESEMIPseudoDatasetConfig,
    group="ops",
    package="_global_",
)


T = TypeVar("T")


def _identity(x: T) -> T:
    """The identity function used as a default transform."""
    return x


def default_test_time_augmentation(x: T) -> List[T]:
    """An identity test-time augmentation."""
    return [x]


def _num_digits(x: int) -> int:
    """Computes the number of digits in a number base 10."""
    return int(math.ceil(math.log10(x)))


def default_image_getter(x) -> Image.Image:
    """The default image getter for a data example.

    Assumes that examples are either an image or a tuple in which the first
    element is an image (standard case).
    """
    if isinstance(x, Image.Image):
        return x
    else:
        return x[0]


def apply_model_to_test_time_augmentations(
    model: Callable[[torch.Tensor], torch.Tensor],
    device: str,
    data: List[List[torch.Tensor]],
    batch_compatible_tensors: bool = True,
) -> List[torch.Tensor]:
    """Applies a model to test-time augmented images.

    Args:
        model: The model to apply.
        device: The device the model is on.
        data: The list of test-time-augmented images. Each
            child list corresponds to test-time augmented images
            from an original input image.
        batch_compatible_tensors: Whether to concatenate compatibly sized
            tensors.

    Returns:
        A list of batches of logits corresponding to the input images.
    """
    if batch_compatible_tensors:
        data_index_by_shape = defaultdict(list)
        for i, items in enumerate(data):
            for j, item in enumerate(items):
                data_index_by_shape[item.shape].append((i, j))

        results = []
        for indices in data_index_by_shape.values():
            tensors = [data[i][j] for i, j in indices]
            tensor_batch = torch.stack(tensors, dim=0)
            results_batch = model(tensor_batch.to(device))
            results.extend(zip(indices, list(results_batch)))
        results.sort()

        outputs = [[] for _ in range(len(data))]
        for (i, j), result in results:
            assert len(outputs[i]) == j
            outputs[i].append(result)

        return [torch.cat(x, dim=0) for x in outputs]
    else:
        return [
            torch.cat([model(tensor[None].to(device)) for tensor in tensors], dim=0)
            for tensors in data
        ]


class TestTimeAugmentationCollator:
    def __init__(
        self,
        preprocessing_transform: Optional[Callable],
        test_time_augmentation: Optional[Callable[[Image.Image], List[Image.Image]]],
        postaugmentation_transform: Optional[Callable],
        image_getter: Optional[Callable],
    ):
        """Initializes the collator.

        Args:
            preprocessing_transform: The preprocessing transform.
            test_time_augmentation: The test-time augmentation that takes an image
                and returns a list of augmented versions of that image.
            postaugmentation_transform: A transform to apply after test-time
                augmentations and which should return a tensor.
            image_getter: The function to extract the source image from
                an example in the dataset.
        """
        self.preprocessing_transform = (
            _identity if preprocessing_transform is None else preprocessing_transform
        )
        self.test_time_augmentation = (
            default_test_time_augmentation
            if test_time_augmentation is None
            else test_time_augmentation
        )
        self.postaugmentation_transform = (
            _identity
            if postaugmentation_transform is None
            else postaugmentation_transform
        )
        self.image_getter = (
            default_image_getter if image_getter is None else image_getter
        )

    def __call__(self, data_batch: List[Any]) -> List[List[torch.Tensor]]:
        """Generates test-time augmented versions of the input images."""
        data_tensors: List[List[torch.Tensor]] = []
        images: List[Image.Image] = []
        for data in data_batch:
            item: Image.Image = self.image_getter(data)
            images.append(item)

            augmentations = self.test_time_augmentation(
                self.preprocessing_transform(item)
            )
            tensors = [self.postaugmentation_transform(x) for x in augmentations]
            data_tensors.append(tensors)
        return images, data_tensors


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
        instantiate(config.image_getter),
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
    num_digits = _num_digits(num_samples)

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


@hydra.main(config_path="./conf", config_name=None)
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
