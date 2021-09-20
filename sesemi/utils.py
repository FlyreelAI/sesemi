#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Utility functions."""
import numpy as np
import os, errno
import torch
import logging

from typing import Any, List, Optional, Union
from torch import Tensor
from torchvision.datasets import ImageFolder

from itertools import combinations
from torch import nn

logger = logging.getLogger(__name__)


def reduce_tensor(tensor: Tensor, reduction: Optional[str] = None) -> Tensor:
    """Reduces a tensor using the given mode.

    Args:
        reduction: An optional method to use when reducing the tensor. Can be one of
            (mean, sum, none).
    """
    if reduction == "mean":
        return torch.mean(tensor)
    elif reduction == "sum":
        return torch.sum(tensor)
    elif reduction is None or reduction == "none":
        return tensor
    else:
        raise ValueError(f"unsupported reduction method {reduction}")


def compute_num_gpus(gpus: Union[int, str, List[int]]) -> int:
    """Computes the number of GPUs being used.

    Args:
        gpus: Either an integer specifying the number of GPUs to use, a list of GPU
            integer IDs, a comma-separated list of GPU IDs, or None to train on the CPU. Setting
            this to -1 uses all GPUs and setting it to 0 also uses the CPU.

    Returns:
        The number of GPUs that will be used.
    """
    num_gpus = 0
    if gpus is not None:
        num_available_gpus = torch.cuda.device_count()
        if isinstance(gpus, int):
            num_gpus = gpus if gpus >= 0 else num_available_gpus
        elif isinstance(gpus, str):
            gpu_ids = [x.strip() for x in gpus.split(",")]
            if len(gpu_ids) > 1:
                num_gpus = len(gpu_ids)
            elif len(gpu_ids) == 1:
                if int(gpu_ids[0]) < 0:
                    num_gpus = num_available_gpus
                else:
                    num_gpus = 1
        else:
            num_gpus = len(gpus)
    return num_gpus


def sigmoid_rampup(curr_iter: int, rampup_iters: int) -> float:
    """Computes the sigmoid ramp-up value.

    Exponential rampup from https://arxiv.org/abs/1610.02242

    Args:
        curr_iter: The current iteration (step).
        rampup_iters: The number of iterations to ramp-up.

    Returns:
        An increasing value between 0 and 1.
    """
    if rampup_iters == 0:
        return 1.0
    else:
        current = np.clip(curr_iter, 0.0, rampup_iters)
        phase = 1.0 - current / rampup_iters
        return float(np.exp(-5.0 * phase * phase))


def assert_same_classes(datasets: List[ImageFolder]):
    """Checks that the image folder datasets have the same classes.

    Args:
        datasets: The datasets to ensure have the same classes.
    """
    if len(datasets) == 1:
        return True
    same_classes = [
        x.class_to_idx == y.class_to_idx for x, y in combinations(datasets, r=2)
    ]
    assert all(
        same_classes
    ), f"The following have mismatched subdirectory names. Check the `Root location`.\n{datasets}"


def validate_paths(paths: List[str]):
    """Validates that the paths exist.

    Args:
        paths: The list of filesystem paths.
    """
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    """Loads the classifier checkpoint.

    Args:
        model: The classifier.
        checkpoint_path: The path to the classifier's checkpoint.
    """
    logger.info(f"Loading {checkpoint_path}")
    logger.info("")
    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f)

    pretrained_state_dict = checkpoint["state_dict"]
    pretrained_state_dict.pop("best_validation_top1_accuracy", None)

    current_state_dict = model.state_dict()
    if "fc.weight" in pretrained_state_dict:
        if "fc.weight" not in current_state_dict or (
            pretrained_state_dict["fc.weight"].shape
            != current_state_dict["fc.weight"].shape
        ):
            pretrained_state_dict.pop("fc.weight")
            pretrained_state_dict.pop("fc.bias")

    incompatible_keys = model.load_state_dict(pretrained_state_dict, strict=False)
    if incompatible_keys.missing_keys:
        logger.info("missing keys:")
        logger.info("---")
        logger.info("\n".join(incompatible_keys.missing_keys))
        logger.info("")

    if incompatible_keys.unexpected_keys:
        logger.info("unexpected keys:")
        logger.info("---")
        logger.info("\n".join(incompatible_keys.unexpected_keys))
        logger.info("")
