#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Utility functions."""
import numpy as np
import os
import errno
import copy
import torch
import math
import logging

from typing import Any, Dict, Iterable, List, Optional, Union
from torchvision.datasets import ImageFolder

from omegaconf.base import SCMode
from omegaconf.omegaconf import OmegaConf

from itertools import combinations
from torch import nn
from torch import Tensor

from omegaconf import DictConfig

from pytorch_lightning.accelerators.registry import AcceleratorRegistry

logger = logging.getLogger(__name__)


def reduce_tensor(tensor: Tensor, weights: Optional[Tensor] = None, reduction: Optional[str] = None) -> Tensor:
    """Reduces a tensor using the given mode.

    Args:
        reduction: An optional method to use when reducing the tensor. Can be one of
            (mean, weighted_mean, sum, none).
    """
    tensor = tensor if weights is None else tensor * weights
    if reduction == "mean":
        return torch.mean(tensor)
    elif reduction == "weighted_mean":
        return torch.mean(tensor) if weights is None else torch.sum(tensor) / (torch.sum(weights) + 1e-8)
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


def compute_num_devices(accelerator: str, devices: Optional[int]) -> int:
    """Computes the number of devices to use for an accelerator.

    Args:
        accelerator: The accelerator reference string.
        devices: The number of devices to use of None for auto selection.

    Returns:
        The number of devices to use.
    """
    accelerator_cls = AcceleratorRegistry.get(accelerator)
    if devices is None:
        return accelerator_cls.auto_device_count()
        
    accelerator_devices = accelerator_cls.parse_devices(devices)
    if isinstance(accelerator_devices, (list, tuple)):
        return len(accelerator_devices)
    else:
        assert isinstance(accelerator_devices, int)
        return accelerator_devices


def compute_device_names(gpus: Union[int, str, List[int]]) -> List[str]:
    """Computes the device names for the given GPU config.

    Args:
        gpus: Either an integer specifying the number of GPUs to use, a list of GPU
            integer IDs, a comma-separated list of GPU IDs, or None to train on the CPU. Setting
            this to -1 uses all GPUs and setting it to 0 also uses the CPU.

    Returns:
        The torch devices that will be used.
    """
    if gpus is not None:
        num_available_gpus = torch.cuda.device_count()
        if isinstance(gpus, int):
            num_gpus = gpus if gpus >= 0 else num_available_gpus
            return [f"cuda:{i}" for i in range(num_gpus)]
        elif isinstance(gpus, str):
            gpu_ids = [x.strip() for x in gpus.split(",")]
            if len(gpu_ids) > 1:
                return [f"cuda:{x}" for x in gpu_ids]
            elif len(gpu_ids) == 1:
                if int(gpu_ids[0]) < 0:
                    return [f"cuda:{i}" for i in range(num_available_gpus)]
                else:
                    return [f"cuda:{gpu_ids[0]}"]
        else:
            return [f"cuda:{x}" for x in gpus]
    return ["cpu"]


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


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = False):
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

    if not strict:
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
    else:
        model.load_state_dict(pretrained_state_dict, strict=True)


def copy_config(
    config: DictConfig, structured_config_mode: SCMode = SCMode.DICT
) -> Union[Dict[str, Any], DictConfig]:
    """Copies an omegaconf configuration by resolving interpolatons.

    Args:
        config: The config to copy.
        structured_config_mode: The format to return.

    Returns:
        The copied config.
    """
    return OmegaConf.to_container(
        config, resolve=True, structured_config_mode=structured_config_mode
    )


def ema_update(
    ema_module: nn.Module,
    module: nn.Module,
    decay: float,
    method: str = "states",
    copy_non_floating_point: bool = True,
):
    """Computes in-place the EMA parameters from the original parameters."""
    if method == "parameters":
        for ema_param, param in zip(ema_module.parameters(), module.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1.0 - decay))
    elif method == "states":
        for ema_state, state in zip(
            ema_module.state_dict().values(), module.state_dict().values()
        ):
            if ema_state.dtype.is_floating_point:
                ema_state.data.mul_(decay).add_(state.data, alpha=(1.0 - decay))
            elif copy_non_floating_point:
                ema_state.data.copy_(state.data)
    else:
        raise ValueError(f"invalid ema update method {method}")


def copy_and_detach(module: Optional[nn.Module]) -> Optional[nn.Module]:
    """Detaches a new module from the computational graph after copying."""
    if module is None:
        return None

    new_module = copy.deepcopy(module)
    for param in new_module.parameters():
        param.detach_()
    return new_module


def freeze_module(module: torch.nn.Module):
    """Freezes the module."""
    for m in module.modules():
        m.eval()
        for param in m.parameters():
            param.requires_grad = False


def has_length(x: Iterable) -> bool:
    """Checks if the iterable has a length."""
    try:
        return type(len(x)) is int
    except Exception:
        return False


def get_distributed_rank() -> Optional[int]:
    """Returns the distributed rank or None if not distributed."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return None


def compute_num_digits(x: int) -> int:
    """Computes the number of digits in a non-negative number base 10."""
    assert x >= 0, "input must be non-negative"
    if x == 0:
        return 1
    return int(math.ceil(math.log10(x)))