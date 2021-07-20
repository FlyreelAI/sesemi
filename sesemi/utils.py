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
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import os, errno
import torch
import logging
from torch.tensor import Tensor
import torchvision.transforms.functional as TF

from itertools import combinations
from torch import nn

logger = logging.getLogger(__name__)


def reduce_tensor(tensor: Tensor, reduction: Optional[str] = None) -> Tensor:
    if reduction == "mean":
        return torch.mean(tensor)
    elif reduction == "sum":
        return torch.sum(tensor)
    elif reduction is None or reduction == "none":
        return tensor
    else:
        raise ValueError(f"unsupported reduction method {reduction}")


def compute_num_gpus(gpus: Union[int, str, List[int]]) -> int:
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
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_iters == 0:
        return 1.0
    else:
        current = np.clip(curr_iter, 0.0, rampup_iters)
        phase = 1.0 - current / rampup_iters
        return float(np.exp(-5.0 * phase * phase))


class GammaCorrection:
    def __init__(self, r: Tuple[float, float] = (0.5, 2.0)):
        self.gamma_range = r

    def __call__(self, x: float) -> float:
        gamma = np.random.uniform(*self.gamma_range)
        return TF.adjust_gamma(x, gamma, gain=1)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(r={})".format(self.gamma_range)


def adjust_polynomial_lr(
    optimizer, curr_iter, *, warmup_iters, warmup_lr, lr, lr_pow, max_iters
):
    """Decay learning rate according to polynomial schedule with warmup"""
    if curr_iter < warmup_iters:
        frac = curr_iter / warmup_iters
        step = lr - warmup_lr
        running_lr = warmup_lr + step * frac
    else:
        frac = (float(curr_iter) - warmup_iters) / (max_iters - warmup_iters)
        scale_running_lr = max((1.0 - frac), 0.0) ** lr_pow
        running_lr = lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = running_lr

    return running_lr


def assert_same_classes(datasets: List[Any]):
    if len(datasets) == 1:
        return True
    same_classes = [
        x.class_to_idx == y.class_to_idx for x, y in combinations(datasets, r=2)
    ]
    assert all(
        same_classes
    ), f"The following have mismatched subdirectory names. Check the `Root location`.\n{datasets}"


def validate_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    logger.info(f"Loading {checkpoint_path}")
    logger.info("")
    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f)

    pretrained_state_dict = checkpoint["state_dict"]
    pretrained_state_dict.pop("best_validation_top1_accuracy", None)

    current_state_dict = model.state_dict()
    # if "fc_unlabeled.weight" in pretrained_state_dict:
    #     if "fc_unlabeled.weight" not in current_state_dict or (
    #         pretrained_state_dict["fc_unlabeled.weight"].shape
    #         != current_state_dict["fc_unlabeled.weight"].shape
    #     ):
    #         pretrained_state_dict.pop("fc_unlabeled.weight")
    #         pretrained_state_dict.pop("fc_unlabeled.bias")

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
