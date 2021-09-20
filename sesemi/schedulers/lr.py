#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Learning rate schedulers."""
from typing import Optional
import warnings
from torch.optim import Optimizer

from torch.optim.lr_scheduler import _LRScheduler

import logging

logger = logging.getLogger(__name__)


class PolynomialLR(_LRScheduler):
    """A polynomial learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_lr: float,
        lr_pow: float,
        max_iters: int,
        warmup_iters: Optional[int] = None,
        warmup_epochs: Optional[int] = None,
        iters_per_epoch: Optional[int] = None,
        last_epoch: int = -1,
    ):
        """Initializes the polynomial learning rate scheduler.

        Args:
            optimizer: The optimizer.
            warmup_lr: The initial warmup learning rate.
            lr_pow: The power of the polynomial update rule.
            max_iters: The maximum number of updates.
            warmup_iters: The number of iterations to warmup. Mutually exclusive with
                `warmup_epochs`.
            warmup_epochs: The number of warmup epochs. Mutually exclusive with `warmup_iters`.
            iters_per_epoch: The number of iterations per epoch. Must be set if `warmup_epochs` is
                set.
            last_epoch: The index of the last epoch.
        """
        if warmup_iters is None:
            assert (
                warmup_epochs is not None and iters_per_epoch is not None
            ), "must specify either warmup_iters or warmup_epochs and iters_per_epoch"

            self.iters_per_epoch = iters_per_epoch
            self.warmup_iters = warmup_epochs * iters_per_epoch
        else:
            self.warmup_iters = warmup_iters

        logger.info(f"Using warmup iterations of {self.warmup_iters}")

        self.warmup_lr = warmup_lr
        self.lr_pow = lr_pow
        self.max_iters = max_iters
        assert self.max_iters >= self.warmup_iters
        super().__init__(optimizer, last_epoch)  # type: ignore

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch >= self.max_iters:
            return [0.0 for _ in self.optimizer.param_groups]
        else:
            if self.last_epoch < self.warmup_iters:
                frac = self.last_epoch / self.warmup_iters
                return [
                    self.warmup_lr + (group["initial_lr"] - self.warmup_lr) * frac
                    for group in self.optimizer.param_groups
                ]
            else:
                frac = (float(self.last_epoch) - self.warmup_iters) / (
                    self.max_iters - self.warmup_iters
                )
                multiplier = max((1.0 - frac), 0.0) ** self.lr_pow
                return [
                    multiplier * group["initial_lr"]
                    for group in self.optimizer.param_groups
                ]
