from typing import List
import warnings
from torch.optim import Optimizer

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        iters_per_epoch: int,
        warmup_lr: float,
        lr_pow: float,
        max_iters: int,
        last_epoch: int = -1,
    ):
        self.iters_per_epoch = iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
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
            return [group["initial_lr"] for group in self.optimizer.param_groups]
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
