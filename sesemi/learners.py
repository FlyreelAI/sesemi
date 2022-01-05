#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""SESEMI learners."""
from typing import Optional, Tuple

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import logging
import os.path as osp
import pytorch_lightning as pl

from pytorch_lightning.trainer.states import RunningStage

from hydra.utils import instantiate
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanMetric

from .config.structs import ClassifierHParams
from .models.backbones.base import Backbone
from .models.heads.base import LinearHead
from .utils import reduce_tensor
from .schedulers.weight import WeightScheduler

logger = logging.getLogger(__name__)


class Classifier(pl.LightningModule):
    """The SESEMI classifier."""

    hparams: ClassifierHParams  # type: ignore

    def __init__(self, hparams: ClassifierHParams):
        """Initializes the module.

        Args:
            hparams: The classifier's hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        # Instantiate shared nn.Modules
        self.shared_backbones = nn.ModuleDict()
        self.shared_backbones["backbone"] = instantiate(hparams.model.backbone)

        self.shared_heads = nn.ModuleDict()
        self.shared_heads["supervised"] = LinearHead(
            self.backbone.out_features, hparams.num_classes
        )

        # Instantiate the supervised loss, and its scheduler and reduction options.
        self.supervised_loss = instantiate(
            hparams.model.supervised_loss.callable, reduction="none"
        )
        if hparams.model.supervised_loss.scheduler is not None:
            self.supervised_loss_scheduler: Optional[WeightScheduler] = instantiate(
                hparams.model.supervised_loss.scheduler
            )
        else:
            self.supervised_loss_scheduler = None
        self.supervised_loss_reduction_method = hparams.model.supervised_loss.reduction
        self.ema = hparams.model.ema

        self.register_buffer(
            "best_validation_top1_accuracy",
            torch.tensor(0.0, dtype=torch.float32, device=self.device),
        )
        self.register_buffer(
            "best_ema_validation_top1_accuracy",
            torch.tensor(0.0, dtype=torch.float32, device=self.device),
        )

        # Instantiate the regularization loss heads,
        # and their optional scheduler and reduction methods.
        self.regularization_loss_heads = nn.ModuleDict()
        self.regularization_loss_schedulers = {}
        self.regularization_loss_reduction_methods = {}

        regularization_loss_head_configs = hparams.model.regularization_loss_heads or {}
        for name, loss_head_config in regularization_loss_head_configs.items():
            self.regularization_loss_heads[name] = instantiate(
                loss_head_config.head, logger=self.logger
            )
            if loss_head_config.scheduler is not None:
                self.regularization_loss_schedulers[name] = instantiate(
                    loss_head_config.scheduler
                )
            else:
                self.regularization_loss_schedulers[name] = None
            self.regularization_loss_reduction_methods[name] = (
                loss_head_config.reduction or "mean"
            )
        self.num_regularization_losses = len(self.regularization_loss_heads)

        if self.ema is not None:
            assert (
                0.0 <= self.ema.decay <= 1.0
            ), "EMA decay value should be between [0, 1]. Default 0.999."
            self.shared_backbones["backbone_ema"] = self._copy_and_detach(self.backbone)
            self.shared_heads["supervised_ema"] = self._copy_and_detach(self.head)

        # Build the regularization loss heads.
        for head in self.regularization_loss_heads.values():
            head.build(self.shared_backbones, self.shared_heads)

        # Initialize training and validation metrics.
        self.training_accuracy = Accuracy(top_k=1, dist_sync_on_step=True)
        self.validation_top1_accuracy = Accuracy(top_k=1)
        self.validation_average_loss = MeanMetric()
        self.ema_validation_top1_accuracy = Accuracy(top_k=1)
        self.ema_validation_average_loss = MeanMetric()

    def _copy_and_detach(self, module):
        """Detaches a new module from the computational graph after copying."""
        new_module = copy.deepcopy(module)
        for param in new_module.parameters():
            param.detach_()
        return new_module

    @property
    def backbone(self) -> Backbone:
        """The supervised backbone."""
        return self.shared_backbones["backbone"]

    @property
    def backbone_ema(self) -> Backbone:
        """The supervised backbone with EMA weights."""
        return self.shared_backbones["backbone_ema"]

    @property
    def head(self) -> LinearHead:
        """The supervised head."""
        return self.shared_heads["supervised"]

    @property
    def head_ema(self) -> LinearHead:
        """The supervised head with EMA weights."""
        return self.shared_heads["supervised_ema"]

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())

        if self.hparams.lr_scheduler is not None:
            lr_dict = dict(self.hparams.lr_scheduler)
            lr_dict["scheduler"] = instantiate(
                lr_dict["scheduler"], optimizer=optimizer
            )
            return [optimizer], [lr_dict]

        return optimizer

    def training_step(self, batch, batch_index):
        shared_features = {}

        inputs_t, targets_t = batch["supervised"][:2]
        features_t = self.backbone(inputs_t)
        outputs_t = self.head(features_t)

        shared_features["backbone"] = features_t
        shared_features["supervised"] = outputs_t

        supervised_loss = self.supervised_loss(outputs_t, targets_t)

        regularization_losses = {
            name: head(
                data=batch,
                backbones=self.shared_backbones,
                heads=self.shared_heads,
                features=shared_features,
                step=self.global_step,
            )
            for name, head in self.regularization_loss_heads.items()
        }

        return {
            "supervised_loss": supervised_loss,
            "regularization_losses": regularization_losses,
            "probs": F.softmax(outputs_t, dim=-1),
            "targets": targets_t,
        }

    def _compute_weighted_training_loss(
        self,
        loss: Tensor,
        reduction: Optional[str],
        scheduler: Optional[WeightScheduler],
        scale_factor: float,
        log_prefix: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, float]:
        reduced_loss = reduce_tensor(loss, reduction)

        if scheduler is not None:
            scheduler_weight = scheduler(self.global_step)
        else:
            scheduler_weight = 1.0

        loss_weight = scale_factor * scheduler_weight
        weighted_loss = loss_weight * reduced_loss

        if log_prefix is not None:
            self.log(osp.join(log_prefix, "loss"), reduced_loss, on_step=True)
            self.log(osp.join(log_prefix, "loss_weight"), loss_weight, on_step=True)
            self.log(osp.join(log_prefix, "weighted_loss"), weighted_loss, on_step=True)

        return reduced_loss, weighted_loss, loss_weight

    def _log_learning_rates(self):
        optim = self.optimizers()
        param_group0_lr = optim.optimizer.param_groups[0]["lr"]
        for i, param_group in enumerate(optim.optimizer.param_groups):
            self.log(f"optim/lr/param_group/{i}", param_group["lr"], on_step=True)

        self.log(
            "lr",
            param_group0_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

    def _ema_update(self, ema_module, module, step, decay):
        """Computes in-place the EMA parameters from the original parameters."""
        # Use the true average until the exponential average is more correct.
        decay = min(1.0 - 1.0 / (step + 1.0), decay)
        for ema_param, param in zip(ema_module.parameters(), module.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1.0 - decay))

    def training_step_end(self, outputs):
        _, weighted_supervised_loss, _ = self._compute_weighted_training_loss(
            loss=outputs["supervised_loss"],
            reduction=self.supervised_loss_reduction_method,
            scheduler=self.supervised_loss_scheduler,
            scale_factor=self.hparams.model.supervised_loss.scale_factor,
            log_prefix="train/supervised",
        )

        weighted_regularization_losses = []
        for name, regularization_loss in outputs["regularization_losses"].items():
            _, weighted_regularization_loss, _ = self._compute_weighted_training_loss(
                loss=regularization_loss,
                reduction=self.regularization_loss_reduction_methods[name],
                scheduler=self.regularization_loss_schedulers[name],
                scale_factor=self.hparams.model.regularization_loss_heads[
                    name
                ].scale_factor,
                log_prefix=f"train/regularization/{name}",
            )

            weighted_regularization_losses.append(weighted_regularization_loss)

        losses = [weighted_supervised_loss] + weighted_regularization_losses
        loss = torch.sum(torch.stack(losses))

        self.training_accuracy.update(outputs["probs"], outputs["targets"])

        self.log("train/loss", loss)

        self._log_learning_rates()

        if self.ema is not None:
            self._ema_update(
                self.backbone_ema, self.backbone, self.global_step, self.ema.decay
            )
            self._ema_update(self.head_ema, self.head, self.global_step, self.ema.decay)

        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log(
            "train/top1",
            self.training_accuracy.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.training_accuracy.reset()

    def compute_validation_outputs(self, inputs, targets, backbone, head):
        outputs_t = head(backbone(inputs))
        probs_t = torch.softmax(outputs_t, dim=-1)
        loss_t = F.cross_entropy(outputs_t, targets, reduction="none")
        return probs_t, loss_t

    def validation_step(self, batch, batch_index):
        inputs_t, targets_t = batch

        regular_outputs = self.compute_validation_outputs(
            inputs_t, targets_t, self.backbone, self.head
        )

        ema_outputs = (
            self.compute_validation_outputs(
                inputs_t, targets_t, self.backbone_ema, self.head_ema
            )
            if self.ema is not None
            else None
        )

        return targets_t, regular_outputs, ema_outputs

    def validation_step_end(self, outputs):
        targets_t, regular_outputs, ema_outputs = outputs

        outputs_t, loss_t = regular_outputs
        self.validation_top1_accuracy.update(outputs_t, targets_t)
        self.validation_average_loss.update(loss_t)

        if ema_outputs is not None:
            ema_outputs_t, ema_loss_t = ema_outputs
            self.ema_validation_top1_accuracy.update(ema_outputs_t, targets_t)
            self.ema_validation_average_loss.update(ema_loss_t)

    def log_validation_metrics(self, top1, best_top1, loss, prefix: str = "val"):
        self.log(
            f"{prefix}/top1",
            top1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            f"{prefix}/top1/best",
            best_top1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_epoch_end(self, outputs):
        top1 = self.validation_top1_accuracy.compute()
        loss = self.validation_average_loss.compute()
        self.validation_top1_accuracy.reset()
        self.validation_average_loss.reset()

        ema_top1: Optional[float] = None
        ema_loss: Optional[float] = None
        if self.ema is not None:
            ema_top1 = self.ema_validation_top1_accuracy.compute()
            ema_loss = self.ema_validation_average_loss.compute()
            self.ema_validation_top1_accuracy.reset()
            self.ema_validation_average_loss.reset()

        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            if top1 > self.best_validation_top1_accuracy:
                self.best_validation_top1_accuracy = torch.tensor(
                    float(top1),
                    dtype=self.best_validation_top1_accuracy.dtype,
                    device=self.best_validation_top1_accuracy.device,
                )

            self.log_validation_metrics(
                top1, self.best_validation_top1_accuracy, loss, prefix="val"
            )

            if self.ema is not None:
                if ema_top1 > self.best_ema_validation_top1_accuracy:
                    self.best_ema_validation_top1_accuracy = torch.tensor(
                        float(ema_top1),
                        dtype=self.best_ema_validation_top1_accuracy.dtype,
                        device=self.best_ema_validation_top1_accuracy.device,
                    )

                self.log_validation_metrics(
                    ema_top1,
                    self.best_ema_validation_top1_accuracy,
                    ema_loss,
                    prefix="ema/val",
                )
