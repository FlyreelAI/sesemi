#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""SESEMI learners."""
from typing import NamedTuple, Optional, Tuple

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import logging
import os.path as osp
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.trainer.states import RunningStage

from hydra.utils import instantiate
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanMetric

from .config.structs import ClassifierHParams
from .models.backbones.base import Backbone
from .models.heads.base import LinearHead
from .utils import reduce_tensor, ema_update, copy_and_detach
from .schedulers.weight import WeightScheduler

logger = logging.getLogger(__name__)


class ValidationOutputs(NamedTuple):
    features: torch.Tensor
    logits: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


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
        self.shared_backbones["supervised_backbone"] = instantiate(
            hparams.model.backbone
        )

        self.shared_heads = nn.ModuleDict()
        self.shared_heads["supervised_head"] = instantiate(
            hparams.model.head,
            in_features=self.backbone.out_features,
            out_features=hparams.num_classes,
        )

        # Instantiate the supervised loss, and its scheduler and reduction options.
        if hparams.model.loss is not None:
            self.loss = instantiate(hparams.model.loss.callable, reduction="none")
            if hparams.model.loss.scheduler is not None:
                self.loss_scheduler: Optional[WeightScheduler] = instantiate(
                    hparams.model.loss.scheduler
                )
            else:
                self.loss_scheduler = None
            self.loss_reduction_method = hparams.model.loss.reduction
        else:
            self.loss = None
            self.loss_reduction_method = None

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
            self.shared_backbones["supervised_backbone_ema"] = copy_and_detach(
                self.backbone
            )
            self.shared_heads["supervised_head_ema"] = copy_and_detach(self.head)

        # Build the regularization loss heads.
        for head in self.regularization_loss_heads.values():
            head.build(self.shared_backbones, self.shared_heads)

        # Initialize training and validation metrics.
        self.training_accuracy = Accuracy(top_k=1, dist_sync_on_step=True)
        self.validation_top1_accuracy = Accuracy(top_k=1)
        self.validation_average_loss = MeanMetric()
        self.ema_validation_top1_accuracy = Accuracy(top_k=1)
        self.ema_validation_average_loss = MeanMetric()

    @property
    def backbone(self) -> Backbone:
        """The supervised backbone."""
        return self.shared_backbones["supervised_backbone"]

    @property
    def backbone_ema(self) -> Backbone:
        """The supervised backbone with EMA weights."""
        return self.shared_backbones["supervised_backbone_ema"]

    @property
    def head(self) -> LinearHead:
        """The supervised head."""
        return self.shared_heads["supervised_head"]

    @property
    def head_ema(self) -> LinearHead:
        """The supervised head with EMA weights."""
        return self.shared_heads["supervised_head_ema"]

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
        step_outputs = {}
        if self.head is not None:
            if "supervised" in batch:
                inputs_t, targets_t = batch["supervised"][:2]

                features_t = self.backbone(inputs_t)
                shared_features["supervised_backbone"] = features_t

                outputs_t = self.head(features_t)
                shared_features["supervised_head"] = outputs_t
                shared_features["supervised_probabilities"] = F.softmax(
                    outputs_t, dim=-1
                )

                if self.ema is not None:
                    features_ema = self.backbone_ema(inputs_t)
                    shared_features["supervised_backbone_ema"] = features_ema

                    outputs_ema = self.head(features_t)
                    shared_features["supervised_head_ema"] = outputs_ema
                    shared_features["supervised_probabilities_ema"] = F.softmax(
                        outputs_ema, dim=-1
                    )

                    step_outputs["probabilities_ema"] = shared_features[
                        "supervised_probabilities_ema"
                    ]

                step_outputs["probabilities"] = shared_features[
                    "supervised_probabilities"
                ]

                step_outputs["targets"] = targets_t

        loss = None
        if self.head is not None and self.loss is not None and "supervised" in batch:
            loss = self.loss(shared_features["supervised_head"], targets_t)

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

        step_outputs["regularization_losses"] = regularization_losses

        if loss is not None:
            step_outputs["loss"] = loss

        return step_outputs

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

    def training_step_end(self, outputs):
        losses = []
        if "loss" in outputs:
            _, weighted_loss, _ = self._compute_weighted_training_loss(
                loss=outputs["loss"],
                reduction=self.loss_reduction_method,
                scheduler=self.loss_scheduler,
                scale_factor=self.hparams.model.loss.scale_factor,
                log_prefix="train/supervised",
            )
            losses.append(weighted_loss)

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

        losses.extend(weighted_regularization_losses)
        loss = torch.sum(torch.stack(losses))

        if "probabilities" in outputs and "targets" in outputs:
            self.training_accuracy.update(outputs["probabilities"], outputs["targets"])

        self.log("train/loss", loss)

        self._log_learning_rates()

        if self.ema is not None:
            ema_update(
                self.backbone_ema,
                self.backbone,
                self.ema.decay,
                method=self.ema.method,
                copy_non_floating_point=self.ema.copy_non_floating_point,
            )
            if self.head_ema is not None:
                ema_update(
                    self.head_ema,
                    self.head,
                    self.ema.decay,
                    method=self.ema.method,
                    copy_non_floating_point=self.ema.copy_non_floating_point,
                )

        return loss

    def training_epoch_end(self, outputs) -> None:
        if self.training_accuracy.mode is not None:
            self.log(
                "train/top1",
                self.training_accuracy.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            self.training_accuracy.reset()

    def compute_validation_outputs(
        self, inputs, targets, backbone, head
    ) -> Optional[ValidationOutputs]:
        features = backbone(inputs)
        if head is not None:
            logits = head(features)
            probabilities = torch.softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, targets, reduction="none")
            return ValidationOutputs(
                features=features,
                logits=logits,
                probabilities=probabilities,
                loss=loss,
            )
        else:
            return ValidationOutputs(
                features=features,
            )

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

    def validation_step_end(
        self,
        outputs: Tuple[
            torch.Tensor, Optional[ValidationOutputs], Optional[ValidationOutputs]
        ],
    ):
        targets, regular_outputs, ema_outputs = outputs

        if regular_outputs.probabilities is not None:
            self.validation_top1_accuracy.update(regular_outputs.probabilities, targets)
        if regular_outputs.loss is not None:
            self.validation_average_loss.update(regular_outputs.loss)

        if ema_outputs is not None:
            if ema_outputs.probabilities is not None:
                self.ema_validation_top1_accuracy.update(
                    ema_outputs.probabilities, targets
                )
            if ema_outputs.loss is not None:
                self.ema_validation_average_loss.update(ema_outputs.loss)

        return regular_outputs.features.cpu().numpy(), targets.cpu().numpy()

    def log_validation_metrics(self, top1, best_top1, loss, prefix: str = "val"):
        if top1 is not None:
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

        if loss is not None:
            self.log(
                f"{prefix}/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def validation_epoch_end(self, outputs):
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            val_features = np.concatenate([x[0] for x in outputs], axis=0)
            val_targets = np.concatenate([x[1] for x in outputs], axis=0)

            if self.validation_top1_accuracy.mode is not None:
                top1 = self.validation_top1_accuracy.compute()
                self.validation_top1_accuracy.reset()

                if top1 > self.best_validation_top1_accuracy:
                    self.best_validation_top1_accuracy = torch.tensor(
                        float(top1),
                        dtype=self.best_validation_top1_accuracy.dtype,
                        device=self.best_validation_top1_accuracy.device,
                    )

                self.log_validation_metrics(
                    top1, self.best_validation_top1_accuracy, None, prefix="val"
                )

                ema_top1: Optional[float] = None
                if self.ema_validation_top1_accuracy.mode is not None:
                    ema_top1 = self.ema_validation_top1_accuracy.compute()
                    self.ema_validation_top1_accuracy.reset()

                    if ema_top1 > self.best_ema_validation_top1_accuracy:
                        self.best_ema_validation_top1_accuracy = torch.tensor(
                            float(ema_top1),
                            dtype=self.best_ema_validation_top1_accuracy.dtype,
                            device=self.best_ema_validation_top1_accuracy.device,
                        )

                    self.log_validation_metrics(
                        ema_top1,
                        self.best_ema_validation_top1_accuracy,
                        None,
                        prefix="ema/val",
                    )

            if self.loss is not None:
                loss = self.validation_average_loss.compute()
                self.validation_average_loss.reset()

                ema_loss: Optional[float] = None
                if self.ema is not None:
                    ema_loss = self.ema_validation_average_loss.compute()
                    self.ema_validation_average_loss.reset()

                self.log_validation_metrics(None, None, loss, prefix="val")

                if self.ema is not None:
                    self.log_validation_metrics(
                        None,
                        None,
                        ema_loss,
                        prefix="ema/val",
                    )
