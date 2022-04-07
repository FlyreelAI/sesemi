#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""SESEMI learners."""
from functools import cached_property
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import numpy as np
import os.path as osp
import pytorch_lightning as pl

from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.core.optimizer import LightningOptimizer

from hydra.utils import instantiate
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanMetric

from natsort import natsorted
from tqdm import tqdm

from sesemi.models.loss_heads.base import LossOutputs

from .config.structs import ClassifierHParams, SESEMIBaseConfig
from .models.backbones.base import Backbone
from .models.heads import LinearHead
from .utils import reduce_tensor, ema_update, copy_and_detach
from .schedulers.weight import WeightScheduler
from .logger import LoggerWrapper

logger = logging.getLogger(__name__)


class ClassifierValidationOutputs(NamedTuple):
    """The classifier's validation outputs."""

    features: torch.Tensor
    logits: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class Classifier(pl.LightningModule):
    """The SESEMI classifier."""

    hparams: SESEMIBaseConfig  # type: ignore

    def __init__(self, sesemi_config: SESEMIBaseConfig, hparams: ClassifierHParams):
        """Initializes the module.

        Args:
            sesemi_config: The full training configuration including the classifier's
                hyperparameters.
            hparams: The classifier's hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()

        assert (
            hparams == sesemi_config.learner.hparams
        ), "expected matching hyperparameters"

        self.sesemi_config = sesemi_config

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
            self.loss = instantiate(hparams.model.loss.head)
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
            self.regularization_loss_heads[name] = instantiate(loss_head_config.head)
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

        # Build the regularization loss heads in natsorted order.
        natsorted_loss_head_names = natsorted(
            list(regularization_loss_head_configs.keys())
        )
        for name in natsorted_loss_head_names:
            head = self.regularization_loss_heads[name]
            head.build(self.shared_backbones, self.shared_heads)

        # Initialize training and validation metrics.
        self.training_accuracy = Accuracy(top_k=1, dist_sync_on_step=True)
        self.validation_top1_accuracy = Accuracy(top_k=1)
        self.validation_average_loss = MeanMetric()
        self.ema_validation_top1_accuracy = Accuracy(top_k=1)
        self.ema_validation_average_loss = MeanMetric()

    @property
    def classifier_hparams(self) -> ClassifierHParams:
        """The classifier's hyperparameters."""
        return self.sesemi_config.learner.hparams

    @cached_property
    def logger_wrapper(self) -> LoggerWrapper:
        """The logger wrapper."""
        return LoggerWrapper(self.logger, self.classifier_hparams.logger)

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

    @property
    def has_ema(self) -> bool:
        """Whether the model has  EMA weights."""
        return self.ema is not None

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def forward_ema(self, x: Tensor) -> Tensor:
        features = self.backbone_ema(x)
        logits = self.head_ema(features)
        return logits

    def configure_optimizers(
        self,
    ) -> Union[Tuple[torch.optim.Optimizer, Dict[str, Any]], torch.optim.Optimizer]:
        optimizer = instantiate(
            self.classifier_hparams.optimizer, params=self.parameters()
        )

        if self.classifier_hparams.lr_scheduler is not None:
            lr_dict = dict(self.classifier_hparams.lr_scheduler)
            lr_dict["scheduler"] = instantiate(
                lr_dict["scheduler"], optimizer=optimizer
            )
            return [optimizer], [lr_dict]

        return optimizer

    def on_before_optimizer_step(
        self, optimizer: LightningOptimizer, optimizer_idx: int
    ):
        if self.classifier_hparams.logger.log_gradients:
            for name, value in self.named_parameters():
                if value.grad is not None:
                    self.logger_wrapper.log_histogram(
                        f"optim/{name}", value.grad, step=self.global_step
                    )

    def training_step(self, batch: Dict[str, Any], batch_index: int):
        shared_features = {}
        step_outputs = {}
        if self.head is not None:
            if "supervised" in batch:
                images, targets = batch["supervised"][:2]

                features = self.backbone(images)
                shared_features["supervised_backbone"] = features

                outputs = self.head(features)
                shared_features["supervised_head"] = outputs
                shared_features["supervised_probabilities"] = F.softmax(outputs, dim=-1)

                if self.ema is not None:
                    features_ema = self.backbone_ema(images)
                    shared_features["supervised_backbone_ema"] = features_ema

                    outputs_ema = self.head_ema(features)
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

                step_outputs["targets"] = targets

                self.logger_wrapper.log_images(
                    "supervised/images", images, step=self.global_step
                )

        loss = None
        if self.head is not None and self.loss is not None and "supervised" in batch:
            loss = self.loss(
                data=batch,
                backbones=self.shared_backbones,
                heads=self.shared_heads,
                features=shared_features,
                step=self.global_step,
                logger_wrapper=self.logger_wrapper,
            ).asdict()

        regularization_losses = {
            name: head(
                data=batch,
                backbones=self.shared_backbones,
                heads=self.shared_heads,
                features=shared_features,
                step=self.global_step,
                logger_wrapper=self.logger_wrapper,
            ).asdict()
            for name, head in self.regularization_loss_heads.items()
        }

        step_outputs["regularization_losses"] = regularization_losses

        if loss is not None:
            step_outputs["loss"] = loss

        return step_outputs

    def _compute_weighted_training_loss(
        self,
        losses: Tensor,
        weights: Optional[Tensor],
        reduction: Optional[str],
        scheduler: Optional[WeightScheduler],
        scale_factor: float,
        log_prefix: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, float]:
        reduced_loss = reduce_tensor(losses, weights=weights, reduction=reduction)

        if scheduler is not None:
            scheduler_weight = scheduler(self.global_step)
        else:
            scheduler_weight = 1.0

        loss_weight = scale_factor * scheduler_weight
        weighted_loss = loss_weight * reduced_loss

        if log_prefix is not None:
            self.log(osp.join(log_prefix, "reduced_loss"), reduced_loss, on_step=True)
            self.log(
                osp.join(log_prefix, "scalar_loss_weight"), loss_weight, on_step=True
            )
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

    def _log_gradients(self):
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

    def training_step_end(self, outputs: Dict[str, Tensor]) -> Tensor:
        losses = []
        if "loss" in outputs:
            supservised_loss: LossOutputs = outputs["loss"]
            _, weighted_loss, _ = self._compute_weighted_training_loss(
                losses=supservised_loss["losses"],
                weights=supservised_loss["weights"],
                reduction=self.loss_reduction_method,
                scheduler=self.loss_scheduler,
                scale_factor=self.classifier_hparams.model.loss.scale_factor,
                log_prefix="train/supervised",
            )
            losses.append(weighted_loss)

        weighted_regularization_losses = []
        regularization_losses: Dict[str, LossOutputs] = outputs["regularization_losses"]
        for name, regularization_loss in regularization_losses.items():
            _, weighted_regularization_loss, _ = self._compute_weighted_training_loss(
                losses=regularization_loss["losses"],
                weights=regularization_loss["weights"],
                reduction=self.regularization_loss_reduction_methods[name],
                scheduler=self.regularization_loss_schedulers[name],
                scale_factor=self.classifier_hparams.model.regularization_loss_heads[
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

        return loss

    def on_train_batch_end(
        self, outputs: Tensor, batch: Dict[str, Any], batch_idx: int, unused: int = 0
    ):
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

    def training_epoch_end(self, outputs: List[Tensor]) -> None:
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
    ) -> Optional[ClassifierValidationOutputs]:
        features = backbone(inputs)
        if head is not None:
            logits = head(features)
            probabilities = torch.softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, targets, reduction="none")
            return ClassifierValidationOutputs(
                features=features,
                logits=logits,
                probabilities=probabilities,
                loss=loss,
            )
        else:
            return ClassifierValidationOutputs(
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
            torch.Tensor,
            Optional[ClassifierValidationOutputs],
            Optional[ClassifierValidationOutputs],
        ],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        targets, regular_outputs, ema_outputs = outputs

        if regular_outputs.probabilities is not None:
            self.validation_top1_accuracy.update(regular_outputs.probabilities, targets)
        if regular_outputs.loss is not None:
            self.validation_average_loss.update(regular_outputs.loss)

        ema_features_np: Optional[np.ndarray] = None
        if ema_outputs is not None:
            if ema_outputs.probabilities is not None:
                self.ema_validation_top1_accuracy.update(
                    ema_outputs.probabilities, targets
                )
            if ema_outputs.loss is not None:
                self.ema_validation_average_loss.update(ema_outputs.loss)
            ema_features_np = ema_outputs.features.cpu().numpy()

        return (
            regular_outputs.features.cpu().numpy(),
            ema_features_np,
            targets.cpu().numpy(),
        )

    def log_validation_metrics(
        self,
        top1: Optional[float],
        best_top1: Optional[float],
        loss: Optional[float],
        prefix: str = "val",
    ):
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

    def _compute_knn_evaluation_features(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        knn_evaluation_dataloader = self.trainer.datamodule.extra_dataloader(
            "knn_evaluation"
        )
        if knn_evaluation_dataloader is None:
            return None

        train_features_l = []
        train_targets_l = []

        for image, targets in tqdm(
            knn_evaluation_dataloader, desc="kNN Evaluation", leave=False
        ):
            feats = self.backbone(image.to(self.device))
            train_features_l.append(feats.cpu().numpy())
            train_targets_l.append(targets.cpu().numpy())

        train_features = np.concatenate(train_features_l, axis=0)
        train_targets = np.concatenate(train_targets_l, axis=0)

        return train_features, train_targets

    def _compute_knn_evaluation_score(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        val_features: np.ndarray,
        val_targets: np.ndarray,
    ) -> float:
        """
        Adapted from `knn_predict` in the following notebook:
        https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
        """
        val_features_tensor = torch.from_numpy(val_features).to(torch.float32)
        train_features_tensor = torch.from_numpy(train_features).to(torch.float32)

        train_targets_tensor = torch.from_numpy(train_targets).to(torch.float32)

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        normalized_val_features = F.normalize(val_features_tensor, dim=-1)
        normalized_train_features_transpose = F.normalize(
            train_features_tensor, dim=-1
        ).T

        sim_matrix = torch.mm(
            normalized_val_features,
            normalized_train_features_transpose,
        )

        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)

        # [B, K]
        sim_labels = torch.gather(
            train_targets_tensor.expand(val_features_tensor.size(0), -1),
            dim=-1,
            index=sim_indices,
        )
        sim_weight = (sim_weight / 0.07).exp()

        # counts for each class
        one_hot_label = torch.zeros(
            val_features_tensor.size(0) * 200,
            self.classifier_hparams.num_classes,
            device=sim_labels.device,
        )
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0
        )
        # weighted score ---> [B, C]
        pred_scores = torch.sum(
            one_hot_label.view(
                val_features_tensor.size(0), -1, self.classifier_hparams.num_classes
            )
            * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )

        pred_labels = pred_scores.argmax(dim=-1)
        pred_labels_np = pred_labels.cpu().numpy()
        knn_evaluation_score = (pred_labels_np == val_targets).sum() / max(
            pred_labels_np.shape[0], 1
        )

        return knn_evaluation_score

    def validation_epoch_end(
        self,
        outputs: List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    ):
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            val_features = np.concatenate([x[0] for x in outputs], axis=0)
            val_targets = np.concatenate([x[2] for x in outputs], axis=0)

            ema_val_features: Optional[np.ndarray] = None
            if outputs[0][1] is not None:
                ema_val_features = np.concatenate([x[1] for x in outputs], axis=0)

            hp_metric: Optional[float] = None
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

                hp_metric = ema_top1 if ema_top1 is not None else top1

            knn_evaluation_features = self._compute_knn_evaluation_features()
            if knn_evaluation_features is not None:
                train_features, train_targets = knn_evaluation_features
                knn_evaluation_score = self._compute_knn_evaluation_score(
                    train_features, train_targets, val_features, val_targets
                )

                ema_knn_evaluation_score: Optional[float] = None
                if ema_val_features is not None:
                    ema_knn_evaluation_score = self._compute_knn_evaluation_score(
                        train_features, train_targets, ema_val_features, val_targets
                    )

                self.log(
                    "val/knn_evaluation/top1",
                    knn_evaluation_score,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                if ema_knn_evaluation_score is not None:
                    if hp_metric is None:
                        hp_metric = ema_loss if ema_loss is not None else loss

                    self.log(
                        "val/knn_evaluation/top1",
                        knn_evaluation_score,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )

                if hp_metric is None:
                    hp_metric = (
                        ema_knn_evaluation_score
                        if ema_knn_evaluation_score is not None
                        else knn_evaluation_score
                    )

            if self.loss is not None:
                loss = self.validation_average_loss.compute()
                self.validation_average_loss.reset()

                ema_loss: Optional[float] = None
                if self.ema is not None:
                    ema_loss = self.ema_validation_average_loss.compute()
                    self.ema_validation_average_loss.reset()

                if hp_metric is None:
                    hp_metric = ema_loss if ema_loss is not None else loss

                self.log_validation_metrics(None, None, loss, prefix="val")

                if self.ema is not None:
                    self.log_validation_metrics(
                        None,
                        None,
                        ema_loss,
                        prefix="ema/val",
                    )

            if hp_metric is not None:
                self.log(
                    "hp_metric",
                    hp_metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
