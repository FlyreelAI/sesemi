#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Omegaconf structured configurations."""
from dataclasses import dataclass
from typing import Any, Dict, Optional
from omegaconf.omegaconf import MISSING
from enum import Enum


class RunMode(Enum):
    """The mode to use when running the training CLI.

    Attributes:
        FIT: Performs training and validation.
        VALIDATE: Generates the validation metrics.
        TEST: Generates the test metrics.
    """

    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"


@dataclass
class DatasetConfig:
    """The SESEMI dataset configuration.

    Attributes:
        name: The name of the dataset in the registry.
        root: The absolute or relative (with respect to the run's data root) path to the dataset.
            If this is None, then the run's `data_root` attribute will be used instead.
        subset: Can either be a the string name of the subset, a list of subsets, or None (null)
            to indicate the full set.
        image_transform: An optional callable transform configuration that is applied to images.
    """

    name: str
    root: Optional[str] = None
    subset: Any = None
    image_transform: Any = None
    _target_: str = "sesemi.dataset"


@dataclass
class DataLoaderConfig:
    """The data loader configuration.

    Most of the attributes are taken directly from PyTorch's DataLoader object.

    Attributes:
        dataset: The dataset configuration.
        batch_size: An optional batch size to use for a PyTorch data loader. Cannot be set with
            `batch_size_per_gpu`.
        batch_size_per_gpu: An optional batch size per GPU to use. Cannot be set with `batch_size`.
        shuffle: Whether to shuffle the dataset at each epoch.
        sampler: An optional sampler configuration.
        batch_sampler: An optional batch sampler configuration.
        num_workers: The number of workers to use for data loading (0 means use main process).
        collate_fn: An optional collation callable configuration.
        pin_memory: Whether to pin tensors into CUDA.
        drop_last: Whether to drop the last unevenly sized batch.
        timeout: The timeout to use get data batches from workers.
        worker_init_fn: An optional callable that is invoked for each worker on initialization.

    References:
        * https://pytorch.org/docs/1.6.0/data.html?highlight=dataloader#torch.utils.data.DataLoader
    """

    dataset: DatasetConfig
    batch_size: Optional[int] = None
    batch_size_per_gpu: Optional[int] = None
    shuffle: bool = False
    sampler: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    num_workers: Optional[int] = 0
    collate_fn: Optional[Any] = None
    pin_memory: bool = False
    drop_last: Optional[bool] = False
    timeout: float = 0
    worker_init_fn: Optional[Any] = None
    _target_: str = "sesemi.DataLoader"


@dataclass
class DataConfig:
    """The data group configuration.

    Attributes:
        train: An optional dictionary of data loader configurations. This configuration is directly
            mapped into dictionaries of data batches.
        val: An optional data loader configuration to use during validation.
        test: An optional data loader configuration to use for testing.
    """

    train: Optional[Dict[str, DataLoaderConfig]] = None
    val: Optional[DataLoaderConfig] = None
    test: Optional[DataLoaderConfig] = None


@dataclass
class RunConfig:
    """The run configuration.

    Attributes:
        seed: An optional random seed used on initialization.
        num_epochs: The number of training epochs to train. Cannot be set with `num_iterations`.
        num_iterations: The number of training iterations to run. Cannot be set with `num_epochs`.
        gpus: Either an integer specifying the number of GPUs to use, a list of GPU
            integer IDs, a comma-separated list of GPU IDs, or None to train on the CPU. Setting
            this to -1 uses all GPUs and setting it to 0 also uses the CPU.
        num_nodes: The number of nodes to use during training (defaults to 1).
        accelerator: Supports either "dp" or "ddp" (the default).
        batch_size_per_gpu: An optional default batch size per GPU to use with all data loaders.
        data_root: The directory to use as the parent of relative dataset root directories
            (see `DatasetConfig`).
        id: The identifier to use for the run.
        dir: The directory to store run outputs (e.g. logs, configurations, etc.).
        mode: The run's mode.
        resume_from_checkpoint: An optional checkpoint path to restore trainer state.
        pretrained_checkpoint_path: An optional checkpoint path to load pretrained model weights.
    """

    seed: Optional[int] = None
    num_epochs: Optional[int] = None
    num_iterations: Optional[int] = None
    gpus: Any = -1
    num_nodes: int = 1
    accelerator: Optional[str] = None
    batch_size_per_gpu: Optional[int] = None
    data_root: str = "./data"
    id: str = "default"
    dir: str = "./runs"
    mode: RunMode = RunMode.FIT
    resume_from_checkpoint: Optional[str] = None
    pretrained_checkpoint_path: Optional[str] = None


@dataclass
class LearnerConfig:
    """A base learner configuration."""

    _target_: str = MISSING


@dataclass
class SESEMIBaseConfig:
    """The base SESEMI configuration.

    Attributes:
        run: The run config.
        data: The data config.
        learner: The learner config.
        trainer: Optional additional parameters that can be passed to a PyTorch Lightning Trainer
            object.

    References:
        * https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api
    """

    run: RunConfig = RunConfig()
    data: DataConfig = DataConfig()
    learner: LearnerConfig = LearnerConfig()
    trainer: Any = None


@dataclass
class LossHeadConfig:
    """The loss head configuration.

    Attributes:
        head: A loss head configuration that can be instantiated.
        scheduler: An optional learning rate scheduler that can be instantiated.
        reduction: The loss reduction method to use (e.g. "mean" or "sum").
        scale_factor: The loss scale factor.
    """

    head: Any
    scheduler: Any = None
    reduction: str = "mean"
    scale_factor: float = 1.0


@dataclass
class LossCallableConfig:
    """A callable loss configuration.

    Attributes:
        callable: An callable configuration that can be instantiated.
        scheduler: An optional learning rate scheduler that can be instantiated.
        reduction: The loss reduction method to use (e.g. "mean" or "sum").
        scale_factor: The loss scale factor.
    """

    callable: Any = MISSING
    scheduler: Any = None
    reduction: str = "mean"
    scale_factor: float = 1.0


@dataclass
class LRSchedulerConfig:
    """A learning rate scheduler configuration.

    Attributes:
        scheduler: A learning rate scheduler that can be instantiated.
        frequency: The frequency of learning rate updates as a multiple of `interval`.
        interval: Specifies the learning rate update interval. Can be either "step" or "epoch".
        monitor: An optional metric that is monitored for specific kinds of schedulers.
        strict: Wehether to enforce that the `monitor` metric should exist on updates.
        name: An optional name that can be used by certain learning rate callbacks.
    """

    scheduler: Any
    frequency: int = 1
    interval: str = "step"
    monitor: Optional[str] = None
    strict: bool = True
    name: Optional[str] = None


@dataclass
class ClassifierModelConfig:
    """The classifier learner's model configuration.

    Attributes:
        backbone: A backbone config that can be instantiated.
        supervised_loss: A callable loss config.
        regularization_loss_heads: An optional dictionary of loss head configs.
    """

    backbone: Any = MISSING
    supervised_loss: LossCallableConfig = LossCallableConfig()
    regularization_loss_heads: Optional[Dict[str, LossHeadConfig]] = None


@dataclass
class ClassifierHParams:
    """The classifier hyperparameters.

    Attributes:
        num_classes: The number of classes to use.
        model: The classifier's model config.
        optimizer: An optimizer config that can be instantiated.
        lr_scheduler: An optional learning rate scheduler config.
    """

    num_classes: int = MISSING
    model: ClassifierModelConfig = ClassifierModelConfig()
    optimizer: Any = MISSING
    lr_scheduler: Optional[LRSchedulerConfig] = None


@dataclass
class ClassifierConfig(LearnerConfig):
    """The classifier configuration.

    Attributes:
        hparams: The classifier's hyperparameters.
    """

    hparams: ClassifierHParams = ClassifierHParams()
    _target_: str = "sesemi.Classifier"
