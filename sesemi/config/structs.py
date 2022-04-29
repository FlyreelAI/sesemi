#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Omegaconf structured configurations."""
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
from omegaconf.dictconfig import DictConfig
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


class DatasetConfig(DictConfig):
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
    root: Optional[str]
    subset: Any
    image_transform: Any
    _target_: str

    def __init__(self, defaults: Optional[Mapping[str, Any]] = None):
        super().__init__(
            {
                "root": None,
                "subset": None,
                "image_transform": None,
                "_target_": "sesemi.dataset",
            }
        )
        if defaults is not None:
            self.update(defaults)


@dataclass
class DataLoaderConfig:
    """The data loader configuration.

    Most of the attributes are taken directly from PyTorch's DataLoader object.

    Attributes:
        dataset: The dataset configuration.
        batch_size: An optional batch size to use for a PyTorch data loader. Cannot be set with
            `batch_size_per_device`.
        batch_size_per_device: An optional batch size per device to use. Cannot be set with `batch_size`.
        shuffle: Whether to shuffle the dataset at each epoch.
        sampler: An optional sampler configuration.
        batch_sampler: An optional batch sampler configuration.
        num_workers: The number of workers to use for data loading (0 means use main process).
        collate_fn: An optional collation callable configuration.
        pin_memory: Whether to pin tensors into CUDA.
        drop_last: Whether to drop the last unevenly sized batch.
        timeout: The timeout to use get data batches from workers.
        worker_init_fn: An optional callable that is invoked for each worker on initialization.
        repeat: The number of times to repeat the dataset on iteration.
        prefetch_factor: The number of samples to prefetch per worker.
        persistent_workers: Whether or not to persist workers after iterating through a dataset.

    References:
        * https://pytorch.org/docs/1.6.0/data.html?highlight=dataloader#torch.utils.data.DataLoader
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    batch_size: Optional[int] = None
    batch_size_per_device: Optional[int] = None
    shuffle: bool = False
    sampler: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    num_workers: Optional[int] = 0
    collate_fn: Optional[Any] = None
    pin_memory: bool = False
    drop_last: Optional[bool] = False
    timeout: float = 0
    worker_init_fn: Optional[Any] = None
    repeat: Optional[int] = None
    prefetch_factor: Optional[int] = 2
    persistent_workers: Optional[bool] = False
    _target_: str = "sesemi.RepeatableDataLoader"


@dataclass
class IgnoredDataConfig:
    """A configuration to specify data loaders that should be ignored.

    Hydra currently has the limitation that `Dict[str, Optional[DataLoaderConfig]]`
    is not considered a valid type to use with a structured config due to the optional value.
    This makes it impossible to override a configuration and set one of the data loaders
    as null. To enable ignoring data loaders in these kind of dictionaries, this supplemental
    configuration supports marking which data loader not to build. A benefit of this approach
    is that the configuration will still be accessible elsewhere.

    Attributes:
        train: An optional dictionary marking which data loaders to ignore.
        extra: An optional dictionary marking which data loaders to ignore.
    """

    train: Optional[Dict[str, bool]] = None
    extra: Optional[Dict[str, bool]] = None


@dataclass
class DataConfig:
    """The data group configuration.

    Attributes:
        train: An optional dictionary of data loader configurations. This configuration is directly
            mapped into dictionaries of data batches.
        val: An optional data loader configuration to use during validation.
        test: An optional data loader configuration to use for testing.
        extra: An optional dictionary of data loader configurations. This configuration is directly
            mapped into dictionaries of data batches.
    """

    train: Optional[Dict[str, DataLoaderConfig]] = None
    val: Optional[DataLoaderConfig] = None
    test: Optional[DataLoaderConfig] = None
    extra: Optional[Dict[str, DataLoaderConfig]] = None

    ignored: IgnoredDataConfig = IgnoredDataConfig()


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
        strategy: Supports either "dp" or "ddp" (the default).
        accelerator: The hardware accelerator to use.
        devices: The number of accelerators to use.
        batch_size_per_device: An optional default batch size per device to use with all data loaders.
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
    strategy: Optional[str] = None
    accelerator: str = "gpu"
    devices: Optional[int] = None
    num_nodes: int = 1
    batch_size_per_device: Optional[int] = None
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
class LoggerConfig:
    """The logger's configuration.

    Attributes:
        decimation: The logging frequency.
        log_images: Whether to log images using the logger.
        log_metrics: Whether to log metrics using the logger.
        log_embeddings: Whether to log embeddings using the logger.
        log_histograms: Whether to log histograms using the logger.
    """

    decimation: int = 50
    log_images: bool = True
    log_metrics: bool = True
    log_embeddings: bool = True
    log_histograms: bool = True


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

    head: Any = MISSING
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
class EMAConfig:
    """A configuration for EMA models.

    Attributes:
        decay: The decay rate of the exponential moving average.
    """

    decay: float = 0.999
    method: str = "states"
    copy_non_floating_point: bool = True


@dataclass
class ClassifierModelConfig:
    """The classifier learner's model configuration.

    Attributes:
        backbone: A backbone config that can be instantiated.
        head: A head config that can be instantiated.
        loss: An optional supervised loss head config.
        regularization_loss_heads: An optional dictionary of loss head configs.
        ema: An optional config for the ema decay coefficient.
    """

    backbone: Any = MISSING
    head: Any = MISSING
    loss: Optional[LossHeadConfig] = LossHeadConfig()
    regularization_loss_heads: Optional[Dict[str, LossHeadConfig]] = None
    ema: Optional[EMAConfig] = None


@dataclass
class ClassifierLoggerConfig(LoggerConfig):
    """The classifier logger's configuration.

    Attributes:
        decimation: The logging frequency.
        log_images: Whether to log images using the logger.
        log_metrics: Whether to log metrics using the logger.
        log_embeddings: Whether to log embeddings using the logger.
        log_histograms: Whether to log histograms using the logger.
        log_gradients: Whether to log gradients.
    """

    log_gradients: bool = False


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
    logger: ClassifierLoggerConfig = ClassifierLoggerConfig()


@dataclass
class ClassifierConfig(LearnerConfig):
    """The classifier configuration.

    Attributes:
        hparams: The classifier's hyperparameters.
    """

    hparams: ClassifierHParams = ClassifierHParams()
    _target_: str = "sesemi.Classifier"


@dataclass
class SESEMIPseudoDatasetConfig:
    """The pseudo-dataset generation config.

    Attributes:
        checkpoint_path: The path to the checkpoint to load.
        seed: The random seed used on initialization.
        output_dir: The directory to save the pseudo-labeled dataset.
        dataset: The dataset configuration.
        preprocessing_transform: The preprocessing transform.
        test_time_augmentation: The test-time augmentation that takes an image
            and returns a list of augmented versions of that image.
        postaugmentation_transform: A transform to apply after test-time
            augmentations and which should return a tensor.
        gpus: Either an integer specifying the number of GPUs to use, a list of
            GPU integer IDs, a comma-separated list of GPU IDs, or None to
            train on the CPU. Setting this to -1 uses all GPUs and setting it
            to 0 also uses the CPU.
        batch_size: The data loading and inference batch size.
        num_workers: The number of workers to use for data loaders.
        symlink_images: Whether to use symlinks for images in the
            pseudo-labeled dataset rather than copying the image.
        use_ema: Whether to use the EMA weights if available.
    """

    checkpoint_path: str = MISSING
    seed: int = 42
    output_dir: str = MISSING
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing_transform: Any = None
    test_time_augmentation: Any = None
    postaugmentation_transform: Any = None
    gpus: Any = -1
    batch_size: int = 16
    num_workers: int = 6
    symlink_images: bool = True
    use_ema: bool = True


@dataclass
class SESEMIInferenceConfig:
    """The inference config.

    Attributes:
        checkpoint_path: The path to the checkpoint to load.
        seed: The random seed used on initialization.
        output_dir: The directory to save the pseudo-labeled dataset.
        data_dir: The dataset directory containing image files potentially nested.
        preprocessing_transform: The preprocessing transform.
        test_time_augmentation: The test-time augmentation that takes an image
            and returns a list of augmented versions of that image.
        postaugmentation_transform: A transform to apply after test-time
            augmentations and which should return a tensor.
        gpus: Either an integer specifying the number of GPUs to use, a list of
            GPU integer IDs, a comma-separated list of GPU IDs, or None to
            train on the CPU. Setting this to -1 uses all GPUs and setting it
            to 0 also uses the CPU.
        batch_size: The data loading and inference batch size.
        num_workers: The number of workers to use for data loaders.
        symlink_images: Whether to use symlinks for images in the
            pseudo-labeled dataset rather than copying the image.
        use_ema: Whether to use the EMA weights if available.
        export_predictions: Whether to export the detailed predictions including
            logits and probabilities.
    """

    checkpoint_path: str = MISSING
    seed: int = 42
    output_dir: str = MISSING
    data_dir: str = MISSING
    preprocessing_transform: Any = None
    test_time_augmentation: Any = None
    postaugmentation_transform: Any = None
    gpus: Any = -1
    batch_size: int = 16
    num_workers: int = 6
    use_ema: bool = True
    export_predictions: bool = True
