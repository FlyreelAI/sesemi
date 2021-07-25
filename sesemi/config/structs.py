from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from omegaconf.omegaconf import MISSING
from enum import Enum


class RunMode(Enum):
    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"


@dataclass
class DatasetConfig:
    name: str
    root: str
    subset: Any = None
    image_transform: Any = None
    _target_: str = "sesemi.dataset"


@dataclass
class DataLoaderConfig:
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
    generator: Any = None
    _target_: str = "sesemi.DataLoader"


@dataclass
class DataConfig:
    train: Optional[Dict[str, DataLoaderConfig]] = None
    val: Optional[DataLoaderConfig] = None
    test: Optional[DataLoaderConfig] = None


@dataclass
class RunConfig:
    seed: Optional[int] = None
    num_epochs: Optional[int] = None
    num_iterations: Optional[int] = None
    gpus: Any = None
    accelerator: Optional[str] = "ddp"
    batch_size_per_gpu: Optional[int] = None
    data_root: Optional[str] = None
    id: str = "default"
    dir: str = "./runs"
    mode: RunMode = RunMode.FIT
    resume_from_checkpoint: Optional[str] = None
    pretrained_checkpoint_path: Optional[str] = None
    no_cuda: bool = False


@dataclass
class LearnerConfig:
    _target_: str = MISSING


@dataclass
class SESEMIConfig:
    run: RunConfig = RunConfig()
    data: DataConfig = DataConfig()
    learner: LearnerConfig = LearnerConfig()
    trainer: Any = MISSING


@dataclass
class LossHeadConfig:
    head: Any
    scheduler: Any = None
    reduction: str = "mean"
    scale_factor: float = 1.0


@dataclass
class LossCallableConfig:
    callable: Any = MISSING
    scheduler: Any = None
    reduction: str = "mean"
    scale_factor: float = 1.0


@dataclass
class LRSchedulerConfig:
    scheduler: Any
    frequency: int = 1
    interval: str = "step"
    monitor: Optional[str] = None
    strict: bool = True
    name: Optional[str] = None


@dataclass
class ClassifierModelConfig:
    backbone: Any = MISSING
    supervised_loss: LossCallableConfig = LossCallableConfig()
    regularization_loss_heads: Optional[Dict[str, LossHeadConfig]] = None


@dataclass
class ClassifierHParams:
    num_classes: int = MISSING
    model: ClassifierModelConfig = ClassifierModelConfig()
    optimizer: Any = MISSING
    lr_scheduler: Optional[LRSchedulerConfig] = None


@dataclass
class ClassifierConfig(LearnerConfig):
    _target_: str = "sesemi.Classifier"
    hparams: ClassifierHParams = ClassifierHParams()
