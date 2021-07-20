from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import MISSING
from enum import Enum


class RunMode(Enum):
    TRAIN = "train"
    EVALUATE_ONLY = "evaluate-only"


@dataclass
class RunConfig:
    seed: Optional[int] = None
    num_epochs: int = MISSING
    id: str = "run01"
    log_dir: str = "./logs"
    mode: RunMode = RunMode.TRAIN
    resume_from_checkpoint: Optional[str] = None
    pretrained_checkpoint_path: Optional[str] = None
    no_cuda: bool = False


@dataclass
class SESEMIConfig:
    run: RunConfig = RunConfig()
    data: Any = MISSING
    trainer: Any = MISSING
    learner: Any = MISSING


@dataclass
class LossHeadConfig:
    head: Any
    scheduler: Any = None
    reduction: str = "mean"


@dataclass
class LossCallableConfig:
    callable: Any = MISSING
    scheduler: Any = None
    reduction: str = "mean"


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
class ClassifierConfig:
    classes: List[str]
    model: ClassifierModelConfig = ClassifierModelConfig()
    optimizer: Any = MISSING
    lr_scheduler: Optional[LRSchedulerConfig] = None
