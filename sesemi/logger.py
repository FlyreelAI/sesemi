#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
from typing import Any, Dict, List, Optional, TypeVar
from pytorch_lightning.loggers import LightningLoggerBase
from sesemi.config.structs import LoggerConfig

from torch import Tensor
from functools import singledispatch
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import rank_zero_only

LoggerConfigType = TypeVar("LoggerConfigType", bound=LoggerConfig)


@singledispatch
def log_image(experiment, tag: str, image: Tensor, step: Optional[int] = None):
    """Log an image using the given tag.

    Args:
        experiment: The experiment logger.
        tag: The tag used for the log.
        image: A rank 3 RGB tensor with shape [H, W, 3].
        step: An optional step used during logging.
    """
    raise NotImplementedError()


@log_image.register
def _(experiment: SummaryWriter, tag: str, image: Tensor, step: Optional[int] = None):
    experiment.add_image(tag, image, global_step=step)


@singledispatch
def log_images(experiment, tag: str, images: Tensor, step: Optional[int] = None):
    """Log images using the given tag.

    Args:
        experiment: The experiment logger.
        tag: The tag used for the log.
        images: A rank 4 RGB tensor with shape [B, H, W, 3].
        step: An optional step used during logging.
    """
    raise NotImplementedError()


@log_images.register
def _(experiment: SummaryWriter, tag: str, images: Tensor, step: Optional[int] = None):
    experiment.add_images(tag, images, global_step=step)


@singledispatch
def log_metrics(experiment, metrics: Dict[str, float], step: Optional[int] = None):
    """Log a set of numeric metrics using the given tag as a prefix.

    Metric tags are formatted as "{tag}/{metric}".

    Args:
        experiment: The experiment logger.
        tag: The prefix tag used for the log.
        metrics: The dictionary of named numeric metrics.
        step: An optional step used during logging.
    """
    raise NotImplementedError()


@log_metrics.register
def _(
    experiment: SummaryWriter,
    metrics: Dict[str, float],
    step: Optional[int] = None,
):
    for key, value in metrics.items():
        experiment.add_scalar(key, value, global_step=step)


@singledispatch
def log_metric(experiment, tag: str, metric: float, step: Optional[int] = None):
    """Log a numeric metric using the given tag.

    Args:
        experiment: The experiment logger.
        tag: The tag used for the log.
        metric: The numeric metric.
        step: An optional step used during logging.
    """
    raise NotImplementedError()


@log_metric.register
def _(
    experiment: SummaryWriter,
    tag: str,
    metric: float,
    step: Optional[int] = None,
):
    experiment.add_scalar(tag, metric, global_step=step)


@singledispatch
def log_embeddings(
    experiment,
    tag: str,
    embeddings: Tensor,
    metadata: Optional[List[str]] = None,
    images: Optional[Tensor] = None,
    step: Optional[int] = None,
):
    """Log a set of embeddings using the given tag.

    Args:
        experiment: The experiment logger.
        tag: The tag used for the log.
        metadata: An optional list of strings to display for each embedding.
        images: An optional batch of images for each embedding to display.
        step: An optional step used during logging.
    """
    raise NotImplementedError()


@log_embeddings.register
def _(
    experiment: SummaryWriter,
    tag: str,
    embeddings: Tensor,
    metadata: Optional[List[str]] = None,
    images: Optional[Tensor] = None,
    step: Optional[int] = None,
):
    experiment.add_embedding(
        embeddings, metadata=metadata, label_img=images, tag=tag, global_step=step
    )


@singledispatch
def log_histogram(
    experiment,
    tag: str,
    values: Tensor,
    step: Optional[int] = None,
):
    """Log a set of embeddings using the given tag.

    Args:
        experiment: The experiment logger.
        tag: The tag used for the log.
        metadata: An optional list of strings to display for each embedding.
        images: An optional batch of images for each embedding to display.
        step: An optional step used during logging.
    """
    raise NotImplementedError()


@log_histogram.register
def _(
    experiment: SummaryWriter,
    tag: str,
    values: Tensor,
    step: Optional[int] = None,
):
    experiment.add_histogram(tag, values, global_step=step)


class LoggerWrapper:
    """A wrapper around the lightning logger that can support many different experiment loggers."""

    def __init__(self, logger: LightningLoggerBase, config: LoggerConfigType):
        """Initialize the logger.

        Args:
            logger: The lightning logger.
            config: The SESEMI logger config.
        """
        self.logger = logger
        self.config = config

    @property
    def experiments(self) -> List[Any]:
        """Returns the list of experiment loggers."""
        if isinstance(self.logger.experiment, (list, tuple)):
            return self.logger.experiment
        return [self.logger.experiment]

    @rank_zero_only
    def log_image(self, tag: str, image: Tensor, step: Optional[int] = None):
        """Log an image using the given tag.

        Args:
            tag: The tag used for the log.
            image: A rank 3 RGB tensor with shape [H, W, 3].
            step: An optional step used during logging.
        """
        if not self.config.log_images:
            return

        if step is not None and step % self.config.decimation != 0:
            return

        min_rgb = image.amin((1, 2), keepdim=True)
        max_rgb = image.amax((1, 2), keepdim=True)

        norm_image = (image - min_rgb) / (max_rgb - min_rgb + 1e-8)
        for experiment in self.experiments:
            log_image(experiment, tag, norm_image, step=step)

    @rank_zero_only
    def log_images(self, tag: str, images: Tensor, step: Optional[int] = None):
        """Log images using the given tag.

        Args:
            tag: The tag used for the log.
            images: A rank 4 RGB tensor with shape [B, H, W, 3].
            step: An optional step used during logging.
        """
        if not self.config.log_images:
            return

        if step is not None and step % self.config.decimation != 0:
            return

        min_rgb = images.amin((0, 2, 3), keepdim=True)
        max_rgb = images.amax((0, 2, 3), keepdim=True)

        norm_images = (images - min_rgb) / (max_rgb - min_rgb + 1e-8)
        for experiment in self.experiments:
            log_images(experiment, tag, norm_images, step=step)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log a set of numeric metrics using the given tag as a prefix.

        Metric tags are formatted as "{tag}/{metric}".

        Args:
            tag: The prefix tag used for the log.
            metrics: The dictionary of named numeric metrics.
            step: An optional step used during logging.
        """
        if not self.config.log_metrics:
            return

        if step is not None and step % self.config.decimation != 0:
            return

        for experiment in self.experiments:
            log_metrics(experiment, metrics, step=step)

    @rank_zero_only
    def log_metric(self, tag: str, metric: float, step: Optional[int] = None):
        """Log a numeric metric using the given tag.

        Args:
            tag: The tag used for the log.
            metric: The numeric metric.
            step: An optional step used during logging.
        """
        if not self.config.log_metrics:
            return

        if step is not None and step % self.config.decimation != 0:
            return

        for experiment in self.experiments:
            log_metric(experiment, tag, metric, step=step)

    @rank_zero_only
    def log_embeddings(
        self,
        tag: str,
        embeddings: Tensor,
        metadata: Optional[List[str]] = None,
        images: Optional[Tensor] = None,
        step: Optional[int] = None,
    ):
        """Log a set of embeddings using the given tag.

        Args:
            tag: The tag used for the log.
            metadata: An optional list of strings to display for each embedding.
            images: An optional batch of images for each embedding to display.
            step: An optional step used during logging.
        """
        if not self.config.log_embeddings:
            return

        if step is not None and step % self.config.decimation != 0:
            return

        for experiment in self.experiments:
            log_embeddings(
                experiment, tag, embeddings, metadata=metadata, images=images, step=step
            )

    @rank_zero_only
    def log_histogram(
        self,
        tag: str,
        values: Tensor,
        step: Optional[int] = None,
    ):
        """Log a histogram using the given tag.

        Args:
            tag: The tag used for the log.
            metadata: An optional list of strings to display for each embedding.
            images: An optional batch of images for each embedding to display.
            step: An optional step used during logging.
        """
        if not self.config.log_histograms:
            return

        if step is not None and step % self.config.decimation != 0:
            return

        for experiment in self.experiments:
            log_histogram(experiment, tag, values, step=step)
