import pytest
import torch

from sesemi.logger import LoggerWrapper
from sesemi.config.structs import LoggerConfig


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("step,decimation,called", [
    (0, 1, True),
    (1, 1, True),
    (2, 1, True),
    (None, 1,True),
    (1, 2, False),
])
def test_logger_wrapper_log_image_tensorboard(tensorboard_logger, log, step, decimation, called, mocker):
    called = called and log
    logger_config = LoggerConfig(decimation=decimation, log_images=log)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    image = torch.ones((3, 32, 32), dtype=torch.float32)
    add_image = mocker.spy(tensorboard_logger.experiment, 'add_image')
    logger_wrapper.log_image("test", image, step=step)
    if called:
        add_image.assert_called_once()
    else:
        add_image.assert_not_called()


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("step,decimation,called", [
    (0, 1, True),
    (1, 1, True),
    (2, 1, True),
    (None, 1,True),
    (1, 2, False),
])
def test_logger_wrapper_log_images_tensorboard(tensorboard_logger, log, step, decimation, called, mocker):
    called = called and log
    logger_config = LoggerConfig(decimation=decimation, log_images=log)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    image = torch.ones((2, 3, 32, 32), dtype=torch.float32)
    add_images = mocker.spy(tensorboard_logger.experiment, 'add_images')
    logger_wrapper.log_images("test", image, step=step)
    if called:
        add_images.assert_called_once()
    else:
        add_images.assert_not_called()


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("step,decimation,called", [
    (0, 1, True),
    (1, 1, True),
    (2, 1, True),
    (None, 1,True),
    (1, 2, False),
])
def test_logger_wrapper_log_metrics_tensorboard(tensorboard_logger, log, step, decimation, called, mocker):
    called = called and log
    logger_config = LoggerConfig(decimation=decimation, log_metrics=log)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    add_scalar = mocker.spy(tensorboard_logger.experiment, 'add_scalar')
    metrics = {'test1': 1.0, 'test2': 2.0}
    logger_wrapper.log_metrics(metrics, step=step)
    if called:
        assert add_scalar.call_count == len(metrics)
    else:
        add_scalar.assert_not_called()


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("step,decimation,called", [
    (0, 1, True),
    (1, 1, True),
    (2, 1, True),
    (None, 1,True),
    (1, 2, False),
])
def test_logger_wrapper_log_metric_tensorboard(tensorboard_logger, log, step, decimation, called, mocker):
    called = called and log
    logger_config = LoggerConfig(decimation=decimation, log_metrics=log)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    add_scalar = mocker.spy(tensorboard_logger.experiment, 'add_scalar')
    logger_wrapper.log_metric("test", 1.0, step=step)
    if called:
        add_scalar.assert_called_once()
    else:
        add_scalar.assert_not_called()


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("step,decimation,called", [
    (0, 1, True),
    (1, 1, True),
    (2, 1, True),
    (None, 1,True),
    (1, 2, False),
])
@pytest.mark.parametrize("metadata", [None, ['a', 'b']])
@pytest.mark.parametrize("images", [None, torch.ones((2, 3, 32, 32), dtype=torch.float32)])
def test_logger_wrapper_log_embeddings_tensorboard(tensorboard_logger, log, step, decimation, metadata, images, called, mocker):
    called = called and log
    logger_config = LoggerConfig(decimation=decimation, log_embeddings=log)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    add_embedding = mocker.spy(tensorboard_logger.experiment, 'add_embedding')
    embeddings = torch.ones((2, 5), dtype=torch.float32)
    logger_wrapper.log_embeddings("test", embeddings, metadata=metadata, images=images, step=step)
    if called:
        add_embedding.assert_called_once()
    else:
        add_embedding.assert_not_called()


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("step,decimation,called", [
    (0, 1, True),
    (1, 1, True),
    (2, 1, True),
    (None, 1,True),
    (1, 2, False),
])
def test_logger_wrapper_log_histogram_tensorboard(tensorboard_logger, log, step, decimation, called, mocker):
    called = called and log
    logger_config = LoggerConfig(decimation=decimation, log_histograms=log)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    add_histogram = mocker.spy(tensorboard_logger.experiment, 'add_histogram')
    histogram = torch.ones((2, 5), dtype=torch.float32)
    logger_wrapper.log_histogram("test", histogram, step=step)
    if called:
        add_histogram.assert_called_once()
    else:
        add_histogram.assert_not_called()
    