import pytest
import torch
import torch.nn as nn

from sesemi.logger import LoggerWrapper
from sesemi.config.structs import LoggerConfig
from sesemi.models.loss_heads.consistency import (
    ConsistencyLossHead,
    EMAConsistencyLossHead,
)

from .utils import initialize_loss_head_mock_inputs


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("step", [0, 1])
@pytest.mark.parametrize("data", ["test1", "test2"])
@pytest.mark.parametrize("backbone", ["supervised_backbone", "consistency_backbone"])
@pytest.mark.parametrize("head", ["supervised_head", "consistency_head"])
@pytest.mark.parametrize("loss_fn", ["softmax_mse_loss", "kl_div_loss"])
def test_consistency_loss_head(
    tensorboard_logger, batch_size, step, data, backbone, head, loss_fn
):
    mock_inputs = initialize_loss_head_mock_inputs(backbones=[backbone], heads=[head])
    mock_inputs["data"][data] = (
        (torch.ones((batch_size, 3, 32, 32)), torch.ones((batch_size, 3, 32, 32))),
        torch.ones((batch_size,), dtype=torch.long),
    )

    logger_config = LoggerConfig(decimation=1)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    loss_head = ConsistencyLossHead(
        data=data, backbone=backbone, head=head, loss_fn=loss_fn
    )
    loss_outputs = loss_head(**mock_inputs, step=step, logger_wrapper=logger_wrapper)

    assert loss_outputs.losses.shape == (batch_size,)
    assert loss_outputs.weights is None


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("step", [0, 1])
@pytest.mark.parametrize("data", ["test1", "test2"])
@pytest.mark.parametrize(
    "student_backbone", ["supervised_backbone", "consistency_backbone"]
)
@pytest.mark.parametrize(
    "teacher_backbone", ["supervised_backbone_ema", "consistency_backbone"]
)
@pytest.mark.parametrize("student_head", ["supervised_head", "consistency_head"])
@pytest.mark.parametrize("teacher_head", ["supervised_head_ema", "consistency_head"])
@pytest.mark.parametrize("loss_fn", ["softmax_mse_loss", "kl_div_loss"])
def test_ema_consistency_loss_head(
    tensorboard_logger,
    batch_size,
    step,
    data,
    student_backbone,
    teacher_backbone,
    student_head,
    teacher_head,
    loss_fn,
):
    mock_inputs = initialize_loss_head_mock_inputs(
        backbones=[student_backbone, teacher_backbone],
        heads=[student_head, teacher_head],
    )
    mock_inputs["data"][data] = (
        (torch.ones((batch_size, 3, 32, 32)), torch.ones((batch_size, 3, 32, 32))),
        torch.ones((batch_size,), dtype=torch.long),
    )

    logger_config = LoggerConfig(decimation=1)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    loss_head = EMAConsistencyLossHead(
        data=data,
        student_backbone=student_backbone,
        teacher_backbone=teacher_backbone,
        student_head=student_head,
        teacher_head=teacher_head,
        loss_fn=loss_fn,
    )
    loss_head.build(backbones=mock_inputs["backbones"], heads=mock_inputs["heads"])
    loss_outputs = loss_head(**mock_inputs, step=step, logger_wrapper=logger_wrapper)

    assert loss_outputs.losses.shape == (batch_size,)
    assert loss_outputs.weights is None
