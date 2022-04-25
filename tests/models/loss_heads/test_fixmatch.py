import pytest
import torch

from sesemi.logger import LoggerWrapper
from sesemi.config.structs import LoggerConfig
from sesemi.models.loss_heads.fixmatch import FixMatchLossHead

from .utils import initialize_loss_head_mock_inputs


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("step", [0, 1])
@pytest.mark.parametrize("data", ["test1", "test2"])
@pytest.mark.parametrize("student_backbone", ["supervised_backbone", "test_backbone"])
@pytest.mark.parametrize(
    "teacher_backbone", ["supervised_backbone_ema", "test_backbone"]
)
@pytest.mark.parametrize("student_head", ["supervised_head", "test_head"])
@pytest.mark.parametrize("teacher_head", ["supervised_head_ema", "test_head"])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_fixmatch_loss_head(
    tensorboard_logger,
    batch_size,
    step,
    data,
    student_backbone,
    teacher_backbone,
    student_head,
    teacher_head,
    threshold,
):
    mock_inputs = initialize_loss_head_mock_inputs(
        backbones=[student_backbone, teacher_backbone],
        heads=[student_head, teacher_head],
    )
    mock_inputs["data"][data] = (
        torch.ones((batch_size, 3, 32, 32)),
        torch.ones((batch_size, 3, 32, 32)),
    )

    logger_config = LoggerConfig(decimation=1)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    loss_head = FixMatchLossHead(
        data=data,
        student_backbone=student_backbone,
        teacher_backbone=teacher_backbone,
        student_head=student_head,
        teacher_head=teacher_head,
        threshold=threshold,
    )
    loss_head.build(backbones=mock_inputs["backbones"], heads=mock_inputs["heads"])
    loss_outputs = loss_head(**mock_inputs, step=step, logger_wrapper=logger_wrapper)

    assert loss_outputs.losses.shape == (batch_size,)
    assert loss_outputs.weights is not None
    assert torch.all(loss_outputs.weights >= 0).item()
