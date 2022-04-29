import pytest
import torch

from sesemi.logger import LoggerWrapper
from sesemi.config.structs import LoggerConfig
from sesemi.models.loss_heads.jigsaw_prediction import JigsawPredictionLossHead

from .utils import initialize_loss_head_mock_inputs


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("step", [0, 1])
@pytest.mark.parametrize("data", ["test", "entropy_minimization"])
@pytest.mark.parametrize("backbone", ["supervised_backbone", "test_backbone"])
def test_jigsaw_prediction_loss_head(
    tensorboard_logger, batch_size, step, data, backbone
):
    mock_inputs = initialize_loss_head_mock_inputs(backbones=[backbone])
    mock_inputs["data"][data] = (
        torch.ones((batch_size, 3, 32, 32)),
        torch.ones((batch_size,), dtype=torch.long),
    )

    logger_config = LoggerConfig(decimation=1)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    loss_head = JigsawPredictionLossHead(data=data, backbone=backbone)
    loss_head.build(backbones=mock_inputs["backbones"], heads=mock_inputs["heads"])
    loss_outputs = loss_head(**mock_inputs, step=step, logger_wrapper=logger_wrapper)

    assert loss_outputs.losses.shape == (batch_size,)
    assert loss_outputs.weights is None
