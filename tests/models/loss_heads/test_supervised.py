import pytest
import torch

from sesemi.logger import LoggerWrapper
from sesemi.config.structs import LoggerConfig
from sesemi.models.loss_heads.supervised import SupervisedLossHead

from .utils import initialize_loss_head_mock_inputs


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("step", [0, 1])
@pytest.mark.parametrize(
    "loss_fn",
    [torch.nn.CrossEntropyLoss(reduction="none")],
)
@pytest.mark.parametrize("data", ["test", "entropy_minimization"])
@pytest.mark.parametrize("logits", ["supervised_head"])
@pytest.mark.parametrize("class_weights", [None, [1] * 5, [0, 1, 2, 3, 4]])
def test_supervised_loss_head(
    tensorboard_logger, batch_size, step, loss_fn, data, logits, class_weights
):
    mock_inputs = initialize_loss_head_mock_inputs(
        backbones=["supervised_backbone"], num_classes=5
    )
    mock_inputs["data"][data] = (
        torch.ones((batch_size, 3, 32, 32)),
        torch.ones((batch_size,), dtype=torch.long),
    )
    mock_inputs["features"] = {
        "supervised_backbone": torch.ones(
            (batch_size, mock_inputs["backbones"]["supervised_backbone"].out_features)
        ),
        "supervised_head": torch.ones(
            (batch_size, mock_inputs["heads"]["supervised_head"].out_features)
        ),
        "supervised_probabilities": torch.ones(
            (batch_size, mock_inputs["heads"]["supervised_head"].out_features)
        ),
        "supervised_backbone_ema": torch.ones(
            (
                batch_size,
                mock_inputs["backbones"]["supervised_backbone_ema"].out_features,
            )
        ),
        "supervised_head_ema": torch.ones(
            (batch_size, mock_inputs["heads"]["supervised_head_ema"].out_features)
        ),
        "supervised_probabilities_ema": torch.ones(
            (batch_size, mock_inputs["heads"]["supervised_head_ema"].out_features)
        ),
    }

    logger_config = LoggerConfig(decimation=1)
    logger_wrapper = LoggerWrapper(tensorboard_logger, logger_config)

    loss_head = SupervisedLossHead(
        loss_fn=loss_fn, data=data, logits=logits, class_weights=class_weights
    )
    loss_head.build(backbones=mock_inputs["backbones"], heads=mock_inputs["heads"])
    loss_outputs = loss_head(**mock_inputs, step=step, logger_wrapper=logger_wrapper)

    assert loss_outputs.losses.shape == (batch_size,)
    if class_weights is not None:
        assert loss_outputs.weights is not None
    else:
        assert loss_outputs.weights is None
