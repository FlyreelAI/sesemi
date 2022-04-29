import pytest
import torch

from sesemi.models.heads import LinearHead

from ...utils import check_model_frozen_state


@pytest.mark.parametrize("in_features", [1, 10])
@pytest.mark.parametrize("out_features", [20, 5])
@pytest.mark.parametrize("freeze", [False, True])
def test_linear_head_freeze(in_features, out_features, freeze):
    head = LinearHead(in_features=in_features, out_features=out_features, freeze=freeze)
    inputs = torch.ones((2, in_features))
    outputs = head(inputs)
    assert outputs.shape == (2, out_features)
    assert head.in_features == in_features
    assert head.out_features == out_features

    check_model_frozen_state(
        head,
        freeze,
        input_shape=(2, in_features),
    )
