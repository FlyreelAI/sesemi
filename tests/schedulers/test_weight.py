import pytest
import numpy as np

from sesemi.schedulers.weight import SigmoidRampupScheduler


@pytest.mark.parametrize(
    "weight,stop_rampup,step,expected",
    [
        (1.0, 1000, 0, float(np.exp(-5.0))),
        (1.0, 1000, 500, float(np.exp(-5.0 * 0.5 * 0.5))),
        (1.0, 1000, 1000, 1.0),
    ],
)
def test_sigmoid_rampup_scheduler(weight, stop_rampup, step, expected):
    scheduler = SigmoidRampupScheduler(weight, stop_rampup)
    assert np.allclose(scheduler(step), expected)
