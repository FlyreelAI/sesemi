#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Weight schedulers."""
from ..utils import sigmoid_rampup


class WeightScheduler:
    """The interface for weight schedulers."""

    def __call__(self, step: int) -> float:
        """Computes the weight for the given step number.

        Attributes:
            step: The step number.

        Returns:
            The weight.
        """
        raise NotImplementedError()


class SigmoidRampupScheduler(WeightScheduler):
    """A sigmoid ramp-up weight scheduler."""

    def __init__(self, weight: float, stop_rampup: int):
        """Initializes the weight scheduler.

        Args:
            weight: The weight to apply to the sigmoidal function.
            stop_rampup: The step to stop ramping up the output weight.
        """
        self.weight = weight
        self.stop_rampup = stop_rampup

    def __call__(self, step: int) -> float:
        return self.weight * sigmoid_rampup(step, self.stop_rampup)


class SigmoidRampdownScheduler(WeightScheduler):
    """A sigmoid ramp-down weight scheduler."""

    def __init__(self, weight: float, stop_rampdown: int):
        """Initializes the weight scheduler.

        Args:
            weight: The weight to apply to the sigmoidal function.
            stop_rampdown: The step to stop ramping down the output weight.
        """
        self.weight = weight
        self.stop_rampdown = stop_rampdown

    def __call__(self, step: int) -> float:
        return self.weight * (1.0 - sigmoid_rampup(step, self.stop_rampdown))
