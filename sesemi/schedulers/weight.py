from ..utils import sigmoid_rampup


class WeightScheduler:
    def __call__(self, global_step: int) -> float:
        raise NotImplementedError()


class SigmoidRampupScheduler(WeightScheduler):
    def __init__(self, initial_weight, stop_rampup):
        self.initial_weight = initial_weight
        self.stop_rampup = stop_rampup

    def __call__(self, global_step: int) -> float:
        return self.initial_weight * sigmoid_rampup(global_step, self.stop_rampup)
