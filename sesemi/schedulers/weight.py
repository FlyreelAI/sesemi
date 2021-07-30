#
# Copyright 2021, Flyreel. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================#
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
