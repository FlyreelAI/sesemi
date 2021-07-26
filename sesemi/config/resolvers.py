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
"""Omegaconf resolvers."""
from typing import Optional


class AttributeResolver:
    """An omegaconf resolver that does attribute lookups on an object."""

    def __call__(self, name: str):
        """Looks up the attribute with the given name."""
        return getattr(self, name)


class SESEMIConfigAttributes(AttributeResolver):
    """The attributes exposed to SESEMI configuration files.

    These attributes can be referenced in the config files by following the omegaconf syntax for
    custom resolvers. For example, ${sesemi:iterations_per_epoch} will reference the
    `iterations_per_epoch` attribute.

    Attributes:
        iterations_per_epoch: The number of training iterations per epoch if training data is
            available.
        max_iterations: The maximum number of training iterations if training data is available.
        num_gpus: The number of GPUs that will be used.
        num_nodes: The number of compute nodes that will be used.
    """

    iterations_per_epoch: Optional[int]
    max_iterations: Optional[int]
    num_gpus: int
    num_nodes: int
