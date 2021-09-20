#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
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
