from typing import Optional


class AttributeResolver:
    def __call__(self, name: str):
        return getattr(self, name)


class SESEMIConfigAttributes(AttributeResolver):
    iterations_per_epoch: Optional[int]
    max_iterations: Optional[int]
    num_gpus: int
    num_nodes: int
