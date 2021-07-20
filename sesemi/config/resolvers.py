class AttributeResolver:
    def __call__(self, name: str):
        return getattr(self, name)


class SESEMIConfigAttributes(AttributeResolver):
    iterations_per_epoch: int
    max_iterations: int
    num_gpus: int
    num_nodes: int
