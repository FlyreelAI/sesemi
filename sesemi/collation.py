import torch


class RotationTransformer:
    def __init__(self):
        self.num_rotation_labels = 4

    def __call__(self, batch):
        tensors, labels = [], []
        for tensor, _ in batch:
            for k in range(self.num_rotation_labels):
                if k == 0:
                    t = tensor
                else:
                    t = torch.rot90(tensor, k, dims=[1, 2])
                tensors.append(t)
                labels.append(torch.LongTensor([k]))
        x = torch.stack(tensors, dim=0)
        y = torch.cat(labels, dim=0)
        return (x, y)
