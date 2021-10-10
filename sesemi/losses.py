"""Loss functions and modules."""
import torch.nn.functional as F

from torch import Tensor


def simsiam_loss(proj1: Tensor, proj2: Tensor) -> Tensor:
    """Computes the SimSiam loss.
    
    Args:
        proj1: The projection of one view.
        proj2: The projection of a second view.

    Returns:
        The negative cosine similarity between the two views.
    """
    return -F.cosine_similarity(proj1, proj2)