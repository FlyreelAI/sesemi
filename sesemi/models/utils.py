#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Utilities to download and prepare models."""
import torch

from typing import Optional

TORCH_HUB_DOWNLOAD_MAX_RETRIES = 100


def load_torch_hub_model(repo: str, model: str, *args, **kwargs):
    """Tries to load a torch hub model and handles different exceptions that could be raised.

    Args:
        repo: The GitHub repository containing the models.
        model: The model name to download.
        max_retries: The maximum number of tries to download the model.

    Returns:
        The downloaded torch model.
    """
    error: Optional[Exception] = None
    for _ in range(TORCH_HUB_DOWNLOAD_MAX_RETRIES + 1):
        try:
            try:
                return torch.hub.load(
                    repo,
                    model,
                    *args,
                    **kwargs,
                )
            except RuntimeError:
                return torch.hub.load(
                    repo,
                    model,
                    *args,
                    **kwargs,
                    force_reload=True,
                )
        except Exception as e:
            error = e

    assert error is not None
    raise error
