"""Loss function factory for training."""

from __future__ import annotations

import torch.nn as nn

__all__ = ["get_loss"]


def get_loss(name: str, *, reduction: str = "none") -> nn.Module:
    """Return a loss function by name.

    Parameters
    ----------
    name:
        Identifier for the desired loss. Supported values are ``"mse"`` and
        ``"huber"``.
    reduction:
        Reduction mode passed to the underlying loss constructor.  The default
        ``"none"`` allows the training loop to apply custom weighting before
        averaging.
    """

    name = name.lower()
    if name in {"mse", "l2"}:
        return nn.MSELoss(reduction=reduction)
    if name in {"huber", "smoothl1"}:
        return nn.SmoothL1Loss(reduction=reduction)
    raise ValueError(f"Unknown loss function: {name}")
