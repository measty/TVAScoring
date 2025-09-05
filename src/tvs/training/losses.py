"""Loss function factory for training."""

from __future__ import annotations

import torch.nn as nn

__all__ = ["get_loss"]


def get_loss(name: str) -> nn.Module:
    """Return a loss function by name.

    Parameters
    ----------
    name:
        Identifier for the desired loss. Supported values are ``"mse"`` and
        ``"huber"``.
    """

    name = name.lower()
    if name in {"mse", "l2"}:
        return nn.MSELoss()
    if name in {"huber", "smoothl1"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss function: {name}")
