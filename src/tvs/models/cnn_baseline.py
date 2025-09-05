"""Compact convolutional baseline model.

The :class:`CNNBaseline` is a small convolutional network with three
``Conv2d`` blocks followed by global average pooling and a two layer
MLP.  A ``tanh`` activation ensures outputs lie in ``[-1, 1]``.
This simple architecture serves as a lightweight baseline for
TubuloVillous scoring experiments.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

__all__ = ["CNNBaseline"]


class CNNBaseline(nn.Module):
    """Small CNN with global pooling and ``tanh`` output.

    Parameters
    ----------
    in_channels:
        Number of input channels.  Default is ``3`` for RGB patches.
    conv_channels:
        Sequence specifying the number of channels for each convolutional
        block.  Defaults to ``(16, 32, 64)``.
    mlp_hidden:
        Hidden dimensionality of the MLP head.  Defaults to ``32``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        conv_channels: Sequence[int] = (16, 32, 64),
        mlp_hidden: int = 32,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for out_ch in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(prev, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            prev = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(prev, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1),
        )
        self.activation = nn.Tanh()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - short
        """Compute forward pass returning scores in ``[-1, 1]``."""

        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        x = self.activation(x)
        return x.squeeze(-1)
