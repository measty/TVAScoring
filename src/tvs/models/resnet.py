"""ResNet-18 backbone with simple regression head.

The :class:`ResNet18Baseline` wraps a torchvision `resnet18`
model, optionally initialised with ImageNet pretrained weights.
The final fully connected layer is replaced by a single linear
layer followed by ``tanh`` to constrain outputs to ``[-1, 1]``.
This provides a stronger baseline compared to the small
:class:`~tvs.models.cnn_baseline.CNNBaseline`.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

__all__ = ["ResNet18Baseline"]


class ResNet18Baseline(nn.Module):
    """ResNet-18 backbone with a ``tanh`` regression head.

    Parameters
    ----------
    pretrained:
        If ``True`` (default), load ImageNet pretrained weights.
    finetune_backbone:
        If ``True``, trains backbone parameters. Defaults to ``True``.
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        finetune_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Tanh(),
        )
        if not finetune_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - short
        """Compute forward pass returning scores in ``[-1, 1]``."""

        x = self.backbone(x)
        x = self.head(x)
        return x.squeeze(-1)
