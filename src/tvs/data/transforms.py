"""Transform utilities including StrongAugment support.

This module provides a small transformation building utility that can
optionally apply `strong-augment <https://github.com/jopo666/StrongAugment>`_
operations.  Basic geometric augmentations (flips and 90° rotations) are
implemented as a lightweight :class:`BasicAugment` transform which operates on
``torch.Tensor`` images.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
from strong_augment import StrongAugment, get_augment_space
from torchvision import transforms
from torchvision.transforms import functional as F

__all__ = ["build_transforms", "BasicAugment"]


class BasicAugment(torch.nn.Module):
    """Simple flip, rotation and optional colour jitter augmentations.

    This transform expects a ``torch.Tensor`` image in ``CxHxW`` format and
    applies random horizontal/vertical flips and a random rotation chosen from
    ``{0°, 90°, 180°, 270°}``.  If ``jitter`` is enabled, a small colour jitter
    is applied after the geometric transformations.

    Parameters
    ----------
    jitter:
        If ``True`` a small :class:`~torchvision.transforms.ColorJitter` is
        applied after the flips and rotations.
    """

    def __init__(self, *, jitter: bool = False) -> None:
        super().__init__()
        self.jitter = jitter
        self._jitter = (
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            )
            if jitter
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - short
        if random.random() < 0.5:
            x = F.hflip(x)
        if random.random() < 0.5:
            x = F.vflip(x)
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, (1, 2))
        if self._jitter is not None:
            x = self._jitter(x)
        return x


def _strong_augment() -> StrongAugment:
    """Create a :class:`StrongAugment` instance with a sane default space."""

    augment_space = get_augment_space()
    # Solarize tends to produce unrealistic patches in this domain.
    augment_space.pop("solarize", None)
    return StrongAugment(
        operations=(0, 2, 3, 4),
        probabilites=(0.3, 0.3, 0.3, 0.1),
        augment_space=augment_space,
    )


def build_transforms(
    *,
    train: bool,
    augmentation: str | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> transforms.Compose:
    """Create a torchvision transform pipeline.

    Parameters
    ----------
    train:
        If ``True`` a training/augmentation pipeline is returned.  Otherwise a
        minimal validation pipeline is produced.
    augmentation:
        Name of the augmentation strategy. ``"strong"`` enables StrongAugment
        in addition to the basic flips/rotations; ``"none"`` disables all
        augmentation.  Any other value defaults to the basic pipeline.
    mean, std:
        Optional normalisation statistics.  Defaults to ImageNet values.
    """

    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)

    tfms: list[transforms.Compose | torch.nn.Module] = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if train:
        aug_name = (augmentation or "basic").lower()
        if aug_name not in {"basic", "strong", "none"}:
            raise ValueError(f"Unknown augmentation strategy: {augmentation}")
        if aug_name != "none":
            # Insert basic augment after ToTensor and before normalisation.
            tfms.insert(1, BasicAugment(jitter=aug_name == "basic"))
        if aug_name == "strong":
            tfms = [_strong_augment()] + tfms

    return transforms.Compose(tfms)
