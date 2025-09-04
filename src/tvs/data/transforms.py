"""Albumentations-based transform pipelines."""

from __future__ import annotations

from typing import Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = ["build_transforms"]


def build_transforms(
    *,
    train: bool,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> A.Compose:
    """Create an Albumentations transform pipeline.

    Parameters
    ----------
    train:
        If ``True`` a training/augmentation pipeline is returned.  Otherwise a
        minimal validation pipeline is produced.
    mean, std:
        Optional normalisation statistics.  Defaults to ImageNet values.
    """

    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)

    if train:
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    else:
        transforms = [A.Normalize(mean=mean, std=std), ToTensorV2()]

    return A.Compose(transforms)
