"""PyTorch dataset wrapping class-specific patch extractors.

This module provides :class:`PatchDataset` which combines several
``SlidingWindowPatchExtractor`` instances—typically one per annotation class—into a
single :class:`torch.utils.data.Dataset`.  Each extractor yields patches from a
specific class and the dataset exposes them with corresponding regression
scores.

The dataset expects a mapping of ``label -> extractor`` and a mapping of
``label -> score``.  When indexed it returns a tuple ``(image, target)`` where
``image`` is a ``torch.Tensor`` and ``target`` is a float tensor containing the
score associated with the patch's class.  Optional Albumentations transforms can
be applied to the raw ``numpy`` images prior to tensor conversion.
"""

from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
from typing import Any, Mapping

import numpy as np
import torch
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from torch.utils.data import Dataset

__all__ = ["PatchDataset"]


class PatchDataset(Dataset):
    """Dataset merging multiple class-conditioned extractors.

    Parameters
    ----------
    extractors:
        Mapping from class label to :class:`SlidingWindowPatchExtractor`.
    label_scores:
        Mapping from class label to regression score (e.g. ``{"villous": 1.0}``).
    transform:
        Optional Albumentations transform applied to the raw patch image.
    return_coords:
        If ``True`` the extractor's coordinates are returned in addition to the
        image and target.  This can be useful for debugging or visualization.
    """

    def __init__(
        self,
        extractors: Mapping[str, SlidingWindowPatchExtractor],
        label_scores: Mapping[str, float],
        *,
        transform: Any | None = None,
        return_coords: bool = False,
    ) -> None:
        self.labels: list[str] = list(extractors.keys())
        self.extractors: list[SlidingWindowPatchExtractor] = [
            extractors[label] for label in self.labels
        ]
        self.scores = {label: float(label_scores[label]) for label in self.labels}
        self.transform = transform
        self.return_coords = return_coords

        self._lengths: list[int] = [len(ex) for ex in self.extractors]
        self._cumulative: list[int] = list(accumulate(self._lengths))
        self._total_length = self._cumulative[-1] if self._cumulative else 0

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 - short and clear
        """Return the total number of available patches."""

        return self._total_length

    # ------------------------------------------------------------------
    def _locate(self, index: int) -> tuple[int, int]:
        """Map a global index to ``(extractor_idx, patch_idx)``."""

        extractor_idx = bisect_right(self._cumulative, index)
        prev_cum = self._cumulative[extractor_idx - 1] if extractor_idx > 0 else 0
        patch_idx = index - prev_cum
        return extractor_idx, patch_idx

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        extractor_idx, patch_idx = self._locate(index)
        label = self.labels[extractor_idx]
        extractor = self.extractors[extractor_idx]

        patch = extractor[patch_idx]  # numpy array
        if isinstance(patch, tuple):
            # SlidingWindowPatchExtractor may return (image, coords)
            img, coords = patch
        else:
            img, coords = patch, None

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            # default conversion to tensor
            img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)

        target = torch.tensor(self.scores[label], dtype=torch.float32)

        if self.return_coords:
            return img, target, coords
        return img, target

    # ------------------------------------------------------------------
    @property
    def class_lengths(self) -> Mapping[str, int]:
        """Number of patches contributed by each class."""

        return {label: length for label, length in zip(self.labels, self._lengths)}
