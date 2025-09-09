"""PyTorch dataset wrapping class-specific patch extractors.

This module provides :class:`PatchDataset` which combines several
``SlidingWindowPatchExtractor`` instances—typically one per annotation class—into
a single :class:`torch.utils.data.Dataset`. Each extractor yields patches from a
specific class and the dataset exposes them with corresponding regression
scores.

The dataset expects a mapping of ``label -> sequence of extractors`` and a
mapping of ``label -> score``. When indexed it returns a tuple ``(image,
target)`` where ``image`` is a ``torch.Tensor`` and ``target`` is a float tensor
containing the score associated with the patch's class. Optional transforms
(Albumentations or torchvision) can be applied to the raw ``numpy`` images prior
to tensor conversion.
"""

from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
from typing import Any, Mapping, Sequence

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
        Mapping from class label to a sequence of
        :class:`SlidingWindowPatchExtractor` instances.
    label_scores:
        Mapping from class label to regression score (e.g. ``{"villous": 1.0}``).
    transform:
        Optional transform applied to the raw patch image.
    label_weights:
        Optional mapping of class label to a scalar weight.  When provided the
        weight corresponding to the patch's class is appended to the returned
        sample and can be used to modulate the loss contribution for that
        example.
    return_coords:
        If ``True`` the extractor's coordinates are returned in addition to the
        image and target.  This can be useful for debugging or visualization.
    """

    def __init__(
        self,
        extractors: Mapping[str, Sequence[SlidingWindowPatchExtractor]],
        label_scores: Mapping[str, float],
        *,
        transform: Any | None = None,
        label_weights: Mapping[str, float] | None = None,
        return_coords: bool = False,
    ) -> None:
        self.labels: list[str] = list(extractors.keys())
        self.extractors: list[list[SlidingWindowPatchExtractor]] = [
            list(extractors[label]) for label in self.labels
        ]
        self.scores = {label: float(label_scores[label]) for label in self.labels}
        self.transform = transform
        self.return_coords = return_coords
        if label_weights is not None:
            self.weights = {
                label: float(label_weights.get(label, 1.0)) for label in self.labels
            }
        else:
            self.weights = None

        # lengths per extractor and per class
        self._extractor_lengths: list[list[int]] = [
            [len(ex) for ex in ex_list] for ex_list in self.extractors
        ]
        self._class_lengths: list[int] = [
            sum(lengths) for lengths in self._extractor_lengths
        ]
        self._cumulative: list[int] = list(accumulate(self._class_lengths))
        self._total_length = self._cumulative[-1] if self._cumulative else 0
        self._extractor_cumulatives: list[list[int]] = [
            list(accumulate(lengths)) for lengths in self._extractor_lengths
        ]

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 - short and clear
        """Return the total number of available patches."""

        return self._total_length

    # ------------------------------------------------------------------
    def _locate(self, index: int) -> tuple[int, int, int]:
        """Map a global index to ``(class_idx, extractor_idx, patch_idx)``."""

        class_idx = bisect_right(self._cumulative, index)
        prev_cum = self._cumulative[class_idx - 1] if class_idx > 0 else 0
        idx_within_class = index - prev_cum
        extractor_cum = self._extractor_cumulatives[class_idx]
        extractor_idx = bisect_right(extractor_cum, idx_within_class)
        prev_ex_cum = extractor_cum[extractor_idx - 1] if extractor_idx > 0 else 0
        patch_idx = idx_within_class - prev_ex_cum
        return class_idx, extractor_idx, patch_idx

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        class_idx, extractor_idx, patch_idx = self._locate(index)
        label = self.labels[class_idx]
        extractor = self.extractors[class_idx][extractor_idx]

        patch = extractor[patch_idx]  # numpy array
        if isinstance(patch, tuple):
            # SlidingWindowPatchExtractor may return (image, coords)
            img, coords = patch
        else:
            img, coords = patch, None

        if self.transform is not None:
            try:
                img = self.transform(image=img)["image"]
            except TypeError:
                img = self.transform(img)
        else:
            # default conversion to tensor
            img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)

        target = torch.tensor(self.scores[label], dtype=torch.float32)
        items = [img, target]
        if self.return_coords:
            items.append(coords)
        if self.weights is not None:
            items.append(torch.tensor(self.weights[label], dtype=torch.float32))
        return tuple(items)

    # ------------------------------------------------------------------
    @property
    def class_lengths(self) -> Mapping[str, int]:
        """Number of patches contributed by each class."""

        return {
            label: length for label, length in zip(self.labels, self._class_lengths)
        }
