"""Custom sampling strategies for patch datasets."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

__all__ = ["BalancedSampler"]


class BalancedSampler(Sampler[int]):
    """Sampler yielding a balanced mixture of classes.

    The sampler expects the dataset to expose ``class_lengths`` and ``_cumulative``
    attributes as provided by :class:`tvs.data.dataset.PatchDataset`.  During
    iteration it cycles through classes so that each contributes an equal number
    of samples, repeating samples from minority classes if necessary.

    Parameters
    ----------
    dataset:
        The :class:`PatchDataset` to sample from.
    shuffle:
        If ``True`` (default) the order of samples within each class is
        randomized on every epoch.
    """

    def __init__(self, dataset, *, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.class_lengths = list(dataset.class_lengths.values())
        self.cumulative = dataset._cumulative  # type: ignore[attr-defined]
        self.max_len = max(self.class_lengths) if self.class_lengths else 0
        self.num_classes = len(self.class_lengths)

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng()
        class_indices: list[np.ndarray] = []
        for length in self.class_lengths:
            idxs = np.arange(length)
            if self.shuffle:
                rng.shuffle(idxs)
            class_indices.append(idxs)

        for i in range(self.max_len):
            for class_idx, idxs in enumerate(class_indices):
                local_idx = idxs[i % len(idxs)]
                base = self.cumulative[class_idx - 1] if class_idx > 0 else 0
                yield int(base + local_idx)

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 - terse
        """Total number of samples that will be drawn."""

        return self.max_len * self.num_classes
