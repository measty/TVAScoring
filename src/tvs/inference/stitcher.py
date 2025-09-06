"""Utilities for stitching patch-level scores into a heatmap."""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["stitch_scores"]


def stitch_scores(
    coords: Sequence[tuple[int, int, int, int]],
    scores: Sequence[float],
    image_shape: tuple[int, int],
    scale: float = 1.0,
) -> np.ndarray:
    """Blend patch scores into a dense heatmap via averaging.

    Parameters
    ----------
    coords:
        Sequence of ``(x, y, w, h)`` patch coordinates at the extraction
        resolution.
    scores:
        Sequence of scalar scores for each patch.
    image_shape:
        ``(height, width)`` of the output heatmap.
    scale:
        Factor by which to multiply patch coordinates and sizes to project them
        onto the heatmap canvas. For example, when scoring patches extracted at
        ``1.0`` mpp and stitching to a ``6.0`` mpp heatmap this would be
        ``1.0/6.0``.

    Returns
    -------
    np.ndarray
        Heatmap with the same shape as ``image_shape`` containing averaged
        scores where patches contributed and zeros elsewhere.
    """
    heatmap = np.zeros(image_shape, dtype=float)
    counts = np.zeros(image_shape, dtype=float)

    for (x, y, w, h), score in zip(coords, scores):
        sx = int(round(x * scale))
        sy = int(round(y * scale))
        sw = int(round(w * scale))
        sh = int(round(h * scale))
        heatmap[sy : sy + sh, sx : sx + sw] += score
        counts[sy : sy + sh, sx : sx + sw] += 1

    mask = counts > 0
    heatmap[mask] /= counts[mask]
    return heatmap
