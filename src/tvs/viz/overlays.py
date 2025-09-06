"""Utilities for visualising score maps as colour overlays."""

from __future__ import annotations

import numpy as np
from matplotlib import cm

__all__ = ["score_to_color", "overlay_scores"]


def score_to_color(
    scores: np.ndarray,
    *,
    cmap: str = "bwr",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """Map a score array to RGB colours using a diverging colormap.

    Parameters
    ----------
    scores:
        Array of scores in the range ``[vmin, vmax]``.
    cmap:
        Matplotlib colormap name used for mapping.
    vmin, vmax:
        Bounds for normalisation.  Values outside this range are clipped.
    """

    if scores.ndim != 2:
        raise ValueError("`scores` must be a 2D array")
    normed = np.clip((scores - vmin) / (vmax - vmin), 0.0, 1.0)
    m_cmap = cm.get_cmap(cmap)
    rgba = m_cmap(normed)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def overlay_scores(
    image: np.ndarray,
    scores: np.ndarray,
    *,
    alpha: float = 0.5,
    cmap: str = "bwr",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """Overlay a colour-mapped score array on top of an image.

    Both ``image`` and ``scores`` must have the same spatial dimensions.
    The image is assumed to be in RGB format.
    """

    if image.shape[:2] != scores.shape:
        raise ValueError("Image and scores must have the same height and width")
    color = score_to_color(scores, cmap=cmap, vmin=vmin, vmax=vmax)
    base = image.astype(np.float32)
    overlay = alpha * color.astype(np.float32) + (1.0 - alpha) * base
    return overlay.astype(np.uint8)
