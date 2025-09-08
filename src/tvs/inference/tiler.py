"""Patch tiling over whole-slide images for inference."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import tiatoolbox.tools.patchextraction as patchextraction
from tiatoolbox.wsicore.wsireader import WSIReader

__all__ = ["tile_wsi"]


def tile_wsi(
    wsi: WSIReader | str,
    *,
    masker: str | None = None,
    patch_size: int,
    stride: int | None = None,
    min_mask_ratio: float = 0.5,
    resolution: float = 0,
    units: str = "mpp",
) -> Iterator[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Yield tiled patches and their coordinates from a WSI.

    Parameters
    ----------
    wsi:
        Whole-slide image or a path to one. If a path is provided it is opened
        via :func:`WSIReader.open`.
    masker:
        Optional masking strategy restricting patch locations. Can be an
        annotation store path, ``None`` for no restriction, or ``"otsu"`` to
        generate a tissue mask on-the-fly.
    patch_size:
        Size of patches to extract in pixels.
    stride:
        Step size between successive patches. Defaults to ``patch_size`` when
        ``None``.
    min_mask_ratio:
        Minimum fraction of a patch that must lie within the mask for it to be
        yielded.
    resolution, units:
        Resolution at which to extract patches and the associated units. Passed
        through to :class:`~tiatoolbox.tools.patchextraction.SlidingWindowPatchExtractor`.

    Yields
    ------
    tuple
        ``((x, y, w, h), patch)`` where ``(x, y)`` is the top-left coordinate
        of the patch at the extraction resolution and ``(w, h)`` are the patch
        dimensions. ``patch`` is the RGB image as a :class:`numpy.ndarray`.
    """
    if not isinstance(wsi, WSIReader):
        wsi = WSIReader.open(wsi)
    if stride is None:
        stride = patch_size

    extractor = patchextraction.SlidingWindowPatchExtractor(
        input_img=wsi,
        input_mask=masker,
        patch_size=patch_size,
        stride=stride,
        within_bound=False,
        pad_mode="constant",
        pad_constant_values=0,
        min_mask_ratio=min_mask_ratio,
        resolution=resolution,
        units=units,
    )

    for coord, patch in zip(extractor.coordinate_list, extractor):
        x1, y1, x2, y2 = coord
        yield (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), patch
