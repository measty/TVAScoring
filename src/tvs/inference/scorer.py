"""High-level utilities for scoring whole-slide images."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from tiatoolbox.wsicore.wsireader import WSIReader

from .stitcher import stitch_scores
from .tiler import tile_wsi

__all__ = ["score_wsi"]


def _infer_batch(
    model: torch.nn.Module,
    patches: Sequence[np.ndarray],
    device: torch.device,
) -> list[float]:
    """Run model inference on a batch of patches."""
    if not patches:
        return []
    tensor = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        preds = model(tensor.to(device))
    preds = preds.reshape(len(patches))
    return preds.detach().cpu().numpy().tolist()


def score_wsi(
    model: torch.nn.Module,
    wsi: WSIReader | str,
    annotation_store: str | None = None,
    *,
    patch_size: int,
    stride: int | None = None,
    batch_size: int = 16,
    min_mask_ratio: float = 0.5,
    resolution: float = 0,
    units: str = "mpp",
    heatmap_mpp: float = 6.0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Score a whole-slide image with a patch-level model.

    Parameters
    ----------
    model:
        PyTorch model returning a scalar score in ``[-1, 1]`` for each patch.
    wsi:
        Whole-slide image or path to one.
    annotation_store:
        Optional annotation store path restricting inference to specific
        regions.
    patch_size, stride, min_mask_ratio:
        Passed through to :func:`tile_wsi`.
    resolution, units:
        Resolution and units used for patch extraction. ``units`` must be
        ``"mpp"``.
    heatmap_mpp:
        Output resolution for the stitched heatmap in microns-per-pixel.
    batch_size:
        Number of patches per forward pass.
    device:
        Device for model execution. Defaults to CPU (or CUDA when available).

    Returns
    -------
    np.ndarray
        Heatmap of scores with shape matching the slide dimensions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if not isinstance(wsi, WSIReader):
        wsi = WSIReader.open(wsi)
    if resolution == 0:
        resolution = float(wsi.info.mpp[0])
    if units != "mpp":  # pragma: no cover - sanity check
        raise ValueError("score_wsi currently only supports units='mpp'")

    coords: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    batch: list[np.ndarray] = []

    for coord, patch in tile_wsi(
        wsi,
        annotation_store=annotation_store,
        patch_size=patch_size,
        stride=stride,
        min_mask_ratio=min_mask_ratio,
        resolution=resolution,
        units=units,
    ):
        coords.append(coord)
        batch.append(patch)
        if len(batch) == batch_size:
            scores.extend(_infer_batch(model, batch, device))
            batch.clear()

    if batch:
        scores.extend(_infer_batch(model, batch, device))

    thumb = wsi.slide_thumbnail(heatmap_mpp, "mpp")
    shape = thumb.shape[:2]
    scale = resolution / heatmap_mpp
    heatmap = stitch_scores(coords, scores, shape, scale)
    return heatmap
