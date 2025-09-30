"""High-level utilities for scoring whole-slide images."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
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
    masker: str | None = None,
    *,
    patch_size: int,
    stride: int | None = None,
    batch_size: int = 16,
    min_mask_ratio: float = 0.5,
    resolution: float = 0,
    units: str = "mpp",
    heatmap_mpp: float = 6.0,
    device: torch.device | None = None,
    return_dataframe: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    """Score a whole-slide image with a patch-level model.

    Parameters
    ----------
    model:
        PyTorch model returning a scalar score in ``[-1, 1]`` for each patch.
    wsi:
        Whole-slide image or path to one.
    masker:
        Optional masking strategy restricting inference to specific regions.
        Can be an annotation store path, ``None`` for no restriction, or
        ``"otsu"`` to generate a tissue mask on-the-fly.
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
    return_dataframe:
        When ``True`` also return a :class:`pandas.DataFrame` containing the
        patch coordinates (in baseline level-0 space) and scores.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, pandas.DataFrame]
        Heatmap of scores with shape matching the slide dimensions. When
        ``return_dataframe`` is ``True`` a tuple of ``(heatmap, dataframe)`` is
        returned.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if not isinstance(wsi, WSIReader):
        wsi = WSIReader.open(wsi)

    native_mpp = getattr(wsi.info, "mpp", None)
    baseline_mpp_x: float | None = None
    baseline_mpp_y: float | None = None
    if native_mpp is not None:
        try:
            baseline_mpp_x = float(native_mpp[0])
        except (TypeError, IndexError, ValueError):
            baseline_mpp_x = None
        try:
            baseline_mpp_y = float(native_mpp[1])
        except (TypeError, IndexError, ValueError):
            baseline_mpp_y = None

    if resolution == 0:
        if baseline_mpp_x is None:
            raise ValueError("Unable to infer resolution because the slide lacks mpp metadata")
        resolution = float(baseline_mpp_x)
    else:
        resolution = float(resolution)

    if units != "mpp":  # pragma: no cover - sanity check
        raise ValueError("score_wsi currently only supports units='mpp'")

    if baseline_mpp_x is None:
        baseline_mpp_x = float(resolution)
    if baseline_mpp_y is None:
        baseline_mpp_y = baseline_mpp_x

    coords: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    batch: list[np.ndarray] = []

    for coord, patch in tile_wsi(
        wsi,
        masker=masker,
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

    if not return_dataframe:
        return heatmap

    scale_x = resolution / baseline_mpp_x if baseline_mpp_x else 1.0
    scale_y = resolution / baseline_mpp_y if baseline_mpp_y else 1.0
    records = [
        {
            "x": x * scale_x,
            "y": y * scale_y,
            "width": w * scale_x,
            "height": h * scale_y,
            "score": score,
        }
        for (x, y, w, h), score in zip(coords, scores)
    ]
    dataframe = pd.DataFrame.from_records(
        records,
        columns=["x", "y", "width", "height", "score"],
    )
    return heatmap, dataframe
