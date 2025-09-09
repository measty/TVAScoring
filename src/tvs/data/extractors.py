"""Wrappers around ``SlidingWindowPatchExtractor`` for class-specific sampling.

The helpers in this module simplify creating a separate
:class:`~tiatoolbox.tools.patchextraction.SlidingWindowPatchExtractor` for
each annotation class.  Each extractor is configured with an appropriate
``store_filter`` so that it only yields patches intersecting annotations of
the target class.  Common parameters such as ``stride`` and
``min_mask_ratio`` are exposed for convenience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import tiatoolbox.tools.patchextraction as patchextraction
from tiatoolbox.wsicore.wsireader import WSIReader

__all__ = [
    "build_filters",
    "make_extractors",
    "make_background_extractors",
]


def build_filters(
    labels: Iterable[str],
    *,
    class_key: str = "name",
) -> dict[str, Callable[[dict[str, str]], bool]]:
    """Generate ``store_filter`` callable for annotation classes.

    Parameters
    ----------
    labels:
        Iterable of annotation class labels.
    class_key:
        Property name within annotation ``props`` storing the class label.

    Returns
    -------
    dict
        Mapping of label → filter fn suitable for ``store_filter``.
    """

    # ``lambda`` within a comprehension would otherwise capture ``label`` by
    # reference leading all generated functions to reference the *last* label
    # in ``labels``.  Using a default argument preserves the current value.
    return {
        label: (lambda props, label=label: props[class_key][0] == label)
        for label in labels
    }


def make_extractors(
    slides: Sequence[tuple[WSIReader | str | Path, str]],
    class_filters: Mapping[str, Callable[[dict[str, str]], bool]],
    *,
    patch_size: int,
    stride: int | tuple[int, int] | None = None,
    resolution: float = 0.5,
    min_mask_ratio: float = 0.5,
    pad_mode: str = "constant",
    pad_constant_values: int = 0,
) -> dict[str, list[patchextraction.SlidingWindowPatchExtractor]]:
    """Create patch extractors for multiple slides and classes.

    Parameters
    ----------
    slides:
        Sequence of ``(wsi, store_path)`` pairs where ``wsi`` is either a
        :class:`WSIReader` or a path to one and ``store_path`` is the associated
        annotation store database.
    class_filters:
        Mapping from class label to a ``store_filter`` callable selecting
        annotations of that class.
    patch_size:
        Size of the extracted patches in pixels.
    stride:
        Step size between successive patches.  Defaults to ``patch_size`` when
        ``None``.
    resolution:
        Physical resolution in microns-per-pixel to extract patches at.
    min_mask_ratio:
        Minimum fraction of a patch that must lie within the annotation mask.
    pad_mode:
        Padding mode passed to ``SlidingWindowPatchExtractor`` when patches
        extend beyond slide boundaries.
    pad_constant_values:
        Constant value used when ``pad_mode`` is ``"constant"``.

    Returns
    -------
    dict
        Mapping of class label to a list of configured
        :class:`SlidingWindowPatchExtractor` instances—one per slide.
    """

    if stride is None:
        stride = patch_size

    extractors: dict[str, list[patchextraction.SlidingWindowPatchExtractor]] = {
        label: [] for label in class_filters
    }

    for wsi, store_path in slides:
        if not isinstance(wsi, WSIReader):
            wsi_reader = WSIReader.open(wsi)
        else:
            wsi_reader = wsi

        for label, store_filter in class_filters.items():
            extractors[label].append(
                patchextraction.SlidingWindowPatchExtractor(
                    input_img=wsi_reader,
                    input_mask=store_path,
                    patch_size=patch_size,
                    stride=stride,
                    within_bound=False,
                    pad_mode=pad_mode,
                    pad_constant_values=pad_constant_values,
                    min_mask_ratio=min_mask_ratio,
                    store_filter=store_filter,
                    resolution=resolution,
                    units="mpp",
                )
            )

    return extractors


def make_background_extractors(
    slides: Sequence[tuple[WSIReader | str | Path, str]],
    *,
    patch_size: int,
    stride: int | tuple[int, int] | None = None,
    resolution: float = 0.5,
    min_mask_ratio: float = 0.5,
    pad_mode: str = "constant",
    pad_constant_values: int = 0,
) -> list[patchextraction.SlidingWindowPatchExtractor]:
    """Create patch extractors sampling generic tissue regions.

    These extractors sample from a slide's tissue mask using ``otsu`` thresholding
    and are intended for gathering "background" patches which are not associated
    with any particular annotation class.  The resulting patches can be given a
    neutral target (e.g. ``0``) to gently nudge the model towards zero on
    uncertain regions.

    Parameters
    ----------
    slides:
        Sequence of ``(wsi, store_path)`` pairs, mirroring
        :func:`make_extractors`.  The annotation store paths are ignored but the
        signature is kept for API compatibility.
    patch_size:
        Size of the extracted patches in pixels.
    stride:
        Step size between successive patches.  Defaults to ``patch_size`` when
        ``None``.
    resolution:
        Physical resolution in microns-per-pixel to extract patches at.
    min_mask_ratio:
        Minimum fraction of a patch that must lie within the tissue mask.
    pad_mode:
        Padding mode passed to ``SlidingWindowPatchExtractor`` when patches
        extend beyond slide boundaries.
    pad_constant_values:
        Constant value used when ``pad_mode`` is ``"constant"``.

    Returns
    -------
    list of :class:`SlidingWindowPatchExtractor`
        One extractor per slide sampling tissue regions.
    """

    if stride is None:
        stride = patch_size

    extractors: list[patchextraction.SlidingWindowPatchExtractor] = []
    for wsi, _ in slides:
        if not isinstance(wsi, WSIReader):
            wsi_reader = WSIReader.open(wsi)
        else:
            wsi_reader = wsi

        extractors.append(
            patchextraction.SlidingWindowPatchExtractor(
                input_img=wsi_reader,
                input_mask="otsu",
                patch_size=patch_size,
                stride=stride,
                within_bound=False,
                pad_mode=pad_mode,
                pad_constant_values=pad_constant_values,
                min_mask_ratio=min_mask_ratio,
                resolution=resolution,
                units="mpp",
            )
        )

    return extractors
