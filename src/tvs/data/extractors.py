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
from typing import Callable, Iterable, Mapping

import tiatoolbox.tools.patchextraction as patchextraction
from tiatoolbox.annotation import AnnotationStore
from tiatoolbox.wsicore.wsireader import WSIReader

__all__ = ["build_filters", "make_extractors"]


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

    return {label: lambda props: props[class_key][0] == label for label in labels}


def make_extractors(
    wsi: WSIReader | str | Path,
    store: AnnotationStore,
    class_filters: Mapping[str, Callable[[dict[str, str]], bool]],
    *,
    patch_size: int,
    stride: int | tuple[int, int] | None = None,
    min_mask_ratio: float = 0.5,
    pad_mode: str = "constant",
    pad_constant_values: int = 0,
) -> dict[str, patchextraction.SlidingWindowPatchExtractor]:
    """Create a ``SlidingWindowPatchExtractor`` for each class filter.

    Parameters
    ----------
    wsi:
        Whole-slide image as a :class:`WSIReader` or a path to one.  If a path
        is supplied it is opened via :func:`WSIReader.open`.
    store:
        Annotation store restricting sampling locations.
    class_filters:
        Mapping from class label to a ``store_filter`` callable selecting
        annotations of that class.
    patch_size:
        Size of the extracted patches in pixels.
    stride:
        Step size between successive patches.  Defaults to ``patch_size`` when
        ``None``.
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
        Mapping of class label to configured
        :class:`SlidingWindowPatchExtractor` instances.
    """

    if not isinstance(wsi, WSIReader):
        wsi = WSIReader.open(wsi)
    if stride is None:
        stride = patch_size

    extractors: dict[str, patchextraction.SlidingWindowPatchExtractor] = {}
    for label, store_filter in class_filters.items():
        extractors[label] = patchextraction.SlidingWindowPatchExtractor(
            input_img=wsi,
            input_mask=store,
            patch_size=patch_size,
            stride=stride,
            within_bound=False,
            pad_mode=pad_mode,
            pad_constant_values=pad_constant_values,
            min_mask_ratio=min_mask_ratio,
            store_filter=store_filter,
        )
    return extractors
