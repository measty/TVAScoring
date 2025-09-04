"""Data utilities for patch extraction and datasets.

This subpackage provides convenience wrappers around TIAToolbox's
``SlidingWindowPatchExtractor`` for class-conditioned sampling.  Additional
helpers and dataset utilities will be added in subsequent tasks.
"""

from .dataset import PatchDataset
from .extractors import build_filters, make_extractors
from .samplers import BalancedSampler
from .transforms import build_transforms

__all__ = [
    "build_filters",
    "make_extractors",
    "PatchDataset",
    "BalancedSampler",
    "build_transforms",
]
