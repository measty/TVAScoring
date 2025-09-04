"""Data utilities for patch extraction and datasets.

This subpackage provides convenience wrappers around TIAToolbox's
``SlidingWindowPatchExtractor`` for class-conditioned sampling.  Additional
helpers and dataset utilities will be added in subsequent tasks.
"""

from .extractors import build_filters, make_extractors

__all__ = ["build_filters", "make_extractors"]
