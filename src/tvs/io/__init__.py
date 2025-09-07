"""Input/output utilities for TVAScoring.

This subpackage exposes helpers for working with whole-slide images and
annotation stores. Only annotation utilities are implemented at this
stage of the project.
"""

from .annotations import geojson_to_sqlite, prepare_annotations
from .paths import find_slide_path, parse_data_paths, resolve_annotation_path

__all__ = [
    "geojson_to_sqlite",
    "prepare_annotations",
    "find_slide_path",
    "parse_data_paths",
    "resolve_annotation_path",
]
