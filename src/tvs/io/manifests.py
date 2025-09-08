"""Helpers for loading slide manifests and resolving paths.

These utilities centralise common CSV parsing logic and path resolution
for manifest files used across CLI tools.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from .paths import find_slide_path, parse_data_paths, resolve_annotation_path

__all__ = ["load_manifest", "slides_from_manifest"]


def load_manifest(path: Path) -> List[Tuple[str, Path]]:
    """Load slide IDs and annotation paths from a CSV manifest.

    Parameters
    ----------
    path:
        Path to the CSV manifest with ``slide_id`` and ``path`` columns.

    Returns
    -------
    list of tuple[str, Path]
        List of ``(slide_id, annotation_path)`` pairs in the order they appear
        in the file. The annotation paths are converted to :class:`Path`
        objects.
    """
    entries: List[Tuple[str, Path]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append((row["slide_id"], Path(row["path"])))
    return entries


def slides_from_manifest(
    manifest: Path, dataset_cfg: dict, data_paths: dict
) -> List[Tuple[Path, str]]:
    """Resolve slide and annotation paths defined in a manifest.

    Parameters
    ----------
    manifest:
        Path to the CSV manifest.
    dataset_cfg:
        Dataset configuration dictionary. The ``slide_extension`` key is used to
        infer slide filenames.
    data_paths:
        Mapping of data types (``slides`` and ``annotations``) to directories to
        search.

    Returns
    -------
    list of tuple[Path, str]
        Resolved pairs of slide image paths and annotation database paths.

    Raises
    ------
    ValueError
        If the manifest is empty.
    """
    entries = load_manifest(manifest)
    if not entries:
        raise ValueError("Manifest contains no entries")

    slide_dirs, ann_dirs = parse_data_paths(data_paths)

    slides: List[Tuple[Path, str]] = []
    for slide_id, ann_path in entries:
        ext = dataset_cfg.get("slide_extension", "svs").lstrip(".")
        wsi_path = find_slide_path(slide_id, slide_dirs, ext)
        ann_path = resolve_annotation_path(ann_path, ann_dirs)
        slides.append((wsi_path, str(ann_path)))
    return slides
