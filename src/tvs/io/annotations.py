"""Utilities for working with annotation stores.

This module provides helpers to convert QuPath `.geojson` annotation
files to the `SQLiteStore` format used by TIAToolbox.  The resulting
SQLite database can be cached for faster subsequent loads.  Basic
validation of the annotation properties is performed, ensuring that each
geometry contains a class label.  Simple statistics (count and total
area per class) are returned for downstream use.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

from tiatoolbox.annotation import AnnotationStore, SQLiteStore

__all__ = ["geojson_to_sqlite", "prepare_annotations"]

def get_class_from_props(props: dict[str, Any], class_key: str = "name") -> str:
    """Get the class from the properties. In our data, the class key is 'name'.
    Annotations are named T1, T2, etc or V1, V2 etc with first letter being the class.
    
    """
    if class_key not in props:
        raise ValueError(
            f"Annotation is missing the '{class_key}' property.",
        )
    return props[class_key][0]

def _validate_and_stats(
    store: AnnotationStore, class_key: str
) -> Dict[str, Dict[str, float]]:
    """Validate annotations and compute per-class statistics.

    Parameters
    ----------
    store:
        The annotation store containing geometries.
    class_key:
        Key within the annotation ``properties`` mapping that stores the
        class label.  Typically ``"name"`` in this project.

    Returns
    -------
    dict
        Mapping of class label to a dictionary with ``count`` and
        ``area`` fields.

    Raises
    ------
    ValueError
        If any annotation is missing the required ``class_key`` property.
    """

    stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "area": 0.0},
    )
    for ann in store.values():
        props = getattr(ann, "properties", {})

        label = get_class_from_props(props, class_key)
        stats[label]["count"] += 1
        # ``geometry`` is a shapely object; ``area`` is in pixel units.
        stats[label]["area"] += ann.geometry.area
    return stats


def geojson_to_sqlite(
    geojson_path: Path,
    db_path: Path,
    *,
    class_key: str = "name",
    cache: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Convert a GeoJSON file to ``SQLiteStore`` and return statistics.

    Parameters
    ----------
    geojson_path:
        Path to the source ``.geojson`` file exported from QuPath.
    db_path:
        Destination path for the resulting SQLite database.  The parent
        directory will be created if it does not already exist.
    class_key:
        Name of the property within each annotation that contains the
        class label.
    cache:
        If ``True`` and ``db_path`` already exists, the existing database
        is loaded instead of re-parsing the GeoJSON.

    Returns
    -------
    dict
        Per-class statistics for the processed slide.  See
        :func:`_validate_and_stats`.
    """

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if cache and db_path.exists():
        store = SQLiteStore.open(db_path)
    else:
        if not cache and db_path.exists():
            db_path.unlink()
        store = SQLiteStore.from_geojson(geojson_path)
        store.dump(db_path)

    try:
        stats = _validate_and_stats(store, class_key)
    finally:
        # Explicitly close the store to release any file handles.
        store.close()
    return stats


def prepare_annotations(
    source_dir: Path,
    dest_dir: Path,
    *,
    class_key: str = "name",
    cache: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Batch convert GeoJSON annotations to ``SQLiteStore`` databases.

    Parameters
    ----------
    source_dir:
        Directory containing one or more ``.geojson`` files.
    dest_dir:
        Directory to write the ``.db`` files into.  Will be created if it
        does not already exist.
    class_key:
        Annotation property containing the class label.
    cache:
        Skip conversion if the corresponding ``.db`` already exists.

    Returns
    -------
    dict
        Mapping from slide identifier (stem of the GeoJSON filename) to
        per-class statistics dictionaries.
    """

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stats_per_slide: Dict[str, Dict[str, Dict[str, float]]] = {}
    for geojson_path in sorted(source_dir.glob("*.geojson")):
        slide_id = geojson_path.stem
        db_path = dest_dir / f"{slide_id}.db"
        stats_per_slide[slide_id] = geojson_to_sqlite(
            geojson_path,
            db_path,
            class_key=class_key,
            cache=cache,
        )
    return stats_per_slide
