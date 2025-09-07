"""Helpers for resolving slide and annotation paths.

This module provides utilities to work with multiple directories for
slides and annotations as specified in ``data_paths.yaml``.  It includes
functions to normalise the configuration, search for slides across
directories and resolve annotation paths that may be relative to one of
several root folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

__all__ = ["parse_data_paths", "find_slide_path", "resolve_annotation_path"]


def _ensure_list(value: Union[str, Path, Sequence[Union[str, Path]]]) -> list[Path]:
    """Return ``value`` as a list of :class:`Path` objects.

    Parameters
    ----------
    value:
        A single path-like object or a sequence of path-like objects.

    Returns
    -------
    list of :class:`Path`
        Normalised list of paths.
    """

    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(v) for v in value]


def parse_data_paths(cfg: dict) -> tuple[list[Path], list[Path]]:
    """Normalise data path configuration and validate consistency.

    Parameters
    ----------
    cfg:
        Mapping loaded from ``data_paths.yaml`` containing ``slides`` and
        optionally ``annotations`` entries.

    Returns
    -------
    tuple[list[Path], list[Path]]
        Two lists containing slide directories and annotation directories
        respectively.  If annotation directories are provided their count
        must match the number of slide directories.

    Raises
    ------
    ValueError
        If the number of slide and annotation directories differ.
    """

    slides = _ensure_list(cfg.get("slides", []))
    annotations = _ensure_list(cfg.get("annotations", []))
    if annotations and len(slides) != len(annotations):
        raise ValueError(
            "'slides' and 'annotations' must contain the same number of directories"
        )
    return slides, annotations


def find_slide_path(slide_id: str, slide_dirs: Sequence[Path], extension: str) -> Path:
    """Locate a slide by searching across multiple directories.

    Parameters
    ----------
    slide_id:
        Identifier of the slide without file extension.
    slide_dirs:
        Sequence of directories to search within.
    extension:
        File extension of the slide, without a leading dot.

    Returns
    -------
    :class:`Path`
        Path to the matching slide.

    Raises
    ------
    FileNotFoundError
        If the slide cannot be located in any of the provided directories.
    """

    for directory in slide_dirs:
        candidate = Path(directory) / f"{slide_id}.{extension}"
        if candidate.exists():
            return candidate
    search_paths = ", ".join(str(p) for p in slide_dirs)
    raise FileNotFoundError(
        f"Slide '{slide_id}' with extension '.{extension}' not found in: {search_paths}"
    )


def resolve_annotation_path(path: Path, annotation_dirs: Sequence[Path]) -> Path:
    """Resolve an annotation path against potential root directories.

    Parameters
    ----------
    path:
        Path to an annotation file.  May be absolute or relative.
    annotation_dirs:
        Sequence of root directories that may contain the annotation.

    Returns
    -------
    :class:`Path`
        Absolute path to the annotation file if found, otherwise the
        original ``path``.
    """

    if path.is_absolute() or not annotation_dirs:
        return path
    for directory in annotation_dirs:
        candidate = Path(directory) / path
        if candidate.exists():
            return candidate
    return path
