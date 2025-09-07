"""CLI for preparing annotation SQLite stores."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from ..io.annotations import prepare_annotations

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the annotation preparation CLI."""

    parser = argparse.ArgumentParser(
        description="Convert GeoJSON annotations to TIAToolbox SQLite stores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        help="One or more directories containing QuPath-exported .geojson files.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Destination directory for the generated .db files.",
    )
    parser.add_argument(
        "--data-paths",
        type=Path,
        default=Path("configs/data_paths.yaml"),
        help="Configuration file specifying data roots.",
    )
    parser.add_argument(
        "--class-key",
        default="name",
        help="Annotation property containing the class label.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate databases even if cached versions exist.",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``tvs-prepare-annotations`` command."""

    parser = build_parser()
    ns = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg: dict | None = None
    if ns.source is None or ns.dest is None:
        cfg = yaml.safe_load(ns.data_paths.read_text())

    if ns.source is None:
        ann_dirs = cfg.get("annotations") if cfg else None
        if ann_dirs is None:
            parser.error("--source or 'annotations' in data paths must be provided")
        if isinstance(ann_dirs, (str, Path)):
            ns.source = [Path(ann_dirs)]
        else:
            ns.source = [Path(p) for p in ann_dirs]

    if ns.dest is None:
        cache_dir = cfg.get("cache") if cfg else None
        if cache_dir is None:
            parser.error("--dest or 'cache' in data paths must be provided")
        ns.dest = Path(cache_dir)

    stats = prepare_annotations(
        ns.source,
        ns.dest,
        class_key=ns.class_key,
        cache=not ns.overwrite,
    )
    log.info("Processed %d slides", len(stats))
    log.info(
        "Per-class statistics:\n%s",
        json.dumps(stats, indent=2),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
