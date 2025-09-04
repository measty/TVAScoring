"""CLI for preparing annotation SQLite stores."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..io.annotations import prepare_annotations

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the annotation preparation CLI."""

    parser = argparse.ArgumentParser(
        description="Convert GeoJSON annotations to TIAToolbox SQLite stores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Directory containing QuPath-exported .geojson files.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Destination directory for the generated .db files.",
    )
    parser.add_argument(
        "--class-key",
        default="class",
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

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
