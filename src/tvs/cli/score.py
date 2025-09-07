"""Command-line interface for whole-slide scoring."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from ..inference import score_wsi
from ..io import find_slide_path, parse_data_paths, resolve_annotation_path

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the score CLI."""

    parser = argparse.ArgumentParser(
        description="Score whole-slide images with a patch-level model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a whole-slide image or a manifest CSV produced by ``tvs-split``",
    )
    parser.add_argument("model", type=Path, help="TorchScript model to use for scoring")
    parser.add_argument(
        "--annotation-db",
        type=Path,
        help="Annotation store restricting inference",
    )
    parser.add_argument(
        "--data-paths",
        type=Path,
        default=Path("configs/data_paths.yaml"),
        help="Data paths configuration YAML",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--heatmap-mpp",
        type=float,
        default=6.0,
        help="Resolution of output heatmap in microns-per-pixel",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("score.png"),
        help="Path to save heatmap PNG for a single slide",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scores"),
        help="Directory to save heatmap PNGs when input is a manifest CSV",
    )
    return parser


def _score_and_save(
    model: torch.nn.Module,
    wsi: Path,
    output_path: Path,
    *,
    annotation_db: Path | None,
    batch_size: int,
    heatmap_mpp: float,
    dataset_cfg: dict,
) -> None:
    """Run scoring for a single slide and write the PNG output."""

    heatmap = score_wsi(
        model,
        wsi,
        annotation_db and str(annotation_db),
        patch_size=dataset_cfg["patch_size"],
        stride=dataset_cfg.get("stride"),
        batch_size=batch_size,
        resolution=dataset_cfg.get("resolution", 0),
        units=dataset_cfg.get("units", "mpp"),
        heatmap_mpp=heatmap_mpp,
    )

    img = ((heatmap + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = torch.jit.load(str(ns.model))
    cfg_path = ns.model.parent / "dataset.yaml"
    dataset_cfg = yaml.safe_load(cfg_path.read_text())

    if ns.input.suffix.lower() == ".csv":
        # Batch scoring from manifest CSV
        output_dir = ns.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        data_paths = yaml.safe_load(ns.data_paths.read_text())
        slide_dirs, ann_dirs = parse_data_paths(data_paths)

        with ns.input.open() as f:
            reader = csv.DictReader(f)
            if not {"slide_id", "path"}.issubset(reader.fieldnames or {}):
                parser.error("Manifest must contain 'slide_id' and 'path' columns")
            for row in reader:
                slide_id = row["slide_id"]
                ann_path = resolve_annotation_path(Path(row["path"]), ann_dirs)
                ext = dataset_cfg.get("slide_extension", "svs").lstrip(".")
                wsi_path = find_slide_path(slide_id, slide_dirs, ext)
                log.info("Scoring %s", slide_id)
                _score_and_save(
                    model,
                    wsi_path,
                    output_dir / f"{slide_id}.png",
                    annotation_db=ann_path,
                    batch_size=ns.batch_size,
                    heatmap_mpp=ns.heatmap_mpp,
                    dataset_cfg=dataset_cfg,
                )
    else:
        # Single-slide scoring
        _score_and_save(
            model,
            ns.input,
            ns.output,
            annotation_db=ns.annotation_db,
            batch_size=ns.batch_size,
            heatmap_mpp=ns.heatmap_mpp,
            dataset_cfg=dataset_cfg,
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
