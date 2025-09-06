"""Command-line interface for whole-slide scoring."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from ..inference import score_wsi

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a whole-slide image with a patch-level model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wsi", type=Path, help="Path to whole-slide image")
    parser.add_argument("model", type=Path, help="TorchScript model to use for scoring")
    parser.add_argument(
        "--annotation-db", type=Path, help="Annotation store restricting inference"
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
        help="Path to save heatmap PNG",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = torch.jit.load(str(ns.model))
    cfg_path = ns.model.parent / "dataset.yaml"
    dataset_cfg = yaml.safe_load(cfg_path.read_text())
    heatmap = score_wsi(
        model,
        ns.wsi,
        ns.annotation_db and str(ns.annotation_db),
        patch_size=dataset_cfg["patch_size"],
        stride=dataset_cfg.get("stride"),
        batch_size=ns.batch_size,
        resolution=dataset_cfg.get("resolution", 0),
        units=dataset_cfg.get("units", "mpp"),
        heatmap_mpp=ns.heatmap_mpp,
    )

    img = ((heatmap + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(ns.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
