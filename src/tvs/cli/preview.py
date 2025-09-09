"""Command-line interface for previewing sampled patches."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

from ..data.dataset import PatchDataset
from ..data.extractors import build_filters, make_background_extractors, make_extractors
from ..io import slides_from_manifest

log = logging.getLogger(__name__)

__all__ = ["build_parser", "make_montage", "main"]

# ---------------------------------------------------------------------------


def _load_dataset(manifest: Path, dataset_cfg: dict, data_paths: dict) -> PatchDataset:
    slides = slides_from_manifest(manifest, dataset_cfg, data_paths)

    class_labels = list(dataset_cfg["class_map"].keys())
    filters = build_filters([lbl[0].upper() for lbl in class_labels])
    extractors = make_extractors(
        slides,
        filters,
        patch_size=dataset_cfg["patch_size"],
        stride=dataset_cfg.get("stride"),
        resolution=dataset_cfg.get("resolution", 0.5),
        min_mask_ratio=dataset_cfg.get("min_mask_ratio", 0.5),
    )
    label_scores = dict(dataset_cfg["class_map"])
    bg_weight = dataset_cfg.get("background_weight")
    if bg_weight is not None:
        bg_extractors = make_background_extractors(
            slides,
            patch_size=dataset_cfg["patch_size"],
            stride=dataset_cfg.get("stride"),
            resolution=dataset_cfg.get("resolution", 0.5),
            min_mask_ratio=dataset_cfg.get("min_mask_ratio", 0.5),
        )
        extractors["background"] = bg_extractors
        label_scores["background"] = 0.0
    return PatchDataset(extractors, label_scores)


# ---------------------------------------------------------------------------


def make_montage(
    patches: Sequence[np.ndarray], n_cols: int | None = None
) -> np.ndarray:
    """Arrange a list of patches into a grid montage."""

    if not patches:
        raise ValueError("No patches provided")
    h, w, c = patches[0].shape
    n = len(patches)
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    canvas = np.zeros((n_rows * h, n_cols * w, c), dtype=patches[0].dtype)
    for idx, patch in enumerate(patches):
        r, c_idx = divmod(idx, n_cols)
        canvas[r * h : (r + 1) * h, c_idx * w : (c_idx + 1) * w] = patch
    return canvas


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview sampled patches per class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="CSV manifest of slides")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset configuration YAML",
    )
    parser.add_argument(
        "--data-paths",
        type=Path,
        default=Path("configs/data_paths.yaml"),
        help="Data paths configuration YAML",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("preview"), help="Output directory"
    )
    parser.add_argument(
        "--per-class", type=int, default=16, help="Number of patches to show per class"
    )
    return parser


# ---------------------------------------------------------------------------


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_cfg = yaml.safe_load(ns.dataset_config.read_text())
    data_paths = yaml.safe_load(ns.data_paths.read_text())
    dataset = _load_dataset(ns.manifest, dataset_cfg, data_paths)

    ns.out_dir.mkdir(parents=True, exist_ok=True)
    for label, extractor in zip(dataset.labels, dataset.extractors):
        patches = []
        for i in range(min(ns.per_class, len(extractor))):
            patch = extractor[i]
            if isinstance(patch, tuple):
                patch = patch[0]
            patches.append(patch)
        if not patches:
            continue
        montage = make_montage(patches)
        out_path = ns.out_dir / f"{label}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        log.info("Saved %s montage to %s", label, out_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
