"""Command-line interface for previewing sampled patches."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

from ..data.dataset import PatchDataset
from ..data.extractors import build_filters, make_extractors

log = logging.getLogger(__name__)

__all__ = ["build_parser", "make_montage", "main"]

# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append((row["slide_id"], Path(row["path"])))
    return entries


def _load_dataset(manifest: Path, dataset_cfg: dict, data_paths: dict) -> PatchDataset:
    entries = _load_manifest(manifest)
    if not entries:
        raise ValueError("Manifest contains no entries")
    slide_id, ann_path = entries[0]
    wsi_path = Path(data_paths["slides"]) / f"{slide_id}.svs"

    class_labels = list(dataset_cfg["class_map"].keys())
    filters = build_filters([lbl[0].upper() for lbl in class_labels])
    extractors = make_extractors(
        wsi_path,
        str(ann_path),
        filters,
        patch_size=dataset_cfg["patch_size"],
        stride=dataset_cfg.get("stride"),
        min_mask_ratio=dataset_cfg.get("min_mask_ratio", 0.5),
    )
    return PatchDataset(extractors, dataset_cfg["class_map"])


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
