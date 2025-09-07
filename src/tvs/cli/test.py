"""CLI for generating patch-level predictions on a slide manifest."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import yaml
from tqdm import tqdm

from ..data.dataset import PatchDataset
from ..data.extractors import build_filters, make_extractors
from ..data.transforms import build_transforms
from ..io import find_slide_path, parse_data_paths, resolve_annotation_path

log = logging.getLogger(__name__)

__all__ = ["build_parser", "main"]


# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append((row["slide_id"], Path(row["path"])))
    return entries


def _build_dataset(manifest: Path, dataset_cfg: dict, data_paths: dict) -> PatchDataset:
    entries = _load_manifest(manifest)
    if not entries:
        raise ValueError("Manifest contains no entries")

    slide_dirs, ann_dirs = parse_data_paths(data_paths)

    slides: List[Tuple[Path, str]] = []
    for slide_id, ann_path in entries:
        ext = dataset_cfg.get("slide_extension", "svs").lstrip(".")
        wsi_path = find_slide_path(slide_id, slide_dirs, ext)
        ann_path = resolve_annotation_path(ann_path, ann_dirs)
        slides.append((wsi_path, str(ann_path)))

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
    transform = build_transforms(
        train=False,
        mean=dataset_cfg.get("mean"),
        std=dataset_cfg.get("std"),
    )
    return PatchDataset(extractors, dataset_cfg["class_map"], transform=transform)


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference on a slide manifest and save patch-level predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="CSV manifest of test slides")
    parser.add_argument("model", type=Path, help="TorchScript model for inference")
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
        "--batch-size", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("predictions.csv"), help="Output CSV path"
    )
    return parser


# ---------------------------------------------------------------------------


def main(args: List[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_cfg = yaml.safe_load(ns.dataset_config.read_text())
    data_paths = yaml.safe_load(ns.data_paths.read_text())
    dataset = _build_dataset(ns.manifest, dataset_cfg, data_paths)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=ns.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(ns.model)).to(device).eval()

    preds: List[float] = []
    targets: List[float] = []
    for imgs, tgt in tqdm(loader, desc="Infer", unit="batch"):
        imgs = imgs.to(device)
        with torch.no_grad():
            out = model(imgs).reshape(len(imgs))
        preds.extend(out.detach().cpu().tolist())
        targets.extend(tgt.detach().cpu().tolist())

    with ns.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prediction", "target"])
        writer.writeheader()
        for p, t in zip(preds, targets):
            writer.writerow({"prediction": p, "target": t})
    log.info("Wrote %d predictions to %s", len(preds), ns.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
