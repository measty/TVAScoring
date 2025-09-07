"""Command-line interface for model training."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
import yaml

from ..data.dataset import PatchDataset
from ..data.extractors import build_filters, make_extractors
from ..data.samplers import BalancedSampler
from ..data.transforms import build_transforms
from ..io import find_slide_path, parse_data_paths, resolve_annotation_path
from ..models import make_model
from ..training import Trainer, get_loss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append((row["slide_id"], Path(row["path"])))
    return entries


def _make_loader(
    manifest: Path,
    dataset_cfg: dict,
    data_paths: dict,
    batch_size: int,
    num_workers: int,
    *,
    train: bool,
) -> torch.utils.data.DataLoader:
    entries = _load_manifest(manifest)
    if not entries:
        raise ValueError("Manifest contains no entries")

    slide_dirs, ann_dirs = parse_data_paths(data_paths)

    slides: list[tuple[Path, str]] = []
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
        train=train, augmentation=dataset_cfg.get("augmentation", {}).get("name")
    )
    dataset = PatchDataset(extractors, dataset_cfg["class_map"], transform=transform)
    sampler = BalancedSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tubulovillous scoring model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="Training configuration YAML")
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
    parser.add_argument("--train-manifest", type=Path, help="CSV of training slides")
    parser.add_argument("--val-manifest", type=Path, help="CSV of validation slides")
    return parser


# ---------------------------------------------------------------------------


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train_cfg = yaml.safe_load(ns.config.read_text())
    dataset_cfg = yaml.safe_load(ns.dataset_config.read_text())
    data_paths = yaml.safe_load(ns.data_paths.read_text())

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.yaml").write_text(yaml.safe_dump(train_cfg))
    (output_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_cfg))

    train_loader = _make_loader(
        ns.train_manifest,
        dataset_cfg,
        data_paths,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 0),
        train=True,
    )
    val_loader = _make_loader(
        ns.val_manifest,
        dataset_cfg,
        data_paths,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 0),
        train=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(train_cfg["model"]).to(device)
    optimizer_cls = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }[train_cfg.get("optimizer", "adamw").lower()]
    optimizer = optimizer_cls(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scheduler = None
    if train_cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.get("epochs", 1)
        )
    loss_fn = get_loss(train_cfg.get("loss", "mse"))
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device,
        scheduler=scheduler,
        max_grad_norm=train_cfg.get("grad_clip"),
        precision=train_cfg.get("precision", "fp32"),
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        output_dir=output_dir,
    )
    trainer.fit(
        train_loader,
        val_loader,
        epochs=train_cfg.get("epochs", 1),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
