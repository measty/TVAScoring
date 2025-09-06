"""Command-line interface for model evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..training.metrics import mae, pearsonr, rmse, spearmanr

log = logging.getLogger(__name__)

__all__ = [
    "build_parser",
    "compute_metrics",
    "reliability_curve",
    "main",
]


# ---------------------------------------------------------------------------


def compute_metrics(
    preds: Sequence[float], targets: Sequence[float]
) -> dict[str, float]:
    """Compute standard regression metrics from predictions and targets."""

    p = torch.tensor(preds, dtype=torch.float32)
    t = torch.tensor(targets, dtype=torch.float32)
    return {
        "mae": mae(p, t),
        "rmse": rmse(p, t),
        "pearson": pearsonr(p, t),
        "spearman": spearmanr(p, t),
    }


# ---------------------------------------------------------------------------


def reliability_curve(
    preds: Sequence[float],
    targets: Sequence[float],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a reliability curve for predictions in ``[-1, 1]``."""

    preds_arr = np.asarray(list(preds), dtype=float)
    targets_arr = np.asarray(list(targets), dtype=float)
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    indices = np.digitize(preds_arr, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_pred = np.full(n_bins, np.nan)
    mean_target = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = indices == b
        if np.any(mask):
            mean_pred[b] = preds_arr[mask].mean()
            mean_target[b] = targets_arr[mask].mean()
            counts[b] = mask.sum()
    return bin_centers, mean_pred, mean_target, counts


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against targets and generate metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "predictions", type=Path, help="CSV with prediction and target columns"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics.json"),
        help="Where to save metrics JSON",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("reliability.png"),
        help="Reliability plot output path",
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Number of bins for reliability plot"
    )
    return parser


# ---------------------------------------------------------------------------


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    preds: list[float] = []
    targets: list[float] = []
    with ns.predictions.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append(float(row["prediction"]))
            targets.append(float(row["target"]))

    if not preds:
        raise ValueError("No predictions found in CSV")

    metrics = compute_metrics(preds, targets)
    log.info("Metrics: %s", metrics)
    ns.output.write_text(json.dumps(metrics, indent=2))

    # Reliability plot
    centers, mean_pred, mean_target, counts = reliability_curve(preds, targets, ns.bins)
    mask = counts > 0
    plt.figure()
    plt.plot(mean_pred[mask], mean_target[mask], marker="o")
    plt.plot([-1, 1], [-1, 1], "--", color="gray")
    plt.xlabel("Mean prediction")
    plt.ylabel("Mean target")
    plt.title("Reliability curve")
    plt.tight_layout()
    plt.savefig(ns.plot)
    plt.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
