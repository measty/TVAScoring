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
from sklearn.metrics import auc, roc_curve

from ..training.metrics import mae, pearsonr, rmse, spearmanr

log = logging.getLogger(__name__)

__all__ = [
    "build_parser",
    "compute_metrics",
    "reliability_curve",
    "roc_curve_tv",
    "main",
]


# ---------------------------------------------------------------------------


def compute_metrics(
    preds: Sequence[float], targets: Sequence[float]
) -> dict[str, float]:
    """Compute standard regression metrics from predictions and targets."""

    p = torch.tensor(preds, dtype=torch.float32)
    t = torch.tensor(targets, dtype=torch.float32)
    metrics = {
        "mae": mae(p, t),
        "rmse": rmse(p, t),
        "pearson": pearsonr(p, t),
        "spearman": spearmanr(p, t),
    }
    try:
        _, _, _, auroc = roc_curve_tv(preds, targets)
    except ValueError:
        auroc = float("nan")
    metrics["auroc"] = auroc
    return metrics


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


def roc_curve_tv(
    preds: Sequence[float], targets: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute ROC curve for tubular (T) vs villous (V) classes.

    Predictions and targets are assumed to lie in ``[-1, 1]`` with targets
    ``<0`` representing class T and ``>0`` representing class V. Neutral
    targets (``0``) are ignored.
    """

    preds_arr = np.asarray(list(preds), dtype=float)
    targets_arr = np.asarray(list(targets), dtype=float)
    mask = targets_arr != 0
    preds_arr = preds_arr[mask]
    targets_arr = targets_arr[mask]
    if preds_arr.size == 0:
        raise ValueError("No T/V samples available for ROC computation")
    labels = (targets_arr > 0).astype(int)
    if np.unique(labels).size < 2:
        raise ValueError("ROC curve requires both T and V samples")
    fpr, tpr, thresholds = roc_curve(labels, preds_arr)
    auroc = auc(fpr, tpr)
    return fpr, tpr, thresholds, auroc


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
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save metrics and plots. Defaults to the predictions "
            "CSV directory if not provided."
        ),
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

    output_dir = ns.output_dir or ns.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    roc_path = output_dir / "roc.png"
    rel_path = output_dir / "reliability.png"

    metrics = compute_metrics(preds, targets)
    log.info("Metrics: %s", metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("Saved metrics to %s", metrics_path)

    # ROC curve
    try:
        fpr, tpr, _, auroc = roc_curve_tv(preds, targets)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve (T vs V)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        log.info("AUROC (T vs V): %.4f", auroc)
        log.info("Saved ROC curve to %s", roc_path)
    except ValueError as e:  # pragma: no cover - edge case
        log.warning("ROC curve not plotted: %s", e)

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
    plt.savefig(rel_path)
    plt.close()
    log.info("Saved reliability plot to %s", rel_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
