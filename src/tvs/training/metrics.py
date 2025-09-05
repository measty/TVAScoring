"""Common regression metrics for model evaluation."""

from __future__ import annotations

import torch

__all__ = ["mae", "rmse", "pearsonr", "spearmanr"]


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error."""

    diff = (preds - targets).double().abs().mean()
    return diff.item()


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Root mean squared error."""

    diff = (preds - targets).float()
    return torch.sqrt((diff**2).mean()).item()


def pearsonr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Pearson correlation coefficient."""

    preds = preds - preds.mean()
    targets = targets - targets.mean()
    num = torch.sum(preds * targets)
    denom = torch.sqrt(torch.sum(preds**2) * torch.sum(targets**2))
    return (num / denom).item() if denom > 0 else 0.0


def spearmanr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Spearman rank correlation coefficient."""

    pred_ranks = torch.argsort(torch.argsort(preds))
    target_ranks = torch.argsort(torch.argsort(targets))
    return pearsonr(pred_ranks.float(), target_ranks.float())
