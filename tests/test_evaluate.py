import numpy as np
import pytest

from tvs.cli.evaluate import compute_metrics, reliability_curve, roc_curve_tv


def test_compute_metrics():
    preds = [-1, -1, -1, 1, 1, 1]
    targets = [-1, -0.5, 0, 0.5, 1, 1]
    metrics = compute_metrics(preds, targets)
    assert metrics["mae"] == pytest.approx(1 / 3, rel=1e-4)
    assert metrics["rmse"] == pytest.approx(0.5, rel=1e-4)
    assert metrics["pearson"] == pytest.approx(0.8944, rel=1e-3)
    assert metrics["spearman"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["auroc"] == pytest.approx(1.0, rel=1e-6)


def test_reliability_curve():
    preds = [-1, -1, -1, 1, 1, 1]
    targets = [-1, -0.5, 0, 0.5, 1, 1]
    centers, mean_pred, mean_target, counts = reliability_curve(
        preds, targets, n_bins=2
    )
    assert counts.tolist() == [3, 3]
    assert mean_pred.tolist() == [-1, 1]
    assert mean_target[0] == pytest.approx(-0.5, rel=1e-4)
    assert mean_target[1] == pytest.approx(0.833333, rel=1e-4)


def test_roc_curve_tv():
    preds = [-0.2, 0.1, 0.8, -0.7]
    targets = [-1, 1, 1, -1]
    fpr, tpr, thresholds, auroc = roc_curve_tv(preds, targets)
    assert auroc == pytest.approx(1.0, rel=1e-6)
    assert fpr[0] == pytest.approx(0.0, rel=1e-6)
    assert tpr[-1] == pytest.approx(1.0, rel=1e-6)
