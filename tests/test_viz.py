import numpy as np

from tvs.cli.preview import make_montage
from tvs.viz.overlays import overlay_scores, score_to_color


def test_score_to_color():
    scores = np.array([[-1.0, 0.0, 1.0]])
    colors = score_to_color(scores)
    assert colors.shape == (1, 3, 3)
    # ensure colors for different scores are different
    assert not np.allclose(colors[0, 0], colors[0, 1])
    assert not np.allclose(colors[0, 1], colors[0, 2])


def test_overlay_scores():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    scores = np.ones((2, 2))
    overlay = overlay_scores(image, scores, alpha=0.5)
    assert overlay.shape == image.shape
    assert overlay.mean() > 0  # overlay introduces color


def test_make_montage():
    patches = [np.full((4, 4, 3), i, dtype=np.uint8) for i in [50, 100, 150, 200]]
    montage = make_montage(patches, n_cols=2)
    assert montage.shape == (8, 8, 3)
    assert (montage[:4, :4] == 50).all()
    assert (montage[:4, 4:] == 100).all()
    assert (montage[4:, :4] == 150).all()
    assert (montage[4:, 4:] == 200).all()
