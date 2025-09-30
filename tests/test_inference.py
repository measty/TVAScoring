import numpy as np
import pandas as pd
import torch
from tiatoolbox.wsicore.wsireader import WSIReader

from tvs.inference import score_wsi, stitch_scores, tile_wsi


class MeanModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Return mean of input in range [0,1]
        return x.mean(dim=(1, 2, 3))


def test_tiler_outputs(annotation_db, wsi_path):
    db_path, _ = annotation_db
    patches = list(
        tile_wsi(
            wsi_path,
            masker=str(db_path),
            patch_size=512,
            stride=512,
            min_mask_ratio=0.0,
            resolution=1.0,
            units="mpp",
        )
    )
    assert patches, "No patches returned"
    for (x, y, w, h), patch in patches:
        assert patch.shape[:2] == (h, w)
        assert x >= 0 and y >= 0


def test_score_wsi_matches_manual(annotation_db, wsi_path):
    db_path, _ = annotation_db
    model = MeanModel()
    wsi = WSIReader.open(wsi_path)

    # Manual pipeline
    coords = []
    patches = []
    for coord, patch in tile_wsi(
        wsi,
        masker=str(db_path),
        patch_size=512,
        stride=512,
        min_mask_ratio=0.0,
        resolution=1.0,
        units="mpp",
    ):
        coords.append(coord)
        patches.append(patch)
    scores = []
    for p in patches:
        tensor = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        scores.append(model(tensor).item())
    thumb = wsi.slide_thumbnail(6.0, "mpp")
    expected = stitch_scores(coords, scores, thumb.shape[:2], scale=1.0 / 6.0)

    heatmap = score_wsi(
        model,
        wsi,
        masker=str(db_path),
        patch_size=512,
        stride=512,
        batch_size=4,
        min_mask_ratio=0.0,
        resolution=1.0,
        units="mpp",
        heatmap_mpp=6.0,
    )
    assert heatmap.shape == expected.shape
    np.testing.assert_allclose(heatmap, expected, rtol=1e-6, atol=1e-6)


def test_score_wsi_dataframe_scaling(annotation_db, wsi_path):
    db_path, _ = annotation_db
    model = MeanModel()
    wsi = WSIReader.open(wsi_path)

    coords = [
        coord
        for coord, _ in tile_wsi(
            wsi,
            masker=str(db_path),
            patch_size=512,
            stride=512,
            min_mask_ratio=0.0,
            resolution=1.0,
            units="mpp",
        )
    ]

    heatmap, dataframe = score_wsi(
        model,
        wsi,
        masker=str(db_path),
        patch_size=512,
        stride=512,
        batch_size=4,
        min_mask_ratio=0.0,
        resolution=1.0,
        units="mpp",
        heatmap_mpp=6.0,
        return_dataframe=True,
    )

    assert isinstance(heatmap, np.ndarray)
    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == ["x", "y", "width", "height", "score"]
    assert len(dataframe) == len(coords)

    native_mpp = getattr(wsi.info, "mpp", None)
    baseline_x = 1.0
    baseline_y = 1.0
    if native_mpp is not None:
        try:
            baseline_x = float(native_mpp[0])
        except (TypeError, IndexError, ValueError):
            baseline_x = 1.0
        try:
            baseline_y = float(native_mpp[1])
        except (TypeError, IndexError, ValueError):
            baseline_y = baseline_x
    else:
        baseline_y = baseline_x
    scale_x = 1.0 / baseline_x
    scale_y = 1.0 / baseline_y
    expected_x = np.array([c[0] for c in coords], dtype=float) * scale_x
    expected_y = np.array([c[1] for c in coords], dtype=float) * scale_y
    expected_w = np.array([c[2] for c in coords], dtype=float) * scale_x
    expected_h = np.array([c[3] for c in coords], dtype=float) * scale_y

    np.testing.assert_allclose(dataframe["x"].to_numpy(), expected_x)
    np.testing.assert_allclose(dataframe["y"].to_numpy(), expected_y)
    np.testing.assert_allclose(dataframe["width"].to_numpy(), expected_w)
    np.testing.assert_allclose(dataframe["height"].to_numpy(), expected_h)


def test_stitch_scores_with_overlap():
    coords = [(0, 0, 4, 4), (2, 0, 4, 4)]
    scores = [1.0, 3.0]
    heatmap = stitch_scores(coords, scores, (4, 6))
    expected = np.array(
        [
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(heatmap, expected)
