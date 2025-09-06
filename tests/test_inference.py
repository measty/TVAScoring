import numpy as np
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
            annotation_store=str(db_path),
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
        annotation_store=str(db_path),
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
        str(db_path),
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
