from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import cm
from PIL import Image

from tvs.cli import score
from tvs.inference import score_wsi
from tvs.models import make_model


def test_cli_score_manifest(tmp_path: Path, wsi_path: Path, annotation_db) -> None:
    """Ensure the CLI can score all slides listed in a manifest CSV."""

    db_path, _ = annotation_db

    # Save model checkpoint and its dataset config
    model = make_model({"name": "cnn_baseline"})
    model_path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, model_path)
    (model_path.parent / "train.yaml").write_text(
        yaml.safe_dump({"model": {"name": "cnn_baseline"}})
    )
    dataset_cfg = {
        "patch_size": 512,
        "stride": 512,
        "resolution": 1.0,
        "units": "mpp",
        "slide_extension": "svs",
    }
    (model_path.parent / "dataset.yaml").write_text(yaml.safe_dump(dataset_cfg))

    # Data paths configuration
    data_paths = {
        "slides": [wsi_path.parent.as_posix()],
        "annotations": [db_path.parent.as_posix()],
    }
    data_paths_cfg = tmp_path / "data_paths.yaml"
    data_paths_cfg.write_text(yaml.safe_dump(data_paths))

    # Create manifest CSV referencing annotation path
    manifest = tmp_path / "slides.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "path"])
        writer.writerow(["test_slide", db_path])

    out_dir = tmp_path / "out"
    score.main(
        [
            str(manifest),
            str(model_path),
            "--restrict-inference",
            "annotations",
            "--data-paths",
            str(data_paths_cfg),
            "--output-dir",
            str(out_dir),
        ]
    )

    out_file = out_dir / "test_slide.png"
    assert out_file.exists()

    # Compare against direct score_wsi call
    expected = score_wsi(
        model,
        wsi_path,
        masker=str(db_path),
        patch_size=512,
        stride=512,
        batch_size=16,
        resolution=1.0,
        units="mpp",
        heatmap_mpp=6.0,
    )
    cmap = cm.get_cmap("coolwarm")
    expected_img = (cmap((expected + 1) / 2)[..., :3] * 255).astype(np.uint8)
    img = np.array(Image.open(out_file))
    np.testing.assert_array_equal(img, expected_img)
