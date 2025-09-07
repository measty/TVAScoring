import csv
import shutil

import torch
import yaml

from tvs.cli.prepare_annotations import main as prepare_main
from tvs.cli.preview import _load_dataset
from tvs.cli.test import main as run_test_cli


class MeanModel(torch.nn.Module):
    def forward(self, x):  # type: ignore[override]
        return x.mean(dim=(1, 2, 3))


def test_load_dataset_respects_slide_extension(tmp_path, annotation_db, test_data_dir):
    db_path, _ = annotation_db
    slides_dir1 = tmp_path / "slides1"
    slides_dir2 = tmp_path / "slides2"
    slides_dir1.mkdir()
    slides_dir2.mkdir()
    shutil.copy(test_data_dir / "test_slide.svs", slides_dir2 / "test_slide.tif")

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "path"])
        writer.writeheader()
        writer.writerow({"slide_id": "test_slide", "path": str(db_path)})

    dataset_cfg = {
        "patch_size": 512,
        "class_map": {"T": -1.0, "V": 1.0},
        "slide_extension": "tif",
    }
    data_paths = {"slides": [str(slides_dir1), str(slides_dir2)]}

    dataset = _load_dataset(manifest, dataset_cfg, data_paths)
    assert len(dataset) > 0


def test_prepare_annotations_uses_data_paths(tmp_path, test_data_dir):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    shutil.copy(test_data_dir / "test_slide.geojson", ann_dir / "test_slide.geojson")

    data_paths = tmp_path / "data_paths.yaml"
    cache_dir = tmp_path / "cache"
    with data_paths.open("w") as f:
        yaml.safe_dump({"annotations": [str(ann_dir)], "cache": str(cache_dir)}, f)

    prepare_main(["--data-paths", str(data_paths)])

    assert (cache_dir / "test_slide.db").exists()


def test_test_cli_generates_predictions(tmp_path, annotation_db, wsi_path):
    db_path, _ = annotation_db

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "path"])
        writer.writeheader()
        writer.writerow({"slide_id": "test_slide", "path": str(db_path)})

    dataset_cfg = {
        "patch_size": 512,
        "class_map": {"T": -1.0, "V": 1.0},
        "resolution": 1.0,
        "min_mask_ratio": 0.0,
    }
    dataset_cfg_path = tmp_path / "dataset.yaml"
    dataset_cfg_path.write_text(yaml.safe_dump(dataset_cfg))

    data_paths = {
        "slides": [str(wsi_path.parent)],
        "annotations": [str(db_path.parent)],
    }
    data_paths_path = tmp_path / "data_paths.yaml"
    data_paths_path.write_text(yaml.safe_dump(data_paths))

    model = torch.jit.script(MeanModel())
    model_path = tmp_path / "model.pt"
    model.save(str(model_path))

    out_csv = tmp_path / "predictions.csv"
    run_test_cli(
        [
            str(manifest),
            str(model_path),
            "--dataset-config",
            str(dataset_cfg_path),
            "--data-paths",
            str(data_paths_path),
            "--output",
            str(out_csv),
        ]
    )

    rows = list(csv.DictReader(out_csv.open()))
    assert rows and {"prediction", "target"} <= rows[0].keys()
