import csv
import shutil

from tvs.cli.preview import _load_dataset


def test_load_dataset_respects_slide_extension(tmp_path, annotation_db, test_data_dir):
    db_path, _ = annotation_db
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    shutil.copy(test_data_dir / "test_slide.svs", slides_dir / "test_slide.tif")

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
    data_paths = {"slides": str(slides_dir)}

    dataset = _load_dataset(manifest, dataset_cfg, data_paths)
    assert len(dataset) > 0
