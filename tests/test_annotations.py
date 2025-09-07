import shutil

import pytest

from tvs.io.annotations import get_class_from_props, prepare_annotations


def test_geojson_to_sqlite_stats(annotation_db):
    db_path, stats = annotation_db
    assert db_path.exists()
    assert stats["T"]["count"] == 2
    assert stats["V"]["count"] == 1
    assert stats["T"]["area"] > 0
    assert stats["V"]["area"] > 0


def test_get_class_from_props():
    assert get_class_from_props({"name": "V1"}) == "V"
    with pytest.raises(ValueError):
        get_class_from_props({}, "name")


def test_prepare_annotations_multiple_sources(tmp_path, test_data_dir):
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    src1.mkdir()
    src2.mkdir()
    shutil.copy(test_data_dir / "test_slide.geojson", src1 / "first.geojson")
    shutil.copy(test_data_dir / "test_slide.geojson", src2 / "second.geojson")

    dest = tmp_path / "out"
    stats = prepare_annotations([src1, src2], dest, cache=False)

    assert (dest / "first.db").exists()
    assert (dest / "second.db").exists()
    assert set(stats.keys()) == {"first", "second"}
