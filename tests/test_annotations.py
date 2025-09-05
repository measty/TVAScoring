import pytest

from tvs.io.annotations import get_class_from_props


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
