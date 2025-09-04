import json
import sqlite3
from pathlib import Path

import pytest

from tvs.io.annotations import geojson_to_sqlite, prepare_annotations

if not hasattr(sqlite3.Connection, "enable_load_extension"):
    pytest.skip(
        "SQLite extension loading not supported",
        allow_module_level=True,
    )


def _write_geojson(path: Path, features: list[dict]) -> None:
    data = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(data))


def _polygon(x0: float, y0: float, x1: float, y1: float, label: str) -> dict:
    return {
        "type": "Feature",
        "properties": {"class": label},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [x0, y0],
                    [x0, y1],
                    [x1, y1],
                    [x1, y0],
                    [x0, y0],
                ]
            ],
        },
    }


def test_geojson_to_sqlite_creates_db_and_stats(tmp_path: Path) -> None:
    geojson_path = tmp_path / "slide.geojson"
    _write_geojson(
        geojson_path,
        [
            _polygon(0, 0, 10, 10, "tubular"),
            _polygon(10, 0, 20, 10, "villous"),
        ],
    )
    db_path = tmp_path / "slide.db"
    stats = geojson_to_sqlite(geojson_path, db_path)
    assert db_path.exists()
    assert stats["tubular"]["count"] == 1
    assert stats["villous"]["area"] == pytest.approx(100.0)


def test_prepare_annotations_batch(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    _write_geojson(src / "a.geojson", [_polygon(0, 0, 5, 5, "villous")])
    _write_geojson(src / "b.geojson", [_polygon(0, 0, 5, 5, "tubular")])
    stats = prepare_annotations(src, dest)
    assert (dest / "a.db").exists()
    assert (dest / "b.db").exists()
    assert set(stats.keys()) == {"a", "b"}
    assert stats["a"]["villous"]["count"] == 1


def test_missing_class_raises(tmp_path: Path) -> None:
    geojson_path = tmp_path / "bad.geojson"
    feature = {
        "type": "Feature",
        "properties": {},  # Missing class
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
        },
    }
    _write_geojson(geojson_path, [feature])
    with pytest.raises(ValueError, match="missing the 'class'"):
        geojson_to_sqlite(geojson_path, tmp_path / "bad.db")


def test_cache_behavior(tmp_path: Path) -> None:
    geojson_path = tmp_path / "slide.geojson"
    _write_geojson(geojson_path, [_polygon(0, 0, 5, 5, "tubular")])
    db_path = tmp_path / "slide.db"
    first = geojson_to_sqlite(geojson_path, db_path)
    # Modify source file; stats should remain same when cache=True
    _write_geojson(geojson_path, [_polygon(0, 0, 10, 10, "tubular")])
    cached = geojson_to_sqlite(geojson_path, db_path, cache=True)
    assert cached == first
    # Recompute when cache disabled
    refreshed = geojson_to_sqlite(geojson_path, db_path, cache=False)
    assert refreshed != first
