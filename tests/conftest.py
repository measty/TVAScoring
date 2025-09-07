import sys
from pathlib import Path

import pytest

# Ensure the src/ directory is on the Python path for tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tvs.data.extractors import build_filters, make_extractors
from tvs.io.annotations import geojson_to_sqlite


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def wsi_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test_slide.svs"


@pytest.fixture(scope="session")
def geojson_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test_slide.geojson"


@pytest.fixture()
def annotation_db(tmp_path: Path, geojson_path: Path):
    """Create a temporary SQLite annotation store for tests."""

    db_path = tmp_path / "test_slide.db"
    stats = geojson_to_sqlite(geojson_path, db_path, cache=False)
    return db_path, stats


@pytest.fixture()
def extractors(annotation_db, wsi_path):
    db_path, _ = annotation_db
    filters = build_filters(["T", "V"])
    return make_extractors(
        [(wsi_path, str(db_path))],
        filters,
        patch_size=512,
        stride=512,
        min_mask_ratio=0.5,
    )
