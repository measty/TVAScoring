from shapely.geometry import box
from tiatoolbox.annotation import SQLiteStore


def test_extractors_patch_coverage(annotation_db, extractors):
    db_path, _ = annotation_db

    # Load polygons grouped by class label
    store = SQLiteStore.open(str(db_path))
    class_polys = {"T": [], "V": []}
    try:
        for ann in store.values():
            label = ann.properties["name"][0]
            class_polys[label].append(ann.geometry)
    finally:
        store.close()

    for label, extractor in extractors.items():
        assert len(extractor) > 0
        polys = class_polys[label]
        for x, y, w, h in extractor.coordinate_list:  # type: ignore[attr-defined]
            patch_poly = box(x, y, x + w, y + h)
            max_cov = max(
                patch_poly.intersection(poly).area / patch_poly.area for poly in polys
            )
            assert max_cov > 0  # patch overlaps annotation
