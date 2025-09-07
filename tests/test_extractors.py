import numpy as np
from shapely.geometry import box
from tiatoolbox.annotation import SQLiteStore
from tiatoolbox.wsicore.wsireader import WSIReader

from tvs.data.extractors import build_filters, make_extractors


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

    for label, ex_list in extractors.items():
        assert ex_list
        polys = class_polys[label]
        for extractor in ex_list:
            assert len(extractor) > 0
            for x, y, w, h in extractor.coordinate_list:  # type: ignore[attr-defined]
                patch_poly = box(x, y, x + w, y + h)
                max_cov = max(
                    patch_poly.intersection(poly).area / patch_poly.area
                    for poly in polys
                )
                assert max_cov > 0  # patch overlaps annotation


def test_make_extractors_respects_resolution(annotation_db, wsi_path):
    db_path, _ = annotation_db
    wsi = WSIReader.open(wsi_path)
    filters = build_filters(["T"])
    extractors = make_extractors(
        [(wsi, str(db_path))],
        filters,
        patch_size=512,
        stride=512,
        resolution=1.0,
        min_mask_ratio=0.0,
    )
    extractor = extractors["T"][0]
    assert len(extractor) > 0
    patch = extractor[0]
    if isinstance(patch, tuple):
        patch = patch[0]
    x0, y0, x1, y1 = extractor.coordinate_list[0]
    expected = wsi.read_rect(
        location=(x0, y0), size=(512, 512), resolution=1.0, units="mpp"
    )
    diff = np.abs(patch.astype("int16") - expected.astype("int16")).mean()
    assert diff < 5
