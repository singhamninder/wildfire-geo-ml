"""Tests for H3 polyfill and geometry helpers."""

import h3
from shapely.geometry import box

from wildfire_geo_ml.features.h3_utils import (
    filter_cells_to_extent,
    filter_cells_to_geometry,
    h3_cells_to_geodataframe,
    polyfill_bbox,
)

# Small Chico-area subset for fast tests
CHICO_BBOX = (-121.9, 39.6, -121.6, 39.8)


def test_polyfill_bbox_returns_res8_cells() -> None:
    cells = polyfill_bbox(CHICO_BBOX, resolution=8)
    assert len(cells) > 0
    assert all(h3.get_resolution(cell) == 8 for cell in cells)


def test_polyfill_bbox_cell_centroids_within_extent() -> None:
    cells = polyfill_bbox(CHICO_BBOX, resolution=8)
    west, south, east, north = CHICO_BBOX
    for cell in cells[:20]:
        lat, lng = h3.cell_to_latlng(cell)
        assert south <= lat <= north
        assert west <= lng <= east


def test_h3_cells_to_geodataframe_schema() -> None:
    cells = polyfill_bbox(CHICO_BBOX, resolution=8)[:5]
    gdf = h3_cells_to_geodataframe(cells)
    assert list(gdf.columns) == ["h3_index", "h3_res", "geometry"]
    assert len(gdf) == 5
    assert gdf.crs.to_epsg() == 4326
    assert gdf.geometry.is_valid.all()


def test_filter_cells_to_extent_reduces_count() -> None:
    study_cells = polyfill_bbox(CHICO_BBOX, resolution=8)
    scene_bbox = (-121.75, 39.65, -121.65, 39.75)
    filtered = filter_cells_to_extent(study_cells, scene_bbox)
    assert 0 < len(filtered) < len(study_cells)
    assert set(filtered).issubset(set(study_cells))


def test_filter_cells_to_geometry_keeps_intersecting_cells() -> None:
    study_cells = polyfill_bbox(CHICO_BBOX, resolution=8)
    # Small WGS84 box, then use its UTM envelope as the filter geometry.
    scene_bbox = (-121.75, 39.65, -121.65, 39.75)
    bbox_cells = filter_cells_to_extent(study_cells, scene_bbox)
    hex_gdf = h3_cells_to_geodataframe(bbox_cells)
    utm_geom = box(*hex_gdf.to_crs("EPSG:32610").total_bounds)

    filtered = filter_cells_to_geometry(bbox_cells, utm_geom, "EPSG:32610")
    assert len(filtered) == len(bbox_cells)
    assert set(filtered).issubset(set(study_cells))

    outside = filter_cells_to_geometry(study_cells, utm_geom, "EPSG:32610")
    assert 0 < len(outside) < len(study_cells)
    assert set(outside).issubset(set(study_cells))
