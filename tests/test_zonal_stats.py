"""Tests for H3 zonal statistics over spectral index rasters."""

import h3
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_origin

from wildfire_geo_ml.features.h3_utils import filter_cells_to_extent, polyfill_bbox
from wildfire_geo_ml.features.zonal_stats import compute_scene_h3_stats


def _utm_bounds_to_wgs84_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[float, float, float, float]:
    """Convert UTM Zone 10N bounds to EPSG:4326 west/south/east/north."""
    transformer = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    corners = [
        transformer.transform(west, south),
        transformer.transform(east, south),
        transformer.transform(east, north),
        transformer.transform(west, north),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return (min(lons), min(lats), max(lons), max(lats))


def _make_index_array(
    values: np.ndarray,
    *,
    origin_x: float | None = None,
    origin_y: float | None = None,
    pixel_size: float = 30.0,
) -> xr.DataArray:
    """Build a geo-referenced index DataArray in EPSG:32610."""
    if origin_x is None or origin_y is None:
        # Center a raster on a known Northern California H3 res-8 cell.
        cell = "882815d0c1fffff"
        lat, lng = h3.cell_to_latlng(cell)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
        center_x, center_y = transformer.transform(lng, lat)
        height, width = values.shape
        span_x = width * pixel_size
        span_y = height * pixel_size
        origin_x = center_x - span_x / 2
        origin_y = center_y + span_y / 2

    height, width = values.shape
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    da = xr.DataArray(
        values[np.newaxis, :, :],
        dims=("band", "y", "x"),
        coords={"band": [1]},
    )
    da = da.rio.write_crs("EPSG:32610")
    da = da.rio.write_transform(transform)
    return da.isel(band=0)


def test_compute_scene_h3_stats_mean_std() -> None:
    values = np.array(
        [
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
        ],
        dtype=np.float64,
    )
    # 32x32 px ~= 960 m span — large enough for at least one H3 res-8 cell.
    values = np.tile(values, (8, 8))
    ndvi = _make_index_array(values)
    west, south, east, north = ndvi.rio.bounds()
    raster_bbox = _utm_bounds_to_wgs84_bbox(west, south, east, north)
    cells = filter_cells_to_extent(polyfill_bbox(raster_bbox, resolution=8), raster_bbox)
    assert len(cells) >= 1

    gdf = compute_scene_h3_stats({"ndvi": ndvi}, cells, stat_names=["mean", "std"])
    assert len(gdf) >= 1
    assert "ndvi_mean" in gdf.columns
    assert "ndvi_std" in gdf.columns
    assert gdf["ndvi_mean"].between(-1.0, 1.0).all()
    assert (gdf["pixel_count"] > 0).all()


def test_compute_scene_h3_stats_uniform_raster_exact_mean() -> None:
    values = np.full((32, 32), 0.5, dtype=np.float64)
    ndvi = _make_index_array(values)
    west, south, east, north = ndvi.rio.bounds()
    raster_bbox = _utm_bounds_to_wgs84_bbox(west, south, east, north)
    cells = filter_cells_to_extent(polyfill_bbox(raster_bbox, 8), raster_bbox)
    gdf = compute_scene_h3_stats({"ndvi": ndvi}, cells)

    assert len(gdf) >= 1
    assert gdf["ndvi_mean"].iloc[0] == pytest.approx(0.5, abs=0.05)
    assert gdf["ndvi_std"].iloc[0] == pytest.approx(0.0, abs=0.05)


def test_compute_scene_h3_stats_empty_cells() -> None:
    gdf = compute_scene_h3_stats({"ndvi": _make_index_array(np.ones((2, 2)))}, [])
    assert len(gdf) == 0
