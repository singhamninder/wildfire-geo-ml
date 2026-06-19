"""Slow integration tests for Sedona corridor zonal statistics."""

import json
from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString, mapping

from wildfire_geo_ml.features.indices import write_scene_index_cogs
from wildfire_geo_ml.sedona.session import create_sedona_session
from wildfire_geo_ml.sedona.zonal_stats_job import (
    load_corridor_regions,
    run_zonal_stats_job,
    stop_spark,
)


def _make_uniform_index_cog(indices_dir: Path, scene_id: str, value: float = 0.5) -> None:
    """Write a small float GeoTIFF index COG centered on Northern California."""
    values = np.full((32, 32), value, dtype=np.float32)
    transform = from_origin(600000.0, 4500000.0, 30.0, 30.0)
    da = xr.DataArray(values, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:32610")
    da = da.rio.write_transform(transform)
    write_scene_index_cogs({"ndvi": da}, indices_dir, scene_id)


def _write_corridor_geojson(path: Path) -> None:
    """Write a short transmission line crossing the synthetic raster extent."""
    transformer = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    line_utm = LineString([(599500.0, 4499500.0), (601000.0, 4500500.0)])
    coords = [transformer.transform(x, y) for x, y in line_utm.coords]
    feature = {
        "type": "Feature",
        "geometry": mapping(LineString(coords)),
        "properties": {"OBJECTID": 1, "TLine Name": "TEST_LINE"},
    }
    payload = {"type": "FeatureCollection", "features": [feature]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.slow
def test_corridor_zonal_stats_local_spark(tmp_path: Path, java17_home: str) -> None:
    scene_id = "LC09_TEST_SCENE"
    indices_dir = tmp_path / "indices"
    _make_uniform_index_cog(indices_dir, scene_id)

    lines_path = tmp_path / "lines.geojson"
    _write_corridor_geojson(lines_path)
    output_dir = tmp_path / "corridor_stats"

    spark = create_sedona_session(app_name="wildfire-corridor-zonal-test")
    try:
        regions_df = load_corridor_regions(
            spark,
            lines_path,
            buffer_m=100.0,
            metric_crs="EPSG:32610",
        )
        run_zonal_stats_job(
            spark,
            indices_dir=indices_dir,
            regions_df=regions_df,
            region_kind="corridor",
            output_dir=output_dir,
            stat_names=["mean", "std"],
        )
        result = spark.read.format("geoparquet").load(str(output_dir))
        assert result.count() >= 1
        if "ndvi_mean" in result.columns:
            sample = result.select("ndvi_mean").collect()[0][0]
            assert sample is not None
            assert -1.0 <= float(sample) <= 1.0
    finally:
        stop_spark(spark)
