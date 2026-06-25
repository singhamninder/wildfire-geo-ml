"""Tests for H3 feature partition CLI and orchestration."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from typer.testing import CliRunner

from tests.conftest import SCENE_ID, reflectance_to_dn, write_scaled_sr_cog
from wildfire_geo_ml.features.h3_partition import (
    app,
    build_features,
    get_scene_valid_footprint,
    get_scene_wgs84_bbox,
    process_scene,
    read_partitioned_geoparquet,
    resolve_feature_scenes,
)
from wildfire_geo_ml.features.h3_utils import (
    filter_cells_to_extent,
    filter_cells_to_geometry,
    polyfill_bbox,
)
from wildfire_geo_ml.ingest.config import FeaturesConfig, load_pipeline_config
from wildfire_geo_ml.ingest.landsat_paths import local_band_path


def write_footprint_test_cog(path: Path, *, size: int = 256) -> None:
    """Write a UTM COG with valid data in the center and fill (0) in the corners."""
    data = np.zeros((size, size), dtype=np.uint16)
    margin = size // 4
    inner = size - 2 * margin
    data[margin : margin + inner, margin : margin + inner] = reflectance_to_dn(0.2)
    transform = from_origin(600000.0, 4500000.0, 30.0, 30.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint16",
        crs="EPSG:32610",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def feature_scene_cog_dir(tmp_path: Path) -> Path:
    """Create a minimal scene with all spectral-index bands."""
    reflectance = {"B3": 0.3, "B4": 0.1, "B5": 0.5, "B7": 0.2}
    for band, value in reflectance.items():
        cog_path = local_band_path(tmp_path, SCENE_ID, band)
        write_scaled_sr_cog(cog_path, value, width=32, height=32)
    return tmp_path


def test_resolve_feature_scenes_filters(feature_scene_cog_dir: Path) -> None:
    scenes = resolve_feature_scenes(feature_scene_cog_dir, wrs_path="044", rows=["032"])
    assert scenes == [SCENE_ID]


def test_get_scene_valid_footprint_excludes_fill_corners(tmp_path: Path) -> None:
    cog_path = tmp_path / "partial_valid.tif"
    write_footprint_test_cog(cog_path)
    result = get_scene_valid_footprint(cog_path)
    assert result is not None
    footprint, crs = result
    assert crs == "EPSG:32610"

    scene_bbox = get_scene_wgs84_bbox(cog_path)
    west, south, east, north = scene_bbox
    padded_bbox = (west - 0.02, south - 0.02, east + 0.02, north + 0.02)
    study_cells = polyfill_bbox(padded_bbox, resolution=8)
    bbox_cells = filter_cells_to_extent(study_cells, scene_bbox)
    footprint_cells = filter_cells_to_geometry(study_cells, footprint, crs)
    assert 0 < len(footprint_cells) <= len(bbox_cells)
    assert len(footprint_cells) < len(bbox_cells)
    assert set(footprint_cells).issubset(set(bbox_cells))


def test_process_scene_returns_expected_columns(
    feature_scene_cog_dir: Path,
    pipeline_config_path: Path,
) -> None:
    pipeline = load_pipeline_config(pipeline_config_path)
    # Small bbox around the synthetic raster center (Chico-area H3 cell).
    study_bbox = (-122.0, 39.5, -121.5, 40.0)
    gdf = process_scene(SCENE_ID, feature_scene_cog_dir, study_bbox, pipeline.features)
    if gdf.empty:
        pytest.skip("Synthetic raster did not intersect any H3 cells in study bbox")

    for column in (
        "h3_index",
        "h3_res8",
        "scene_id",
        "acquisition_date",
        "ndvi_mean",
        "nbr_mean",
        "ndwi_mean",
        "geometry",
    ):
        assert column in gdf.columns


def test_build_features_writes_partitioned_parquet(
    feature_scene_cog_dir: Path,
    pipeline_config_path: Path,
    tmp_path: Path,
) -> None:
    pipeline = load_pipeline_config(pipeline_config_path)
    output_dir = tmp_path / "features"
    study_bbox = (-122.0, 39.5, -121.5, 40.0)

    out_path = build_features(
        feature_scene_cog_dir,
        [SCENE_ID],
        pipeline,
        study_bbox=study_bbox,
        output_dir=output_dir,
    )

    assert out_path == output_dir
    assert any(output_dir.iterdir()), "Expected partitioned GeoParquet output"
    gdf = read_partitioned_geoparquet(output_dir)
    assert "ndvi_mean" in gdf.columns
    assert (output_dir / f"h3_res8={gdf['h3_res8'].iloc[0]}").is_dir()


def test_cli_builds_features(
    feature_scene_cog_dir: Path,
    pipeline_config_path: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cli_features"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--config",
            str(pipeline_config_path),
            "--cog-dir",
            str(feature_scene_cog_dir),
            "--output",
            str(output_dir),
            "--bbox",
            "-122.0,39.5,-121.5,40.0",
        ],
    )
    assert result.exit_code == 0, result.output


def test_features_config_defaults() -> None:
    config = FeaturesConfig()
    assert config.h3_resolution == 8
    assert config.required_bands == ["B3", "B4", "B5", "B7"]
