"""Tests for H3 feature partition CLI and orchestration."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import SCENE_ID, write_scaled_sr_cog
from wildfire_geo_ml.features.h3_partition import (
    app,
    build_features,
    process_scene,
    read_partitioned_geoparquet,
    resolve_feature_scenes,
)
from wildfire_geo_ml.ingest.config import FeaturesConfig, load_pipeline_config
from wildfire_geo_ml.ingest.landsat_paths import local_band_path


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
