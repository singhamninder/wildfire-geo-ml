"""Tests for Sedona and corridor config sections."""

from pathlib import Path

from wildfire_geo_ml.ingest.config import load_pipeline_config


def test_pipeline_config_loads_sedona_and_corridors() -> None:
    config_path = Path("config/pipeline.yaml")
    pipeline = load_pipeline_config(config_path)
    assert pipeline.features.indices_dir == "data/indices"
    assert "sedona-spark-4.0_2.13" in pipeline.sedona.jar_packages
    assert pipeline.corridors is not None
    assert pipeline.corridors.buffer_m == 100.0
    assert pipeline.corridors.cached_geojson.endswith(".geojson")
