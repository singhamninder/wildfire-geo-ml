"""Shared pytest fixtures."""

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from wildfire_geo_ml.ingest.config import IngestConfig
from wildfire_geo_ml.ingest.landsat_paths import mtl_key, sr_band_key

SCENE_ID = "LC09_L2SP_044032_20240715_20240717_02_T1"
BUCKET = "usgs-landsat"
COLLECTION_PREFIX = "collection02/level-2/standard/oli-tirs"
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]


@pytest.fixture
def ingest_config() -> IngestConfig:
    """Minimal ingest config for tests."""
    return IngestConfig(
        bucket=BUCKET,
        collection_prefix=COLLECTION_PREFIX,
        bands=BANDS,
        wrs_path="044",
        scenes=[SCENE_ID],
    )


@pytest.fixture
def mock_landsat_bucket(ingest_config: IngestConfig) -> str:
    """
    Seed a moto S3 bucket with dummy Landsat objects for one scene.

    Yields
    ------
    str
        Bucket name.
    """
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-west-2")
        s3.create_bucket(
            Bucket=ingest_config.bucket,
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        for band in ingest_config.bands:
            key = sr_band_key(SCENE_ID, band, ingest_config.collection_prefix)
            s3.put_object(Bucket=ingest_config.bucket, Key=key, Body=b"fake-tif")
        mtl = mtl_key(SCENE_ID, ingest_config.collection_prefix)
        s3.put_object(Bucket=ingest_config.bucket, Key=mtl, Body=b'{"test": true}')
        yield ingest_config.bucket


@pytest.fixture
def pipeline_config_path(tmp_path: Path) -> Path:
    """Write a minimal pipeline.yaml for CLI tests."""
    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text(
        f"""
ingest:
  bucket: {BUCKET}
  collection_prefix: {COLLECTION_PREFIX}
  bands: {BANDS}
  wrs_path: "044"
  scenes:
    - {SCENE_ID}
    - LC09_L2SP_044033_20240715_20240717_02_T1
    - LC09_L2SP_044032_20240731_20240802_02_T1
""",
        encoding="utf-8",
    )
    return config_file
