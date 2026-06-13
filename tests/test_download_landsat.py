"""Tests for Landsat scene ID parsing, S3 keys, and download logic."""

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from tests.conftest import (
    BUCKET,
    COLLECTION_PREFIX,
    SCENE_ID,
)
from wildfire_geo_ml.ingest.config import IngestConfig, load_pipeline_config
from wildfire_geo_ml.ingest.download_landsat import (
    download_file,
    download_scene,
    get_landsat_s3_client,
)
from wildfire_geo_ml.ingest.landsat_paths import (
    filter_scenes,
    local_band_path,
    mtl_key,
    normalize_scene_id,
    parse_scene_id,
    sr_band_key,
)

ALL_SCENES = [
    "LC09_L2SP_044032_20240715_20240717_02_T1",
    "LC09_L2SP_044033_20240715_20240717_02_T1",
    "LC09_L2SP_044032_20240731_20240802_02_T1",
]


@pytest.mark.parametrize(
    ("scene_id", "path", "row", "year", "date"),
    [
        ("LC09_L2SP_044032_20240715_20240717_02_T1", "044", "032", 2024, "20240715"),
        ("LC09_L2SP_044033_20240715_20240717_02_T1", "044", "033", 2024, "20240715"),
        ("LC09_L2SP_044032_20240731_20240802_02_T1", "044", "032", 2024, "20240731"),
    ],
)
def test_parse_scene_id(
    scene_id: str,
    path: str,
    row: str,
    year: int,
    date: str,
) -> None:
    meta = parse_scene_id(scene_id)
    assert meta.wrs_path == path
    assert meta.wrs_row == row
    assert meta.year == year
    assert meta.acquisition_date == date


def test_parse_scene_id_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid scene ID"):
        parse_scene_id("not-a-scene")


def test_normalize_scene_id_leaves_canonical_id_unchanged() -> None:
    assert normalize_scene_id(SCENE_ID) == SCENE_ID


def test_normalize_scene_id_strips_trailing_sr() -> None:
    assert normalize_scene_id(f"{SCENE_ID}_SR") == SCENE_ID


def test_build_s3_keys() -> None:
    b4_key = sr_band_key(SCENE_ID, "B4", COLLECTION_PREFIX)
    assert b4_key == (
        f"collection02/level-2/standard/oli-tirs/2024/044/032/{SCENE_ID}/{SCENE_ID}_SR_B4.TIF"
    )
    mtl = mtl_key(SCENE_ID, COLLECTION_PREFIX)
    assert mtl.endswith(f"{SCENE_ID}/{SCENE_ID}_MTL.json")


def test_filter_scenes_by_row_and_date() -> None:
    filtered = filter_scenes(
        ALL_SCENES,
        wrs_path="044",
        rows=["032"],
        dates=["20240715"],
    )
    assert filtered == ["LC09_L2SP_044032_20240715_20240717_02_T1"]


def test_filter_scenes_all_when_no_extra_filters() -> None:
    assert filter_scenes(ALL_SCENES, wrs_path="044") == ALL_SCENES


@mock_aws
def test_download_skips_existing(
    tmp_path: Path,
    ingest_config: IngestConfig,
) -> None:
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.create_bucket(
        Bucket=BUCKET,
        CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
    )
    key = sr_band_key(SCENE_ID, "B4", COLLECTION_PREFIX)
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"data")

    dest = local_band_path(tmp_path, SCENE_ID, "B4")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"already here")

    status = download_file(s3, BUCKET, key, dest, force=False)
    assert status == "skipped"
    assert dest.read_bytes() == b"already here"


def test_download_scene_from_mock_bucket(
    tmp_path: Path,
    ingest_config: IngestConfig,
    mock_landsat_bucket: str,
) -> None:
    del mock_landsat_bucket  # fixture seeds the bucket
    s3 = boto3.client("s3", region_name="us-west-2")
    report = download_scene(s3, ingest_config, SCENE_ID, tmp_path)

    assert len(report.results) == 7
    assert len(report.failed) == 0
    assert len(report.downloaded) == 7

    scene_dir = tmp_path / SCENE_ID
    assert scene_dir.is_dir()
    tif_files = list(scene_dir.glob("*_SR_B*.TIF"))
    assert len(tif_files) == 6
    assert (scene_dir / f"{SCENE_ID}_MTL.json").is_file()


def test_load_pipeline_config() -> None:
    config_path = Path("config/pipeline.yaml")
    if not config_path.is_file():
        pytest.skip("config/pipeline.yaml not present")
    pipeline = load_pipeline_config(config_path)
    assert len(pipeline.ingest.scenes) == 3
    assert pipeline.ingest.bucket == "usgs-landsat"
    assert pipeline.discover.max_cloud_cover == 10.0
    assert len(pipeline.study_area.bbox) == 4


@mock_aws
def test_get_landsat_s3_client() -> None:
    client = get_landsat_s3_client()
    assert callable(client.download_file)
