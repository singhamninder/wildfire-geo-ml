"""Shared pytest fixtures."""

import os
import shutil
import subprocess
from pathlib import Path

import boto3
import h3
import numpy as np
import pytest
import rasterio
from moto import mock_aws
from pyproj import Transformer
from rasterio.transform import from_origin

from wildfire_geo_ml.features.indices import LANDSAT9_OFFSET, LANDSAT9_SCALE
from wildfire_geo_ml.ingest.config import IngestConfig
from wildfire_geo_ml.ingest.landsat_paths import mtl_key, sr_band_key

SCENE_ID = "LC09_L2SP_044032_20240715_20240717_02_T1"
BUCKET = "usgs-landsat"
COLLECTION_PREFIX = "collection02/level-2/standard/oli-tirs"
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
FEATURE_BANDS = ["B3", "B4", "B5", "B7"]


def _java_major_version(java_executable: str) -> int | None:
    """Return the major Java version for an executable, if detectable."""
    proc = subprocess.run(
        [java_executable, "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stderr + proc.stdout
    if 'version "1.8' in combined:
        return 8
    for major in range(25, 16, -1):
        if f'version "{major}' in combined or f"version {major}." in combined:
            return major
    return None


def resolve_java17_home() -> str | None:
    """
    Return JAVA_HOME for Java 17+ when available.

    PySpark 4.1 requires Java 17. Sedona slow tests skip when only Java 8/11
    is on PATH.
    """
    java_home_tool = Path("/usr/libexec/java_home")
    if java_home_tool.is_file():
        for version_flag in ("25", "24", "23", "22", "21", "17"):
            proc = subprocess.run(
                [str(java_home_tool), "-v", version_flag],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                candidate = proc.stdout.strip()
                major = _java_major_version(str(Path(candidate) / "bin" / "java"))
                if major is not None and major >= 17:
                    return candidate
    java_executable = shutil.which("java")
    if java_executable is None:
        return None
    major = _java_major_version(java_executable)
    if major is not None and major >= 17:
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            return java_home
        java_path = Path(java_executable).resolve()
        if java_path.parent.name == "bin":
            return str(java_path.parent.parent)
    return None


@pytest.fixture
def java17_home(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set JAVA_HOME to Java 17+ or skip Sedona/Spark tests."""
    home = resolve_java17_home()
    if home is None:
        pytest.skip("Java 17+ required for PySpark 4.1 / Sedona tests")
    monkeypatch.setenv("JAVA_HOME", home)
    monkeypatch.setenv("PATH", f"{home}/bin:{os.environ.get('PATH', '')}")
    return home


def reflectance_to_dn(reflectance: float) -> int:
    """Convert surface reflectance to Landsat-9 L2 SR digital number."""
    dn = round((reflectance - LANDSAT9_OFFSET) / LANDSAT9_SCALE)
    return max(1, dn)


def write_scaled_sr_cog(
    path: Path,
    reflectance: float | np.ndarray,
    *,
    width: int = 4,
    height: int = 4,
    origin_x: float | None = None,
    origin_y: float | None = None,
    pixel_size: float = 30.0,
) -> None:
    """
    Write a small single-band GeoTIFF with Landsat SR scaling in EPSG:32610.

    Parameters
    ----------
    path : Path
        Output GeoTIFF path.
    reflectance : float or ndarray
        Surface reflectance value(s) to encode as DN.
    width : int
        Raster width in pixels.
    height : int
        Raster height in pixels.
    origin_x : float
        Upper-left X coordinate in UTM.
    origin_y : float
        Upper-left Y coordinate in UTM.
    pixel_size : float
        Pixel size in meters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.isscalar(reflectance):
        data = np.full((height, width), reflectance_to_dn(float(reflectance)), dtype=np.uint16)
    else:
        arr = np.asarray(reflectance, dtype=np.float64)
        vectorized = np.vectorize(lambda v: reflectance_to_dn(float(v)))
        data = vectorized(arr).astype(np.uint16)

    if origin_x is None or origin_y is None:
        cell = "882815d0c1fffff"
        lat, lng = h3.cell_to_latlng(cell)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
        center_x, center_y = transformer.transform(lng, lat)
        span_x = width * pixel_size
        span_y = height * pixel_size
        origin_x = center_x - span_x / 2
        origin_y = center_y + span_y / 2

    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs="EPSG:32610",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def ingest_config() -> IngestConfig:
    """Minimal ingest config for tests."""
    return IngestConfig(
        bucket=BUCKET,
        collection_prefix=COLLECTION_PREFIX,
        bands=BANDS,
        wrs_path="044",
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
study_area:
  bbox: [-122.5, 39.0, -120.0, 41.5]
  datetime: "2024-07-01/2024-08-31"

discover:
  stac_api_url: "https://landsatlook.usgs.gov/stac-server"
  collection: "landsat-c2l2-sr"
  platform: "LANDSAT_9"
  max_cloud_cover: 10

ingest:
  bucket: {BUCKET}
  collection_prefix: {COLLECTION_PREFIX}
  bands: {BANDS}
  wrs_path: "044"

features:
  h3_resolution: 8
  stat_names: [mean, std]
  required_bands: [B3, B4, B5, B7]
  output_dir: data/features/h3_partitioned
""",
        encoding="utf-8",
    )
    return config_file
