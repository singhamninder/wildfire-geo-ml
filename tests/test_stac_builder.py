"""Tests for STAC catalog builder and MTL parser."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin

from tests.conftest import BANDS, COLLECTION_PREFIX, SCENE_ID
from wildfire_geo_ml.ingest.config import IngestConfig
from wildfire_geo_ml.ingest.landsat_paths import local_band_path, local_mtl_path, sr_band_key
from wildfire_geo_ml.stac_builder.build_catalog import (
    build_and_save_catalog,
    build_catalog_from_dirs,
    collect_cog_paths,
    create_collection,
    create_item_from_cog,
    main,
    resolve_catalog_scenes,
)
from wildfire_geo_ml.stac_builder.mtl_parser import parse_mtl_dict, parse_mtl_file

MTL_FIXTURE: dict = {
    "LANDSAT_METADATA_FILE": {
        "IMAGE_ATTRIBUTES": {
            "CLOUD_COVER_LAND": "4.12",
            "SUN_ELEVATION_LAND": "65.5",
            "SUN_AZIMUTH_LAND": "145.2",
            "DATE_ACQUIRED": "2024-07-15",
            "SCENE_CENTER_TIME": "18:45:23.1234560Z",
        },
        "PROJECTION_ATTRIBUTES": {
            "UTM_ZONE": "10",
            "GRID_CELL_SIZE_REFLECTIVE": "30.000",
            "REFLECTIVE_LINES": "8031",
            "REFLECTIVE_SAMPLES": "8001",
            "CORNER_UL_PROJECTION_X_PRODUCT": "600000.000",
            "CORNER_UL_PROJECTION_Y_PRODUCT": "4500000.000",
        },
    }
}


def _write_tiny_cog(path: Path) -> None:
    """Write a small single-band GeoTIFF in EPSG:32610 for spatial metadata tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(600000.0, 4500000.0, 30.0, 30.0)
    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint16",
        crs="EPSG:32610",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def mtl_metadata() -> dict:
    """Parsed metadata dict from :data:`MTL_FIXTURE`."""
    return parse_mtl_dict(MTL_FIXTURE)


@pytest.fixture
def scene_cog_paths(tmp_path: Path) -> dict[str, Path]:
    """Minimal B4+B5 COG paths for one scene."""
    paths: dict[str, Path] = {}
    for band in ("B4", "B5"):
        cog_path = local_band_path(tmp_path, SCENE_ID, band)
        _write_tiny_cog(cog_path)
        paths[band] = cog_path
    return paths


def test_parse_mtl_dict_extracts_stac_fields(mtl_metadata: dict) -> None:
    assert mtl_metadata["eo:cloud_cover"] == pytest.approx(4.12)
    assert mtl_metadata["view:sun_elevation"] == pytest.approx(65.5)
    assert mtl_metadata["view:sun_azimuth"] == pytest.approx(145.2)
    assert mtl_metadata["proj:shape"] == [8031, 8001]
    assert mtl_metadata["proj:transform"] == [30.0, 0.0, 600000.0, 0.0, -30.0, 4500000.0]
    assert mtl_metadata["proj:epsg"] == 32610
    assert mtl_metadata["datetime"] == datetime(2024, 7, 15, 18, 45, 23, tzinfo=UTC)


def test_parse_mtl_file(tmp_path: Path) -> None:
    mtl_path = tmp_path / f"{SCENE_ID}_MTL.json"
    mtl_path.write_text(json.dumps(MTL_FIXTURE), encoding="utf-8")
    metadata = parse_mtl_file(mtl_path)
    assert metadata["eo:cloud_cover"] == pytest.approx(4.12)


def test_create_item_from_cog_validates(
    scene_cog_paths: dict[str, Path],
    mtl_metadata: dict,
    ingest_config: IngestConfig,
) -> None:
    item = create_item_from_cog(SCENE_ID, scene_cog_paths, mtl_metadata, ingest_config)

    assert item.id == SCENE_ID
    assert item.bbox is not None
    assert len(item.bbox) == 4
    assert "B4" in item.assets
    assert "B5" in item.assets
    assert "MTL_JSON" in item.assets
    assert item.assets["B4"].href.startswith("s3://usgs-landsat/")
    assert item.properties.get("eo:cloud_cover") == pytest.approx(4.12)


def test_create_item_local_hrefs(
    scene_cog_paths: dict[str, Path],
    mtl_metadata: dict,
    ingest_config: IngestConfig,
    tmp_path: Path,
) -> None:
    mtl_path = tmp_path / f"{SCENE_ID}_MTL.json"
    mtl_path.write_text("{}", encoding="utf-8")
    item = create_item_from_cog(
        SCENE_ID,
        scene_cog_paths,
        mtl_metadata,
        ingest_config,
        local_hrefs=True,
        mtl_path=mtl_path,
    )
    assert item.assets["B4"].href == scene_cog_paths["B4"].as_posix()


def test_create_collection_union_extent(
    scene_cog_paths: dict[str, Path],
    mtl_metadata: dict,
    ingest_config: IngestConfig,
) -> None:
    item_a = create_item_from_cog(SCENE_ID, scene_cog_paths, mtl_metadata, ingest_config)
    other_id = "LC09_L2SP_044033_20240715_20240717_02_T1"
    other_paths = {
        "B4": scene_cog_paths["B4"],
        "B5": scene_cog_paths["B5"],
    }
    item_b = create_item_from_cog(other_id, other_paths, mtl_metadata, ingest_config)

    collection = create_collection([item_a, item_b])
    assert collection.id == "landsat-9-northern-ca-2024"
    assert collection.extent.temporal is not None
    assert len(list(collection.get_items())) == 2


def test_build_and_save_catalog_writes_json(
    scene_cog_paths: dict[str, Path],
    mtl_metadata: dict,
    ingest_config: IngestConfig,
    tmp_path: Path,
) -> None:
    item = create_item_from_cog(SCENE_ID, scene_cog_paths, mtl_metadata, ingest_config)
    collection = create_collection([item])
    output_dir = tmp_path / "stac"

    catalog = build_and_save_catalog(collection, output_dir)

    assert (output_dir / "catalog.json").is_file()
    assert catalog.id == "wildfire-vegetation-risk"
    assert len(list(catalog.get_items(recursive=True))) == 1


def test_collect_cog_paths(tmp_path: Path) -> None:
    for band in ("B4", "B5"):
        _write_tiny_cog(local_band_path(tmp_path, SCENE_ID, band))
    paths = collect_cog_paths(tmp_path, SCENE_ID, BANDS)
    assert set(paths.keys()) == {"B4", "B5"}


def test_resolve_catalog_scenes(tmp_path: Path) -> None:
    _write_tiny_cog(local_band_path(tmp_path, SCENE_ID, "B4"))
    scenes = resolve_catalog_scenes(tmp_path, wrs_path="044")
    assert scenes == [SCENE_ID]


def test_build_catalog_from_dirs_integration(
    tmp_path: Path,
    ingest_config: IngestConfig,
) -> None:
    cog_dir = tmp_path / "cog"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "stac"

    for band in BANDS:
        _write_tiny_cog(local_band_path(cog_dir, SCENE_ID, band))
    mtl_path = local_mtl_path(raw_dir, SCENE_ID)
    mtl_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path.write_text(json.dumps(MTL_FIXTURE), encoding="utf-8")

    catalog = build_catalog_from_dirs(
        cog_dir,
        raw_dir,
        output_dir,
        ingest_config,
        [SCENE_ID],
        local_hrefs=True,
    )

    assert (output_dir / "catalog.json").is_file()
    items = list(catalog.get_items(recursive=True))
    assert len(items) == 1
    assert items[0].id == SCENE_ID


def test_cli_smoke(tmp_path: Path, pipeline_config_path: Path) -> None:
    cog_dir = tmp_path / "cog"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "stac"

    for band in ("B4", "B5"):
        _write_tiny_cog(local_band_path(cog_dir, SCENE_ID, band))
    mtl_path = local_mtl_path(raw_dir, SCENE_ID)
    mtl_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path.write_text(json.dumps(MTL_FIXTURE), encoding="utf-8")

    mock_catalog = MagicMock()
    mock_catalog.get_items.return_value = []
    with patch(
        "wildfire_geo_ml.stac_builder.build_catalog.build_catalog_from_dirs",
        return_value=mock_catalog,
    ) as mock_build:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config",
                str(pipeline_config_path),
                "--cog-dir",
                str(cog_dir),
                "--raw-dir",
                str(raw_dir),
                "--output-dir",
                str(output_dir),
                "--local-hrefs",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_build.assert_called_once()


def test_s3_band_key_matches_ingest_config(ingest_config: IngestConfig) -> None:
    key = sr_band_key(SCENE_ID, "B4", ingest_config.collection_prefix)
    assert key == (f"{COLLECTION_PREFIX}/2024/044/032/{SCENE_ID}/{SCENE_ID}_SR_B4.TIF")
