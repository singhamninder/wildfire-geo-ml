"""Tests for Landsat STAC discovery over AOI."""

from unittest.mock import MagicMock, patch

import pytest
from pystac import Item

from wildfire_geo_ml.ingest.config import (
    DiscoverConfig,
    IngestConfig,
    PipelineConfig,
    StudyAreaConfig,
)
from wildfire_geo_ml.ingest.discover_scenes import discover_scenes, resolve_scene_list
from wildfire_geo_ml.ingest.landsat_paths import normalize_scene_id, sr_band_key

SCENE_A = "LC09_L2SP_044032_20240715_20240717_02_T1"
SCENE_A_STAC = f"{SCENE_A}_SR"
SCENE_B = "LC09_L2SP_044033_20240715_20240717_02_T1"
SCENE_CLOUDY = "LC09_L2SP_044032_20240731_20240802_02_T1"


def _item(scene_id: str, cloud: float) -> Item:
    return Item.from_dict(
        {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": scene_id,
            "geometry": None,
            "bbox": [-122.0, 39.5, -121.0, 40.5],
            "properties": {
                "datetime": "2024-07-15T00:00:00Z",
                "eo:cloud_cover": cloud,
            },
            "links": [],
            "assets": {},
        }
    )


@pytest.fixture
def study_area() -> StudyAreaConfig:
    return StudyAreaConfig(
        bbox=[-122.5, 39.0, -120.0, 41.5],
        datetime="2024-07-01/2024-08-31",
    )


@pytest.fixture
def discover_config() -> DiscoverConfig:
    return DiscoverConfig(max_cloud_cover=10.0)


@pytest.fixture
def pipeline_config(study_area: StudyAreaConfig, discover_config: DiscoverConfig) -> PipelineConfig:
    return PipelineConfig(
        study_area=study_area,
        discover=discover_config,
        ingest=IngestConfig(
            wrs_path="044",
            scenes=[SCENE_A, SCENE_B, SCENE_CLOUDY],
        ),
    )


def _mock_stac_search(items: list[Item]) -> MagicMock:
    mock_client = MagicMock()
    mock_search = MagicMock()
    mock_search.items.return_value = iter(items)
    mock_client.search.return_value = mock_search
    return mock_client


@patch("wildfire_geo_ml.ingest.discover_scenes.Client.open")
def test_discover_builds_stac_query_with_cloud_filter(
    mock_open: MagicMock,
    study_area: StudyAreaConfig,
    discover_config: DiscoverConfig,
) -> None:
    mock_open.return_value = _mock_stac_search([_item(SCENE_A, 5.0)])

    discover_scenes(study_area, discover_config)

    mock_open.return_value.search.assert_called_once_with(
        collections=["landsat-c2l2-sr"],
        bbox=study_area.bbox,
        datetime=study_area.datetime,
        query={
            "eo:cloud_cover": {"lt": 10.0},
            "platform": {"eq": "LANDSAT_9"},
        },
    )


@patch("wildfire_geo_ml.ingest.discover_scenes.Client.open")
def test_discover_excludes_high_cloud_client_side(
    mock_open: MagicMock,
    study_area: StudyAreaConfig,
    discover_config: DiscoverConfig,
) -> None:
    mock_open.return_value = _mock_stac_search(
        [
            _item(SCENE_A, 5.0),
            _item(SCENE_CLOUDY, 12.0),
            _item(SCENE_B, 8.0),
        ]
    )

    results = discover_scenes(study_area, discover_config)

    assert [scene.scene_id for scene in results] == [SCENE_A, SCENE_B]
    assert results[0].cloud_cover == 5.0
    assert results[1].cloud_cover == 8.0


@patch("wildfire_geo_ml.ingest.discover_scenes.Client.open")
def test_discover_deduplicates_and_sorts_by_cloud(
    mock_open: MagicMock,
    study_area: StudyAreaConfig,
    discover_config: DiscoverConfig,
) -> None:
    mock_open.return_value = _mock_stac_search(
        [
            _item(SCENE_A, 9.0),
            _item(SCENE_A, 3.0),
            _item(SCENE_B, 7.0),
        ]
    )

    results = discover_scenes(study_area, discover_config)

    assert len(results) == 2
    assert results[0].scene_id == SCENE_A
    assert results[0].cloud_cover == 3.0
    assert results[1].scene_id == SCENE_B


@patch("wildfire_geo_ml.ingest.discover_scenes.Client.open")
def test_discover_raises_when_all_above_cloud_threshold(
    mock_open: MagicMock,
    study_area: StudyAreaConfig,
    discover_config: DiscoverConfig,
) -> None:
    mock_open.return_value = _mock_stac_search([_item(SCENE_CLOUDY, 15.0)])

    with pytest.raises(ValueError, match="cloud_cover < 10"):
        discover_scenes(study_area, discover_config)


@patch("wildfire_geo_ml.ingest.discover_scenes.Client.open")
def test_discover_normalizes_stac_sr_suffix(
    mock_open: MagicMock,
    study_area: StudyAreaConfig,
    discover_config: DiscoverConfig,
) -> None:
    mock_open.return_value = _mock_stac_search([_item(SCENE_A_STAC, 4.0)])

    results = discover_scenes(study_area, discover_config)

    assert len(results) == 1
    assert results[0].scene_id == SCENE_A
    assert results[0].cloud_cover == 4.0


def test_normalize_scene_id_strips_sr_for_s3_keys() -> None:
    stac_id = "LC09_L2SP_044031_20240812_20240813_02_T1_SR"
    base_id = normalize_scene_id(stac_id)
    assert base_id == "LC09_L2SP_044031_20240812_20240813_02_T1"

    key = sr_band_key(base_id, "B4")
    assert key.endswith(f"{base_id}/{base_id}_SR_B4.TIF")
    assert "_SR_SR_" not in key


@patch("wildfire_geo_ml.ingest.discover_scenes.Client.open")
def test_discover_respects_max_cloud_cover_override(
    mock_open: MagicMock,
    study_area: StudyAreaConfig,
    discover_config: DiscoverConfig,
) -> None:
    mock_open.return_value = _mock_stac_search([_item(SCENE_A, 4.0)])

    discover_scenes(study_area, discover_config, max_cloud_cover=5.0)

    call_kwargs = mock_open.return_value.search.call_args.kwargs
    assert call_kwargs["query"]["eo:cloud_cover"] == {"lt": 5.0}


def test_resolve_scene_list_use_config_scenes(pipeline_config: PipelineConfig) -> None:
    results = resolve_scene_list(pipeline_config, use_config_scenes=True)

    assert len(results) == 3
    assert results[0].scene_id == SCENE_A


@patch("wildfire_geo_ml.ingest.discover_scenes.discover_scenes")
def test_resolve_scene_list_discover_default(
    mock_discover: MagicMock,
    pipeline_config: PipelineConfig,
) -> None:
    from wildfire_geo_ml.ingest.discover_scenes import DiscoveredScene

    mock_discover.return_value = [
        DiscoveredScene(scene_id=SCENE_A, cloud_cover=2.0),
        DiscoveredScene(scene_id=SCENE_B, cloud_cover=4.0),
        DiscoveredScene(scene_id=SCENE_CLOUDY, cloud_cover=6.0),
    ]

    results = resolve_scene_list(
        pipeline_config,
        use_config_scenes=False,
        rows=["032"],
    )

    assert [scene.scene_id for scene in results] == [SCENE_A, SCENE_CLOUDY]
    mock_discover.assert_called_once()
