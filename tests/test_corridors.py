"""Tests for CEC transmission line fetch and cache."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wildfire_geo_ml.sedona.corridors import (
    build_arcgis_query_url,
    fetch_transmission_lines,
)


def test_build_arcgis_query_url_contains_bbox() -> None:
    url = build_arcgis_query_url(
        "https://example.com/query",
        (-122.5, 39.0, -120.0, 41.5),
    )
    assert "geometry=-122.5%2C39.0%2C-120.0%2C41.5" in url
    assert "f=geojson" in url


def test_fetch_transmission_lines_uses_cache(tmp_path: Path) -> None:
    out_path = tmp_path / "lines.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": None, "properties": {"OBJECTID": 1}}],
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    result = fetch_transmission_lines(
        (-122.5, 39.0, -120.0, 41.5),
        out_path,
        rest_service_url="https://example.com/query",
        force_refresh=False,
    )
    assert result == out_path


@patch("wildfire_geo_ml.sedona.corridors.urllib.request.urlopen")
def test_fetch_transmission_lines_downloads_geojson(
    mock_urlopen: MagicMock,
    tmp_path: Path,
) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": None, "properties": {"OBJECTID": 1}}],
    }
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.read.return_value = json.dumps(payload).encode("utf-8")
    mock_urlopen.return_value = response

    out_path = tmp_path / "lines.geojson"
    result = fetch_transmission_lines(
        (-122.0, 39.5, -121.5, 40.0),
        out_path,
        rest_service_url="https://example.com/query",
        force_refresh=True,
    )
    assert result.is_file()
    cached = json.loads(result.read_text(encoding="utf-8"))
    assert cached["type"] == "FeatureCollection"
    assert len(cached["features"]) == 1


def test_fetch_transmission_lines_empty_features_raises(tmp_path: Path) -> None:
    with patch("wildfire_geo_ml.sedona.corridors.urllib.request.urlopen") as mock_urlopen:
        payload = {"type": "FeatureCollection", "features": []}
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value = response

        with pytest.raises(ValueError, match="zero transmission line features"):
            fetch_transmission_lines(
                (-122.0, 39.5, -121.5, 40.0),
                tmp_path / "lines.geojson",
                rest_service_url="https://example.com/query",
                force_refresh=True,
            )
