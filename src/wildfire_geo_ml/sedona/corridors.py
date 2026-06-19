"""
Fetch and cache California Energy Commission transmission line geometries.

Lines are buffered in a metric CRS for corridor zonal statistics in Sedona.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _bbox_to_arcgis_envelope(bbox: tuple[float, float, float, float]) -> str:
    """
    Convert a WGS84 bbox to an ArcGIS envelope geometry string.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box as (west, south, east, north).

    Returns
    -------
    str
        Comma-separated ``xmin,ymin,xmax,ymax`` for ArcGIS REST.
    """
    west, south, east, north = bbox
    return f"{west},{south},{east},{north}"


def build_arcgis_query_url(
    rest_service_url: str,
    bbox: tuple[float, float, float, float],
    *,
    out_fields: str = "*",
    result_record_count: int = 2000,
) -> str:
    """
    Build an ArcGIS FeatureServer query URL for a bbox-filtered GeoJSON response.

    Parameters
    ----------
    rest_service_url : str
        Base query endpoint for the feature service.
    bbox : tuple[float, float, float, float]
        AOI bounding box in EPSG:4326.
    out_fields : str
        Fields to return from the service.
    result_record_count : int
        Maximum records returned by the query.

    Returns
    -------
    str
        Fully qualified query URL.
    """
    params = {
        "where": "1=1",
        "geometry": _bbox_to_arcgis_envelope(bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "f": "geojson",
        "resultRecordCount": str(result_record_count),
    }
    query = urllib.parse.urlencode(params)
    separator = "&" if "?" in rest_service_url else "?"
    return f"{rest_service_url}{separator}{query}"


def _validate_geojson(payload: dict[str, Any]) -> None:
    """
    Validate a minimal GeoJSON FeatureCollection payload.

    Parameters
    ----------
    payload : dict
        Parsed GeoJSON response.

    Raises
    ------
    ValueError
        If the payload is empty or not a FeatureCollection with features.
    """
    if payload.get("type") != "FeatureCollection":
        msg = "ArcGIS response is not a GeoJSON FeatureCollection"
        raise ValueError(msg)
    features = payload.get("features")
    if not isinstance(features, list) or len(features) == 0:
        msg = "ArcGIS query returned zero transmission line features for the AOI"
        raise ValueError(msg)


def fetch_transmission_lines(
    bbox: tuple[float, float, float, float],
    out_path: Path,
    *,
    rest_service_url: str,
    force_refresh: bool = False,
) -> Path:
    """
    Fetch AOI-filtered CEC transmission lines and cache them as GeoJSON.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Study-area bounding box in EPSG:4326.
    out_path : Path
        Local cache path for the GeoJSON file.
    rest_service_url : str
        ArcGIS REST query endpoint.
    force_refresh : bool
        When True, re-download even if ``out_path`` already exists.

    Returns
    -------
    Path
        Path to the cached GeoJSON file.

    Raises
    ------
    ValueError
        If the HTTP response is not 200 or contains no features.
    urllib.error.URLError
        If the network request fails.
    """
    if out_path.is_file() and not force_refresh:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = build_arcgis_query_url(rest_service_url, bbox)
    request = urllib.request.Request(url, headers={"User-Agent": "wildfire-geo-ml/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        status = getattr(response, "status", response.getcode())
        if status != 200:
            msg = f"ArcGIS query failed with HTTP {status}"
            raise ValueError(msg)
        raw = response.read()

    payload = json.loads(raw.decode("utf-8"))
    _validate_geojson(payload)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return out_path
