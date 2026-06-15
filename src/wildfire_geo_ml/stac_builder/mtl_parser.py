"""Parse Landsat Collection 2 MTL JSON into STAC-friendly metadata."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast


def _as_float(value: object) -> float | None:
    """Convert MTL string/number values to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_int(value: object) -> int | None:
    """Convert MTL string/number values to int."""
    if value is None:
        return None
    float_value = _as_float(value)
    if float_value is None:
        return None
    return int(float_value)


def _parse_scene_center_datetime(
    date_acquired: str | None, scene_center_time: str | None
) -> datetime | None:
    """
    Combine DATE_ACQUIRED and SCENE_CENTER_TIME into a timezone-aware datetime.

    Parameters
    ----------
    date_acquired : str, optional
        ISO date string, e.g. ``2024-07-15``.
    scene_center_time : str, optional
        Time string, e.g. ``18:45:23.1234560Z``.

    Returns
    -------
    datetime | None
        UTC datetime when both parts are present and parseable.
    """
    if not date_acquired or not scene_center_time:
        return None

    time_part = scene_center_time.strip()
    if time_part.endswith("Z"):
        time_part = time_part[:-1]
    if "." in time_part:
        time_part = time_part.split(".", maxsplit=1)[0]

    try:
        return datetime.strptime(
            f"{date_acquired}T{time_part}",
            "%Y-%m-%dT%H:%M:%S",
        ).replace(tzinfo=UTC)
    except ValueError:
        return None


def _build_proj_transform(projection: dict[str, object]) -> list[float] | None:
    """
    Build a 6-element GDAL affine from Landsat C2 projection attributes.

    Parameters
    ----------
    projection : dict
        ``PROJECTION_ATTRIBUTES`` section from MTL JSON.

    Returns
    -------
    list[float] | None
        ``[x_scale, 0, x_origin, 0, y_scale, y_origin]`` or None if incomplete.
    """
    cell_size = _as_float(projection.get("GRID_CELL_SIZE_REFLECTIVE"))
    ulx = _as_float(projection.get("CORNER_UL_PROJECTION_X_PRODUCT"))
    uly = _as_float(projection.get("CORNER_UL_PROJECTION_Y_PRODUCT"))
    if cell_size is None or ulx is None or uly is None:
        return None
    return [cell_size, 0.0, ulx, 0.0, -cell_size, uly]


def parse_mtl_dict(mtl: dict[str, object]) -> dict[str, object]:
    """
    Parse Landsat C2 MTL JSON content into flat STAC metadata fields.

    Parameters
    ----------
    mtl : dict
        Parsed MTL JSON (top-level or ``LANDSAT_METADATA_FILE`` wrapper).

    Returns
    -------
    dict
        Keys include ``eo:cloud_cover``, ``view:sun_elevation``,
        ``view:sun_azimuth``, ``proj:shape``, ``proj:transform``,
        ``proj:epsg``, and ``datetime`` when available.
    """
    root_obj = mtl.get("LANDSAT_METADATA_FILE", mtl)
    if not isinstance(root_obj, dict):
        msg = "MTL JSON must contain a mapping at LANDSAT_METADATA_FILE"
        raise ValueError(msg)
    root: dict[str, object] = cast(dict[str, object], root_obj)
    image_attrs_obj = root.get("IMAGE_ATTRIBUTES")
    projection_attrs_obj = root.get("PROJECTION_ATTRIBUTES")
    image_attrs: dict[str, object] = (
        cast(dict[str, object], image_attrs_obj) if isinstance(image_attrs_obj, dict) else {}
    )
    projection_attrs: dict[str, object] = (
        cast(dict[str, object], projection_attrs_obj)
        if isinstance(projection_attrs_obj, dict)
        else {}
    )

    cloud_land = image_attrs.get("CLOUD_COVER_LAND")
    cloud_cover_raw = cloud_land if cloud_land is not None else image_attrs.get("CLOUD_COVER")
    cloud_cover = _as_float(cloud_cover_raw)
    sun_elevation = _as_float(image_attrs.get("SUN_ELEVATION_LAND"))
    sun_azimuth = _as_float(image_attrs.get("SUN_AZIMUTH_LAND"))

    lines = _as_int(projection_attrs.get("REFLECTIVE_LINES"))
    samples = _as_int(projection_attrs.get("REFLECTIVE_SAMPLES"))
    proj_shape = [lines, samples] if lines is not None and samples is not None else None

    utm_zone = _as_int(projection_attrs.get("UTM_ZONE"))
    proj_epsg = 32600 + utm_zone if utm_zone is not None else 32610

    metadata: dict[str, object] = {
        "eo:cloud_cover": cloud_cover if cloud_cover is not None else 0.0,
        "view:sun_elevation": sun_elevation if sun_elevation is not None else 0.0,
        "view:sun_azimuth": sun_azimuth if sun_azimuth is not None else 0.0,
        "proj:epsg": proj_epsg,
    }

    if proj_shape is not None:
        metadata["proj:shape"] = proj_shape

    proj_transform = _build_proj_transform(projection_attrs)
    if proj_transform is not None:
        metadata["proj:transform"] = proj_transform

    date_acquired = image_attrs.get("DATE_ACQUIRED")
    scene_center_time = image_attrs.get("SCENE_CENTER_TIME")
    scene_dt = _parse_scene_center_datetime(
        date_acquired if isinstance(date_acquired, str) else None,
        scene_center_time if isinstance(scene_center_time, str) else None,
    )
    if scene_dt is not None:
        metadata["datetime"] = scene_dt

    return metadata


def parse_mtl_file(mtl_path: Path) -> dict[str, object]:
    """
    Load and parse a Landsat MTL JSON file.

    Parameters
    ----------
    mtl_path : Path
        Path to ``{scene_id}_MTL.json``.

    Returns
    -------
    dict
        Flat STAC metadata fields from :func:`parse_mtl_dict`.

    Raises
    ------
    FileNotFoundError
        If the MTL file does not exist.
    ValueError
        If the file is not valid JSON or has unexpected structure.
    """
    if not mtl_path.is_file():
        msg = f"MTL file not found: {mtl_path}"
        raise FileNotFoundError(msg)

    with mtl_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        msg = f"MTL file {mtl_path} must contain a JSON object"
        raise ValueError(msg)

    return parse_mtl_dict(cast(dict[str, object], raw))
