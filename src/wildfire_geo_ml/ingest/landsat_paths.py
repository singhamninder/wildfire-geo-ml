"""Pure helpers for Landsat-9 scene IDs, S3 keys, and local paths."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_COLLECTION_PREFIX = "collection02/level-2/standard/oli-tirs"


@dataclass(frozen=True)
class SceneMeta:
    """Parsed metadata from a Landsat Collection 2 scene ID."""

    scene_id: str
    wrs_path: str
    wrs_row: str
    year: int
    acquisition_date: str


def parse_scene_id(scene_id: str) -> SceneMeta:
    """
    Parse WRS path/row, year, and acquisition date from a Landsat scene ID.

    Parameters
    ----------
    scene_id : str
        E.g. ``LC09_L2SP_044032_20240715_20240717_02_T1``.

    Returns
    -------
    SceneMeta
        Parsed path, row, year, and acquisition date (YYYYMMDD).

    Raises
    ------
    ValueError
        If the scene ID does not match the expected Landsat C2 format.
    """
    parts = scene_id.split("_")
    if len(parts) < 4:
        msg = (
            f"Invalid scene ID {scene_id!r}: expected format "
            "LC09_L2SP_PPPRRR_YYYYMMDD_YYYYMMDD_CC_TX"
        )
        raise ValueError(msg)

    wrs_token = parts[2]
    if len(wrs_token) != 6 or not wrs_token.isdigit():
        msg = f"Invalid WRS token {wrs_token!r} in scene ID {scene_id!r}"
        raise ValueError(msg)

    acquisition_date = parts[3]
    if len(acquisition_date) != 8 or not acquisition_date.isdigit():
        msg = f"Invalid acquisition date {acquisition_date!r} in scene ID {scene_id!r}"
        raise ValueError(msg)

    return SceneMeta(
        scene_id=scene_id,
        wrs_path=wrs_token[:3],
        wrs_row=wrs_token[3:],
        year=int(acquisition_date[:4]),
        acquisition_date=acquisition_date,
    )


def normalize_scene_id(scene_id: str) -> str:
    """
    Convert a LandsatLook STAC item ID to the USGS/AWS scene ID.

    The ``landsat-c2l2-sr`` STAC collection appends ``_SR`` to item IDs; S3 object
    keys under ``s3://usgs-landsat`` use the base scene ID without that suffix.

    Parameters
    ----------
    scene_id : str
        STAC item ID or canonical scene ID.

    Returns
    -------
    str
        Scene ID suitable for S3 key construction.
    """
    if scene_id.endswith("_SR"):
        return scene_id[:-3]
    return scene_id


def scene_prefix(
    scene_id: str,
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX,
) -> str:
    """
    Build the S3 key prefix for a scene (directory under the bucket).

    Parameters
    ----------
    scene_id : str
        Landsat scene ID.
    collection_prefix : str
        Collection path under the bucket root.

    Returns
    -------
    str
        Prefix ending with ``{scene_id}/``.
    """
    meta = parse_scene_id(scene_id)
    return f"{collection_prefix.rstrip('/')}/{meta.year}/{meta.wrs_path}/{meta.wrs_row}/{scene_id}/"


def sr_band_key(
    scene_id: str,
    band: str,
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX,
) -> str:
    """
    Build the S3 object key for a surface-reflectance band GeoTIFF.

    Parameters
    ----------
    scene_id : str
        Landsat scene ID.
    band : str
        Band name, e.g. ``B4``.
    collection_prefix : str
        Collection path under the bucket root.

    Returns
    -------
    str
        Full S3 key relative to bucket root.
    """
    band_upper = band.upper().removeprefix("B")
    if not band_upper.isdigit():
        msg = f"Invalid band {band!r}; expected B2–B7"
        raise ValueError(msg)
    band_label = f"B{band_upper}"
    prefix = scene_prefix(scene_id, collection_prefix)
    return f"{prefix}{scene_id}_SR_{band_label}.TIF"


def mtl_key(
    scene_id: str,
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX,
) -> str:
    """
    Build the S3 object key for the scene MTL JSON metadata file.

    Parameters
    ----------
    scene_id : str
        Landsat scene ID.
    collection_prefix : str
        Collection path under the bucket root.

    Returns
    -------
    str
        Full S3 key relative to bucket root.
    """
    prefix = scene_prefix(scene_id, collection_prefix)
    return f"{prefix}{scene_id}_MTL.json"


def local_scene_dir(output_dir: Path, scene_id: str) -> Path:
    """Return the per-scene directory under the download root."""
    return output_dir / scene_id


def local_band_path(output_dir: Path, scene_id: str, band: str) -> Path:
    """
    Return the local path for a surface-reflectance band file.

    Preserves the USGS filename under ``output_dir/{scene_id}/``.
    """
    band_upper = band.upper().removeprefix("B")
    band_label = f"B{band_upper}"
    filename = f"{scene_id}_SR_{band_label}.TIF"
    return local_scene_dir(output_dir, scene_id) / filename


def local_mtl_path(output_dir: Path, scene_id: str) -> Path:
    """Return the local path for the scene MTL JSON file."""
    return local_scene_dir(output_dir, scene_id) / f"{scene_id}_MTL.json"


def discover_scenes_on_disk(input_dir: Path) -> list[str]:
    """
    List scene IDs with SR band GeoTIFFs under ``input_dir``.

    Parameters
    ----------
    input_dir : Path
        Root data directory (e.g. ``data/raw`` or ``data/cog``).

    Returns
    -------
    list[str]
        Scene directory names containing at least one ``*_SR_B*.TIF`` file.
    """
    if not input_dir.is_dir():
        return []

    scenes: list[str] = []
    for child in sorted(input_dir.iterdir()):
        if child.is_dir() and any(child.glob("*_SR_B*.TIF")):
            scenes.append(child.name)
    return scenes


def filter_scenes(
    scenes: list[str],
    wrs_path: str | None = None,
    rows: list[str] | None = None,
    dates: list[str] | None = None,
) -> list[str]:
    """
    Filter scene IDs by WRS path, row, and/or acquisition date.

    Parameters
    ----------
    scenes : list[str]
        Scene IDs to filter.
    wrs_path : str, optional
        WRS-2 path to match (e.g. ``044``).
    rows : list[str], optional
        WRS-2 rows to match (e.g. ``["032", "033"]``).
    dates : list[str], optional
        Acquisition dates as YYYYMMDD.

    Returns
    -------
    list[str]
        Scene IDs passing all supplied filters.
    """
    filtered: list[str] = []
    for scene_id in scenes:
        meta = parse_scene_id(scene_id)
        if wrs_path is not None and meta.wrs_path != wrs_path:
            continue
        if rows is not None and meta.wrs_row not in rows:
            continue
        if dates is not None and meta.acquisition_date not in dates:
            continue
        filtered.append(scene_id)
    return filtered
