"""
Build a STAC Catalog from Landsat-9 COG scenes.

WHY: pystac provides a Python-native way to construct STAC catalogs programmatically.
The resulting static JSON catalog can be hosted on S3 without a server — analysts
query it with pystac or pystac_client just like a dynamic STAC API.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import pystac
import rasterio
import typer
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.view import ViewExtension
from rasterio.warp import transform_bounds

from wildfire_geo_ml.ingest.config import IngestConfig, load_pipeline_config
from wildfire_geo_ml.ingest.landsat_paths import (
    discover_scenes_on_disk,
    filter_scenes,
    local_band_path,
    local_mtl_path,
    mtl_key,
    sr_band_key,
)
from wildfire_geo_ml.stac_builder.mtl_parser import parse_mtl_file

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config/pipeline.yaml")
DEFAULT_COG_DIR = Path("data/cog")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("stac")

CATALOG_ID = "wildfire-vegetation-risk"
COLLECTION_ID = "landsat-9-northern-ca-2024"
MEDIA_TYPE_COG = "image/tiff; application=geotiff; profile=cloud-optimized"

LANDSAT9_BANDS: dict[str, Band] = {
    "B2": Band.create(
        name="blue",
        common_name="blue",
        center_wavelength=0.482,
        full_width_half_max=0.060,
    ),
    "B3": Band.create(
        name="green",
        common_name="green",
        center_wavelength=0.562,
        full_width_half_max=0.057,
    ),
    "B4": Band.create(
        name="red",
        common_name="red",
        center_wavelength=0.655,
        full_width_half_max=0.038,
    ),
    "B5": Band.create(
        name="nir08",
        common_name="nir08",
        center_wavelength=0.865,
        full_width_half_max=0.028,
    ),
    "B6": Band.create(
        name="swir16",
        common_name="swir16",
        center_wavelength=1.610,
        full_width_half_max=0.085,
    ),
    "B7": Band.create(
        name="swir22",
        common_name="swir22",
        center_wavelength=2.200,
        full_width_half_max=0.200,
    ),
}

REQUIRED_BANDS = {"B4", "B5"}


def _parse_scene_datetime(scene_id: str) -> datetime:
    """
    Extract acquisition datetime from Landsat scene ID (noon UTC fallback).

    Parameters
    ----------
    scene_id : str
        E.g. ``LC09_L2SP_044032_20240715_20240717_02_T1``.

    Returns
    -------
    datetime
        UTC datetime of acquisition (date from scene ID; time at noon UTC).
    """
    date_str = scene_id.split("_")[3]
    return datetime(
        int(date_str[:4]),
        int(date_str[4:6]),
        int(date_str[6:8]),
        12,
        0,
        0,
        tzinfo=UTC,
    )


def _get_scene_bbox_and_footprint(cog_path: Path) -> tuple[list[float], dict]:
    """
    Read spatial extent and footprint polygon from a COG via rasterio.

    Parameters
    ----------
    cog_path : Path
        Path to any single-band COG from the scene.

    Returns
    -------
    tuple[list[float], dict]
        Bbox as ``[west, south, east, north]`` in EPSG:4326 and GeoJSON Polygon.
    """
    with rasterio.open(cog_path) as src:
        bounds = src.bounds
        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )
    bbox = [west, south, east, north]
    footprint = {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }
    return bbox, footprint


def _s3_href(bucket: str, key: str) -> str:
    """Build an ``s3://`` URI for a bucket object key."""
    return f"s3://{bucket}/{key}"


def collect_cog_paths(cog_dir: Path, scene_id: str, bands: list[str]) -> dict[str, Path]:
    """
    Collect existing COG band paths for a scene.

    Parameters
    ----------
    cog_dir : Path
        COG root directory.
    scene_id : str
        Landsat scene ID.
    bands : list[str]
        Band names to collect.

    Returns
    -------
    dict[str, Path]
        Mapping of band label to local COG path for files that exist.
    """
    paths: dict[str, Path] = {}
    for band in bands:
        band_path = local_band_path(cog_dir, scene_id, band)
        if band_path.is_file():
            paths[band] = band_path
    return paths


def create_item_from_cog(
    scene_id: str,
    cog_paths: dict[str, Path],
    metadata: dict,
    ingest: IngestConfig,
    *,
    local_hrefs: bool = False,
    mtl_path: Path | None = None,
) -> pystac.Item:
    """
    Build a fully-validated STAC Item for a Landsat-9 scene.

    Parameters
    ----------
    scene_id : str
        Landsat scene ID.
    cog_paths : dict[str, Path]
        Mapping of band name to local COG file path.
    metadata : dict
        Scene metadata from MTL JSON (cloud cover, view, projection, datetime).
    ingest : IngestConfig
        Bucket and collection prefix for S3 asset hrefs.
    local_hrefs : bool
        If True, use local COG paths instead of ``s3://usgs-landsat`` URIs.
    mtl_path : Path, optional
        Local MTL JSON path (required when ``local_hrefs`` is True).

    Returns
    -------
    pystac.Item
        Validated STAC Item with EO, Projection, and View extensions.

    Raises
    ------
    ValueError
        If required bands (B4, B5) are missing from ``cog_paths``.
    """
    missing = REQUIRED_BANDS - set(cog_paths.keys())
    if missing:
        msg = f"Missing required bands for NDVI in scene {scene_id}: {missing}"
        raise ValueError(msg)

    ref_band_path = cog_paths["B4"]
    bbox, footprint = _get_scene_bbox_and_footprint(ref_band_path)
    dt = metadata.get("datetime", _parse_scene_datetime(scene_id))

    item = pystac.Item(
        id=scene_id,
        geometry=footprint,
        bbox=bbox,
        datetime=dt,
        properties={
            "platform": "landsat-9",
            "instruments": ["OLI", "TIRS"],
            "constellation": "landsat",
            "landsat:wrs_path": scene_id.split("_")[2][:3],
            "landsat:wrs_row": scene_id.split("_")[2][3:],
            "landsat:processing_level": "L2SP",
            "landsat:collection_number": "02",
            "landsat:collection_category": "T1",
        },
    )

    eo_ext = EOExtension.ext(item, add_if_missing=True)
    eo_ext.apply(
        bands=[LANDSAT9_BANDS[b] for b in sorted(cog_paths.keys()) if b in LANDSAT9_BANDS],
        cloud_cover=float(metadata.get("eo:cloud_cover", 0.0)),
    )

    proj_ext = ProjectionExtension.ext(item, add_if_missing=True)
    proj_ext.apply(
        epsg=int(metadata.get("proj:epsg", 32610)),
        shape=metadata.get("proj:shape", [8031, 8001]),
        transform=metadata.get("proj:transform"),
    )

    view_ext = ViewExtension.ext(item, add_if_missing=True)
    view_ext.apply(
        sun_elevation=float(metadata.get("view:sun_elevation", 0.0)),
        sun_azimuth=float(metadata.get("view:sun_azimuth", 0.0)),
    )

    for band_name, cog_path in sorted(cog_paths.items()):
        if local_hrefs:
            href = cog_path.as_posix()
        else:
            key = sr_band_key(scene_id, band_name, ingest.collection_prefix)
            href = _s3_href(ingest.bucket, key)
        band_obj = LANDSAT9_BANDS.get(band_name)
        asset = pystac.Asset(
            href=href,
            media_type=MEDIA_TYPE_COG,
            title=f"Band {band_name}",
            roles=["data"],
            extra_fields={"eo:bands": [band_obj.to_dict()] if band_obj else []},
        )
        item.add_asset(band_name, asset)

    if local_hrefs:
        if mtl_path is None:
            msg = "mtl_path is required when local_hrefs=True"
            raise ValueError(msg)
        mtl_href = mtl_path.as_posix()
    else:
        mtl_href = _s3_href(ingest.bucket, mtl_key(scene_id, ingest.collection_prefix))
    item.add_asset(
        "MTL_JSON",
        pystac.Asset(
            href=mtl_href,
            media_type="application/json",
            title="MTL JSON Metadata",
            roles=["metadata"],
        ),
    )

    item.validate()
    return item


def create_collection(items: list[pystac.Item]) -> pystac.Collection:
    """
    Build a STAC Collection from validated Items with computed extents.

    Parameters
    ----------
    items : list[pystac.Item]
        STAC Items to include.

    Returns
    -------
    pystac.Collection
        Collection with union spatial/temporal extents and EO band summaries.
    """
    all_bboxes = [item.bbox for item in items if item.bbox]
    west = min(b[0] for b in all_bboxes)
    south = min(b[1] for b in all_bboxes)
    east = max(b[2] for b in all_bboxes)
    north = max(b[3] for b in all_bboxes)

    datetimes = [item.datetime for item in items if item.datetime]
    temporal_extent = pystac.TemporalExtent(intervals=[[min(datetimes), max(datetimes)]])

    extent = pystac.Extent(
        spatial=pystac.SpatialExtent(bboxes=[[west, south, east, north]]),
        temporal=temporal_extent,
    )

    collection = pystac.Collection(
        id=COLLECTION_ID,
        title="Landsat-9 Northern California Summer 2024",
        description=(
            "Landsat-9 Collection 2 Level-2 Surface Reflectance over HFTD Tier 3 "
            "territory (Butte, Plumas, Shasta counties). Summer 2024 peak fire season. "
            "Built for vegetation encroachment risk analysis on transmission corridors."
        ),
        extent=extent,
        license="proprietary",
        providers=[
            pystac.Provider(
                name="USGS",
                roles=[pystac.ProviderRole.PRODUCER, pystac.ProviderRole.LICENSOR],
                url="https://earthexplorer.usgs.gov",
            ),
            pystac.Provider(
                name="AWS Open Data",
                roles=[pystac.ProviderRole.HOST],
                url="https://registry.opendata.aws/usgs-landsat/",
            ),
        ],
    )
    collection.summaries = pystac.Summaries(
        {
            "platform": ["landsat-9"],
            "instruments": ["OLI", "TIRS"],
            "eo:bands": [band.to_dict() for band in LANDSAT9_BANDS.values()],
        }
    )

    for item in items:
        collection.add_item(item)

    return collection


def build_and_save_catalog(
    collection: pystac.Collection,
    output_dir: Path,
    catalog_type: pystac.CatalogType = pystac.CatalogType.SELF_CONTAINED,
) -> pystac.Catalog:
    """
    Assemble root Catalog, normalize hrefs, validate, and write to disk.

    Parameters
    ----------
    collection : pystac.Collection
        Populated collection to attach to the root catalog.
    output_dir : Path
        Directory where ``catalog.json`` and subdirectories will be written.
    catalog_type : pystac.CatalogType
        ``SELF_CONTAINED`` (default) for portable relative hrefs.

    Returns
    -------
    pystac.Catalog
        Validated, written catalog object.
    """
    catalog = pystac.Catalog(
        id=CATALOG_ID,
        description=(
            "Landsat-9 imagery catalog for vegetation encroachment risk assessment "
            "on HFTD power line corridors, Northern California. "
            "Portfolio wildfire geospatial ML pipeline."
        ),
        title="Wildfire Vegetation Risk — Landsat-9 Northern CA",
    )
    catalog.add_child(collection)
    catalog.normalize_hrefs(str(output_dir))
    catalog.validate_all()
    catalog.save(catalog_type=catalog_type, dest_href=str(output_dir))
    return catalog


def build_catalog_from_dirs(
    cog_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    ingest: IngestConfig,
    scene_ids: list[str],
    *,
    local_hrefs: bool = False,
) -> pystac.Catalog:
    """
    Build and save a STAC catalog from on-disk COGs and MTL metadata.

    Parameters
    ----------
    cog_dir : Path
        Validated COG root.
    raw_dir : Path
        Raw download root containing MTL JSON files.
    output_dir : Path
        STAC catalog output directory.
    ingest : IngestConfig
        Band list and S3 href settings.
    scene_ids : list[str]
        Scene IDs to include.
    local_hrefs : bool
        Use local file paths for assets when True.

    Returns
    -------
    pystac.Catalog
        Written catalog object.
    """
    items: list[pystac.Item] = []
    for scene_id in scene_ids:
        mtl_path = local_mtl_path(raw_dir, scene_id)
        metadata = parse_mtl_file(mtl_path)
        cog_paths = collect_cog_paths(cog_dir, scene_id, ingest.bands)
        logger.info("Building STAC item for %s (%d bands)", scene_id, len(cog_paths))
        items.append(
            create_item_from_cog(
                scene_id,
                cog_paths,
                metadata,
                ingest,
                local_hrefs=local_hrefs,
                mtl_path=mtl_path,
            )
        )

    collection = create_collection(items)
    return build_and_save_catalog(collection, output_dir)


def resolve_catalog_scenes(
    cog_dir: Path,
    wrs_path: str | None = None,
    rows: list[str] | None = None,
    dates: list[str] | None = None,
) -> list[str]:
    """
    List scene IDs with COGs on disk, optionally filtered.

    Parameters
    ----------
    cog_dir : Path
        COG root directory.
    wrs_path : str, optional
        WRS-2 path filter.
    rows : list[str], optional
        WRS-2 row filter.
    dates : list[str], optional
        Acquisition date filter (YYYYMMDD).

    Returns
    -------
    list[str]
        Scene IDs to catalog.
    """
    return filter_scenes(
        discover_scenes_on_disk(cog_dir),
        wrs_path=wrs_path,
        rows=rows,
        dates=dates,
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        exists=True,
        dir_okay=False,
        help="Path to pipeline.yaml",
    ),
    cog_dir: Path = typer.Option(
        DEFAULT_COG_DIR,
        "--cog-dir",
        file_okay=False,
        help="Root directory for validated COG outputs",
    ),
    raw_dir: Path = typer.Option(
        DEFAULT_RAW_DIR,
        "--raw-dir",
        file_okay=False,
        help="Root directory for MTL JSON metadata",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        file_okay=False,
        help="Directory for STAC catalog JSON output",
    ),
    wrs_path: str | None = typer.Option(
        None,
        "--path",
        help="Filter by WRS-2 path (e.g. 044); defaults to config wrs_path",
    ),
    rows: list[str] = typer.Option(
        [],
        "--rows",
        help="Filter by WRS-2 row(s), e.g. --rows 032 --rows 033",
    ),
    dates: list[str] = typer.Option(
        [],
        "--dates",
        help="Filter by acquisition date YYYYMMDD",
    ),
    local_hrefs: bool = typer.Option(
        False,
        "--local-hrefs",
        help="Use local COG paths for asset hrefs instead of s3://usgs-landsat",
    ),
) -> None:
    """
    Build a self-contained STAC catalog from Landsat-9 COG scenes.

    Scans ``data/cog/`` for on-disk scenes, reads MTL metadata from ``data/raw/``,
    and writes a validated catalog to ``stac/``. Band asset hrefs default to
    ``s3://usgs-landsat`` URIs for cloud-native consumption.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = load_pipeline_config(config_path)
    ingest = pipeline.ingest

    path_filter = wrs_path if wrs_path is not None else ingest.wrs_path
    row_filter = list(rows) if rows else None
    date_filter = list(dates) if dates else None

    scene_ids = resolve_catalog_scenes(
        cog_dir,
        wrs_path=path_filter,
        rows=row_filter,
        dates=date_filter,
    )

    if not scene_ids:
        typer.echo(
            f"No scenes with COG bands found under {cog_dir} matching filters.",
            err=True,
        )
        raise typer.Exit(1)

    catalog = build_catalog_from_dirs(
        cog_dir,
        raw_dir,
        output_dir,
        ingest,
        scene_ids,
        local_hrefs=local_hrefs,
    )

    item_count = len(list(catalog.get_items(recursive=True)))
    typer.echo(f"\nSTAC catalog written to {output_dir / 'catalog.json'}")
    typer.echo(f"  Items: {item_count}")
    typer.echo(f"  Asset hrefs: {'local' if local_hrefs else 's3://' + ingest.bucket}")


if __name__ == "__main__":
    app()
