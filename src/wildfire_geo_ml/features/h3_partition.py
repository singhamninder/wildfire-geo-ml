"""
H3-partitioned feature engineering CLI and orchestration.

Computes NDVI, NBR, and NDWI zonal statistics per H3 cell for each Landsat-9
scene and writes Hive-partitioned GeoParquet for downstream Sedona and ML joins.
"""

import logging
import shutil
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import typer
from affine import Affine
from rasterio import features as rio_features
from rasterio.warp import transform_bounds
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from wildfire_geo_ml.features.geopandas_io import empty_gdf
from wildfire_geo_ml.features.h3_utils import (
    coerce_bbox4,
    filter_cells_to_extent,
    filter_cells_to_geometry,
    polyfill_bbox,
)
from wildfire_geo_ml.features.indices import LANDSAT9_FILL, load_scene_indices
from wildfire_geo_ml.features.zonal_stats import compute_scene_h3_stats
from wildfire_geo_ml.ingest.config import FeaturesConfig, PipelineConfig, load_pipeline_config
from wildfire_geo_ml.ingest.landsat_paths import (
    discover_scenes_on_disk,
    filter_scenes,
    local_band_path,
    parse_scene_id,
)
from wildfire_geo_ml.stac_builder.build_catalog import collect_cog_paths

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config/pipeline.yaml")
DEFAULT_COG_DIR = Path("data/cog")
# Decimate footprint raster to ~120 m before vectorizing (balance speed vs fidelity).
FOOTPRINT_TARGET_RES_M = 120.0


def get_scene_valid_footprint(
    cog_path: Path,
) -> tuple[BaseGeometry, str] | None:
    """
    Derive a valid-data footprint polygon from a Landsat SR band COG.

    Reads a decimated band, vectorizes pixels where DN is not fill (``0``),
    and returns a simplified/buffered polygon in the raster CRS.

    Parameters
    ----------
    cog_path : Path
        Path to a single-band Landsat SR COG (typically B4).

    Returns
    -------
    tuple[BaseGeometry, str] or None
        Footprint geometry and its CRS string, or ``None`` when no valid pixels.
    """
    with rasterio.open(cog_path) as src:
        if src.crs is None:
            msg = f"COG missing CRS: {cog_path}"
            raise ValueError(msg)

        pixel_size = max(abs(src.transform.a), abs(src.transform.e))
        # Decimate full-resolution scene to reduce vectorization cost.
        decimation = max(1, round(FOOTPRINT_TARGET_RES_M / pixel_size))
        out_height = max(1, src.height // decimation)
        out_width = max(1, src.width // decimation)
        data = src.read(1, out_shape=(out_height, out_width))
        scaled_transform = src.transform * Affine.scale(
            src.width / out_width,
            src.height / out_height,
        )
        crs = src.crs.to_string()

    # Mask fill pixels (DN=0) to isolate valid surface-reflectance coverage.
    valid = data != LANDSAT9_FILL
    if not np.any(valid):
        return None

    # Vectorize valid-pixel mask into polygons in the raster CRS.
    polygons = [
        shape(geom)
        for geom, value in rio_features.shapes(
            valid.astype(np.uint8),
            mask=valid,
            transform=scaled_transform,
        )
        if value == 1
    ]
    if not polygons:
        return None

    footprint = unary_union(polygons)
    coarse_pixel = max(abs(scaled_transform.a), abs(scaled_transform.e))
    # Simplify and buffer by one coarse pixel to close small gaps in the footprint.
    footprint = footprint.simplify(coarse_pixel, preserve_topology=True)
    footprint = footprint.buffer(coarse_pixel)
    if footprint.is_empty:
        return None
    return footprint, crs


def get_scene_wgs84_bbox(cog_path: Path) -> tuple[float, float, float, float]:
    """
    Read a scene bounding box in EPSG:4326 from any band COG.

    Parameters
    ----------
    cog_path : Path
        Path to a single-band COG from the scene.

    Returns
    -------
    tuple[float, float, float, float]
        Bounding box as (west, south, east, north).
    """
    with rasterio.open(cog_path) as src:
        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
        )
    return (west, south, east, north)


def collect_required_cog_paths(
    cog_dir: Path,
    scene_id: str,
    required_bands: list[str],
) -> dict[str, Path]:
    """
    Collect COG paths for all bands required by the feature pipeline.

    Parameters
    ----------
    cog_dir : Path
        COG root directory.
    scene_id : str
        Landsat scene ID.
    required_bands : list[str]
        Band labels required for spectral indices.

    Returns
    -------
    dict[str, Path]
        Mapping of band label to local COG path.

    Raises
    ------
    ValueError
        If any required band is missing on disk.
    """
    cog_paths = collect_cog_paths(cog_dir, scene_id, required_bands)
    missing = set(required_bands) - set(cog_paths.keys())
    if missing:
        msg = (
            f"Scene {scene_id} is missing required COG bands: {sorted(missing)}. "
            f"Expected files under {local_band_path(cog_dir, scene_id, 'B4').parent}"
        )
        raise ValueError(msg)
    return cog_paths


def resolve_feature_scenes(
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
        Scene IDs to process.
    """
    return filter_scenes(
        discover_scenes_on_disk(cog_dir),
        wrs_path=wrs_path,
        rows=rows,
        dates=dates,
    )


def process_scene(
    scene_id: str,
    cog_dir: Path,
    study_bbox: tuple[float, float, float, float] | list[float],
    features_config: FeaturesConfig,
) -> gpd.GeoDataFrame:
    """
    Compute H3 zonal statistics for one Landsat scene.

    Parameters
    ----------
    scene_id : str
        Landsat scene ID.
    cog_dir : Path
        COG root directory.
    study_bbox : tuple or list of float
        Study area bbox as (west, south, east, north) in EPSG:4326.
    features_config : FeaturesConfig
        Feature engineering settings.

    Returns
    -------
    gpd.GeoDataFrame
        Per-cell spectral index statistics for the scene.
    """
    cog_paths = collect_required_cog_paths(cog_dir, scene_id, features_config.required_bands)
    index_arrays = load_scene_indices(cog_paths)

    # Polyfill study bbox, then narrow to scene extent via footprint or WGS84 bbox.
    study_cells = polyfill_bbox(coerce_bbox4(study_bbox), features_config.h3_resolution)
    scene_bbox = get_scene_wgs84_bbox(cog_paths["B4"])
    bbox_cells = filter_cells_to_extent(study_cells, scene_bbox)
    footprint_result = get_scene_valid_footprint(cog_paths["B4"])
    if footprint_result is not None:
        footprint_geom, footprint_crs = footprint_result
        # Prefer valid-data footprint over axis-aligned bbox when available.
        scene_cells = filter_cells_to_geometry(study_cells, footprint_geom, footprint_crs)
        logger.info(
            "Scene %s: %d H3 cells after footprint filter (%d after WGS84 bbox)",
            scene_id,
            len(scene_cells),
            len(bbox_cells),
        )
    else:
        logger.warning(
            "Scene %s: no valid-data footprint; falling back to WGS84 bbox filter",
            scene_id,
        )
        scene_cells = bbox_cells

    gdf = compute_scene_h3_stats(
        index_arrays,
        scene_cells,
        stat_names=features_config.stat_names,
    )
    if gdf.empty:
        logger.warning("No H3 cells with valid pixels for scene %s", scene_id)
        return gdf

    # Attach scene metadata columns for downstream Hive/ML joins.
    meta = parse_scene_id(scene_id)
    gdf["scene_id"] = scene_id
    gdf["acquisition_date"] = meta.acquisition_date
    return gdf


def read_partitioned_geoparquet(output_dir: Path) -> gpd.GeoDataFrame:
    """
    Read Hive-partitioned GeoParquet files from ``output_dir``.

    Parameters
    ----------
    output_dir : Path
        Directory containing ``h3_res8=<cell>/part-0.parquet`` files.

    Returns
    -------
    gpd.GeoDataFrame
        Combined feature table with ``h3_res8`` restored from paths.
    """
    parquet_files = sorted(output_dir.glob("h3_res8=*/part-0.parquet"))
    if not parquet_files:
        return empty_gdf()

    frames = [gpd.read_parquet(path) for path in parquet_files]
    combined = cast(gpd.GeoDataFrame, pd.concat(frames, ignore_index=True))
    if "h3_res8" not in combined.columns:
        combined["h3_res8"] = combined["h3_index"]
    return combined


def write_partitioned_geoparquet(gdf: gpd.GeoDataFrame, output_dir: Path) -> Path:
    """
    Write GeoParquet partitioned by ``h3_res8``.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Combined feature rows for one or more scenes.
    output_dir : Path
        Output directory for partitioned parquet files.

    Returns
    -------
    Path
        Output directory path.
    """
    if output_dir.exists():
        # Remove prior partition tree so re-runs produce a clean snapshot.
        shutil.rmtree(output_dir)

    if gdf.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("No feature rows to write; created empty output directory %s", output_dir)
        return output_dir

    for h3_res8, group in gdf.groupby("h3_res8", sort=True):
        part_dir = output_dir / f"h3_res8={h3_res8}"
        part_dir.mkdir(parents=True, exist_ok=True)
        # Hive partition key is encoded in the directory name, not the parquet file.
        group.drop(columns=["h3_res8"]).to_parquet(part_dir / "part-0.parquet", index=False)

    return output_dir


def build_features(
    cog_dir: Path,
    scene_ids: list[str],
    pipeline_config: PipelineConfig,
    *,
    study_bbox: tuple[float, float, float, float] | list[float] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Build H3-partitioned spectral features for all requested scenes.

    Parameters
    ----------
    cog_dir : Path
        COG root directory.
    scene_ids : list[str]
        Scene IDs to process.
    pipeline_config : PipelineConfig
        Validated pipeline configuration.
    study_bbox : tuple or list of float, optional
        Override study area bbox. Defaults to ``pipeline_config.study_area.bbox``.
    output_dir : Path, optional
        Override output directory. Defaults to ``features.output_dir`` from config.

    Returns
    -------
    Path
        Directory containing partitioned GeoParquet output.

    Raises
    ------
    ValueError
        If ``scene_ids`` is empty.
    """
    if not scene_ids:
        msg = f"No scenes to process under {cog_dir}"
        raise ValueError(msg)

    bbox_tuple = coerce_bbox4(
        study_bbox if study_bbox is not None else pipeline_config.study_area.bbox
    )
    features_config = pipeline_config.features
    out_dir = output_dir if output_dir is not None else Path(features_config.output_dir)

    frames: list[gpd.GeoDataFrame] = []
    for scene_id in scene_ids:
        logger.info("Processing scene %s", scene_id)
        scene_gdf = process_scene(scene_id, cog_dir, bbox_tuple, features_config)
        if not scene_gdf.empty:
            frames.append(scene_gdf)

    # Concatenate per-scene frames; fall back to empty schema when all scenes are empty.
    if frames:
        combined = cast(gpd.GeoDataFrame, pd.concat(frames, ignore_index=True))
    else:
        combined = empty_gdf()
    return write_partitioned_geoparquet(combined, out_dir)


def _parse_bbox_option(value: str) -> tuple[float, float, float, float]:
    """Parse ``west,south,east,north`` bbox string from ``--bbox``."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise typer.BadParameter(
            "--bbox requires four comma-separated floats: west,south,east,north"
        )
    try:
        west, south, east, north = (float(part) for part in parts)
    except ValueError as exc:
        raise typer.BadParameter(
            "--bbox requires four comma-separated floats: west,south,east,north"
        ) from exc
    return west, south, east, north


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        exists=True,
        dir_okay=False,
        help="Pipeline YAML config path",
    ),
    cog_dir: Path = typer.Option(
        DEFAULT_COG_DIR,
        "--cog-dir",
        file_okay=False,
        help="Root directory for validated COG outputs",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        file_okay=False,
        help="Output directory for H3-partitioned GeoParquet",
    ),
    resolution: int | None = typer.Option(
        None,
        "--resolution",
        help="H3 resolution override (default from config)",
    ),
    bbox: str | None = typer.Option(
        None,
        "--bbox",
        help="Study bbox override as west,south,east,north",
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
) -> None:
    """
    Compute H3-partitioned NDVI, NBR, and NDWI features from Landsat-9 COGs.

    Scans ``data/cog/`` for on-disk scenes, polyfills the study-area bbox at the
    configured H3 resolution, and writes Hive-partitioned GeoParquet.

    Notes
    -----
    Exits with code 1 when no COG scenes match the configured filters.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = load_pipeline_config(config_path)
    if resolution is not None:
        pipeline.features.h3_resolution = resolution

    path_filter = wrs_path if wrs_path is not None else pipeline.ingest.wrs_path
    row_filter = list(rows) if rows else None
    date_filter = list(dates) if dates else None

    scene_ids = resolve_feature_scenes(
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

    study_bbox = _parse_bbox_option(bbox) if bbox is not None else pipeline.study_area.bbox
    out_path = build_features(
        cog_dir,
        scene_ids,
        pipeline,
        study_bbox=study_bbox,
        output_dir=output_dir,
    )

    if out_path.exists() and list(out_path.glob("h3_res8=*/part-0.parquet")):
        result = read_partitioned_geoparquet(out_path)
        scene_count = result["scene_id"].nunique() if "scene_id" in result.columns else 0
        typer.echo(f"\nH3 features written to {out_path}")
        typer.echo(f"  Rows: {len(result)}")
        typer.echo(f"  Scenes: {scene_count}")
    else:
        typer.echo(f"\nNo feature rows written; output directory: {out_path}")


if __name__ == "__main__":
    app()
