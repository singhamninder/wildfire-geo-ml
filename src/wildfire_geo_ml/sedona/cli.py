"""
Typer CLI for Sedona corridor and optional hex zonal statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import typer

from wildfire_geo_ml.features.h3_partition import (
    collect_required_cog_paths,
    get_scene_wgs84_bbox,
    resolve_feature_scenes,
)
from wildfire_geo_ml.features.h3_utils import (
    coerce_bbox4,
    filter_cells_to_extent,
    h3_cells_to_geodataframe,
    polyfill_bbox,
)
from wildfire_geo_ml.features.indices import load_scene_indices, write_scene_index_cogs
from wildfire_geo_ml.ingest.config import PipelineConfig, load_pipeline_config
from wildfire_geo_ml.sedona.corridors import fetch_transmission_lines
from wildfire_geo_ml.sedona.session import create_sedona_session
from wildfire_geo_ml.sedona.zonal_stats_job import (
    load_corridor_regions,
    load_hex_regions,
    run_zonal_stats_job,
    stop_spark,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config/pipeline.yaml")
DEFAULT_COG_DIR = Path("data/cog")
RegionChoice = Literal["corridor", "hex", "both"]


def _require_corridors_config(pipeline: PipelineConfig) -> None:
    if pipeline.corridors is None:
        msg = "pipeline.yaml is missing a corridors: section required for Sedona jobs"
        raise ValueError(msg)


def prepare_index_cogs(
    cog_dir: Path,
    scene_ids: list[str],
    pipeline: PipelineConfig,
    *,
    indices_dir: Path | None = None,
) -> Path:
    """
    Write per-scene NDVI/NBR/NDWI GeoTIFFs for Sedona raster SQL.

    Parameters
    ----------
    cog_dir : Path
        Landsat SR COG root directory.
    scene_ids : list[str]
        Scene IDs to process.
    pipeline : PipelineConfig
        Validated pipeline configuration.
    indices_dir : Path, optional
        Output root for index COGs.

    Returns
    -------
    Path
        Index COG output directory.
    """
    out_dir = indices_dir or Path(pipeline.features.indices_dir)
    features = pipeline.features
    for scene_id in scene_ids:
        logger.info("Writing index COGs for scene %s", scene_id)
        cog_paths = collect_required_cog_paths(cog_dir, scene_id, features.required_bands)
        index_arrays = load_scene_indices(cog_paths)
        write_scene_index_cogs(index_arrays, out_dir, scene_id)
    return out_dir


def export_hex_regions_geojson(
    scene_id: str,
    cog_dir: Path,
    pipeline: PipelineConfig,
    output_path: Path,
) -> Path:
    """
    Export study∩scene H3 hex polygons as GeoJSON for optional Sedona hex stats.

    Parameters
    ----------
    scene_id : str
        Landsat scene ID used to clip hex cells.
    cog_dir : Path
        COG root directory.
    pipeline : PipelineConfig
        Pipeline configuration.
    output_path : Path
        GeoJSON output path.

    Returns
    -------
    Path
        Written GeoJSON path.
    """
    features = pipeline.features
    cog_paths = collect_required_cog_paths(cog_dir, scene_id, features.required_bands)
    study_bbox = coerce_bbox4(pipeline.study_area.bbox)
    study_cells = polyfill_bbox(study_bbox, features.h3_resolution)
    scene_bbox = get_scene_wgs84_bbox(cog_paths["B4"])
    scene_cells = filter_cells_to_extent(study_cells, scene_bbox)
    hex_gdf = h3_cells_to_geodataframe(scene_cells)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hex_gdf.to_file(output_path, driver="GeoJSON")
    return output_path


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
    region: RegionChoice = typer.Option(
        "corridor",
        "--region",
        case_sensitive=False,
        help="Region layer(s) to summarize",
    ),
    cog_dir: Path = typer.Option(
        DEFAULT_COG_DIR,
        "--cog-dir",
        file_okay=False,
        help="Landsat SR COG root directory",
    ),
    indices_dir: Path | None = typer.Option(
        None,
        "--indices-dir",
        file_okay=False,
        help="Per-scene index COG directory (default from config)",
    ),
    lines_path: Path | None = typer.Option(
        None,
        "--lines",
        dir_okay=False,
        help="Cached transmission line GeoJSON (default from config)",
    ),
    hex_regions: Path | None = typer.Option(
        None,
        "--hex-regions",
        dir_okay=False,
        help="Hex region GeoJSON for optional Sedona hex stats",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        file_okay=False,
        help="Output directory override",
    ),
    master: str | None = typer.Option(
        None,
        "--master",
        help="Spark master override (default SPARK_MASTER or local[*])",
    ),
    prepare_indices: bool = typer.Option(
        True,
        "--prepare-indices/--no-prepare-indices",
        help="Write NDVI/NBR/NDWI COGs before running Sedona",
    ),
    fetch_lines: bool = typer.Option(
        True,
        "--fetch-lines/--no-fetch-lines",
        help="Fetch/cache CEC transmission lines for the study bbox",
    ),
    wrs_path: str | None = typer.Option(
        None,
        "--path",
        help="Filter scenes by WRS-2 path",
    ),
    rows: list[str] = typer.Option(
        [],
        "--rows",
        help="Filter scenes by WRS-2 row(s)",
    ),
    dates: list[str] = typer.Option(
        [],
        "--dates",
        help="Filter scenes by acquisition date YYYYMMDD",
    ),
) -> None:
    """
    Run Sedona RS_ZonalStats over transmission corridors and optional H3 hexes.

    Writes per-scene index COGs locally, then executes distributed zonal stats in
    Sedona on ``local[*]`` (or ``SPARK_MASTER`` for EMR Serverless).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pipeline = load_pipeline_config(config_path)
    _require_corridors_config(pipeline)
    corridors = pipeline.corridors
    assert corridors is not None

    path_filter = wrs_path if wrs_path is not None else pipeline.ingest.wrs_path
    scene_ids = resolve_feature_scenes(
        cog_dir,
        wrs_path=path_filter,
        rows=list(rows) if rows else None,
        dates=list(dates) if dates else None,
    )
    if not scene_ids:
        typer.echo(f"No scenes found under {cog_dir}", err=True)
        raise typer.Exit(1)

    indices_out = indices_dir or Path(pipeline.features.indices_dir)
    if prepare_indices:
        prepare_index_cogs(cog_dir, scene_ids, pipeline, indices_dir=indices_out)

    bbox = coerce_bbox4(pipeline.study_area.bbox)
    cached_lines = lines_path or Path(corridors.cached_geojson)
    if fetch_lines:
        fetch_transmission_lines(
            bbox,
            cached_lines,
            rest_service_url=corridors.rest_service_url,
        )

    sedona_cfg = pipeline.sedona
    spark = create_sedona_session(
        app_name=sedona_cfg.app_name,
        master=master,
        jar_packages=sedona_cfg.jar_packages,
        shuffle_partitions=str(sedona_cfg.shuffle_partitions),
    )

    try:
        stat_names = list(pipeline.features.stat_names)
        if region in ("corridor", "both"):
            corridor_out = output_dir or Path(sedona_cfg.corridor_output_dir)
            regions_df = load_corridor_regions(
                spark,
                cached_lines,
                buffer_m=corridors.buffer_m,
                metric_crs=corridors.metric_crs,
            )
            run_zonal_stats_job(
                spark,
                indices_dir=indices_out,
                regions_df=regions_df,
                region_kind="corridor",
                output_dir=corridor_out,
                stat_names=stat_names,
            )
            typer.echo(f"Corridor zonal stats written to {corridor_out}")

        if region in ("hex", "both"):
            hex_out = output_dir or Path(sedona_cfg.hex_output_dir)
            hex_path = hex_regions
            if hex_path is None:
                hex_path = indices_out / "hex_regions.geojson"
                export_hex_regions_geojson(scene_ids[0], cog_dir, pipeline, hex_path)
            regions_df = load_hex_regions(
                spark,
                hex_path,
                h3_resolution=pipeline.features.h3_resolution,
            )
            run_zonal_stats_job(
                spark,
                indices_dir=indices_out,
                regions_df=regions_df,
                region_kind="hex",
                output_dir=hex_out,
                stat_names=stat_names,
                h3_resolution=pipeline.features.h3_resolution,
            )
            typer.echo(f"Hex Sedona zonal stats written to {hex_out}")
    finally:
        stop_spark(spark)


if __name__ == "__main__":
    app()
