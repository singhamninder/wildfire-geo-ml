"""
Sedona zonal statistics over transmission corridors and optional H3 hex regions.

Uses the Sedona ``raster`` loader (GeoTIFF path reads) and ``RS_ZonalStats`` /
``RS_ZonalStatsAll`` for Earth-Engine-style reduceRegions over vector zones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F  # noqa: N812

RegionKind = Literal["corridor", "hex"]


def discover_index_cog_paths(indices_dir: Path) -> list[tuple[str, str, str]]:
    """
    Discover per-scene index COG paths under ``indices_dir``.

    Parameters
    ----------
    indices_dir : Path
        Root directory containing ``<scene_id>/<index>.tif`` files.

    Returns
    -------
    list[tuple[str, str, str]]
        Rows of ``(scene_id, index_name, absolute_cog_path)``.
    """
    if not indices_dir.is_dir():
        msg = f"Index COG directory not found: {indices_dir}"
        raise ValueError(msg)

    rows: list[tuple[str, str, str]] = []
    for scene_dir in sorted(indices_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for cog_path in sorted(scene_dir.glob("*.tif")):
            rows.append((scene_dir.name, cog_path.stem, str(cog_path.resolve())))
    if not rows:
        msg = f"No index COGs found under {indices_dir}"
        raise ValueError(msg)
    return rows


def load_index_rasters(
    spark: SparkSession,
    index_rows: list[tuple[str, str, str]],
) -> DataFrame:
    """
    Load index GeoTIFFs as Sedona raster objects via ``binaryFile`` + ``RS_FromGeoTiff``.

    Parameters
    ----------
    spark : SparkSession
        Active Sedona-enabled Spark session.
    index_rows : list[tuple[str, str, str]]
        ``(scene_id, index_name, cog_path)`` rows.

    Returns
    -------
    DataFrame
        Columns: ``scene_id``, ``index_name``, ``cog_path``, ``raster``.
    """
    paths_df = spark.createDataFrame(index_rows, ["scene_id", "index_name", "cog_path"])
    cog_paths = [row[2] for row in index_rows]
    binary_df = spark.read.format("binaryFile").load(cog_paths)
    binary_df = binary_df.withColumnRenamed("path", "cog_path")
    return (
        binary_df.join(paths_df, on="cog_path", how="inner")
        .withColumn("raster", F.expr("RS_FromGeoTiff(content)"))
        .select("scene_id", "index_name", "cog_path", "raster")
    )


def load_corridor_regions(
    spark: SparkSession,
    geojson_path: Path,
    *,
    buffer_m: float,
    metric_crs: str,
) -> DataFrame:
    """
    Load transmission lines and buffer them for corridor zonal statistics.

    Parameters
    ----------
    spark : SparkSession
        Active Sedona-enabled Spark session.
    geojson_path : Path
        Cached CEC transmission line GeoJSON path.
    buffer_m : float
        Buffer distance in meters.
    metric_crs : str
        Metric CRS used for buffering (e.g. EPSG:32610).

    Returns
    -------
    DataFrame
        Columns: ``region_id``, ``geometry``, ``region_kind``.
    """
    if not geojson_path.is_file():
        msg = f"Transmission line GeoJSON not found: {geojson_path}"
        raise FileNotFoundError(msg)

    lines_df = spark.read.format("geojson").load(str(geojson_path))
    id_col = "OBJECTID" if "OBJECTID" in lines_df.columns else None
    if id_col is None:
        lines_df = lines_df.withColumn("region_id", F.monotonically_increasing_id().cast("string"))
    else:
        lines_df = lines_df.withColumn("region_id", F.col(id_col).cast("string"))

    buffer_expr = (
        f"ST_Transform("
        f"ST_Buffer(ST_Transform(geometry, '{metric_crs}'), {buffer_m}), "
        f"'{metric_crs}')"
    )
    return (
        lines_df.withColumn("geometry", F.expr(buffer_expr))
        .withColumn("region_kind", F.lit("corridor"))
        .select("region_id", "geometry", "region_kind")
    )


def load_hex_regions(
    spark: SparkSession,
    geojson_path: Path,
    *,
    h3_resolution: int,
) -> DataFrame:
    """
    Load H3 hex polygons exported as GeoJSON for optional Sedona hex zonal stats.

    Parameters
    ----------
    spark : SparkSession
        Active Sedona-enabled Spark session.
    geojson_path : Path
        GeoJSON file with hex geometries and ``h3_index`` property.
    h3_resolution : int
        H3 resolution used for ``h3_res8`` partition column naming.

    Returns
    -------
    DataFrame
        Columns: ``region_id``, ``geometry``, ``region_kind``, ``h3_res8``.
    """
    if not geojson_path.is_file():
        msg = f"Hex region GeoJSON not found: {geojson_path}"
        raise FileNotFoundError(msg)

    hex_df = spark.read.format("geojson").load(str(geojson_path))
    id_col = "h3_index" if "h3_index" in hex_df.columns else "region_id"
    partition_col = f"h3_res{h3_resolution}"
    return (
        hex_df.withColumn("region_id", F.col(id_col).cast("string"))
        .withColumn(partition_col, F.col(id_col).cast("string"))
        .withColumn("region_kind", F.lit("hex"))
        .select("region_id", "geometry", "region_kind", partition_col)
    )


def compute_region_zonal_stats(
    rasters_df: DataFrame,
    regions_df: DataFrame,
    *,
    region_kind: RegionKind,
    stat_names: list[str],
    h3_resolution: int | None = None,
) -> DataFrame:
    """
    Compute zonal statistics for each raster tile row and region geometry.

    Parameters
    ----------
    rasters_df : DataFrame
        Index rasters with ``scene_id``, ``index_name``, ``raster``.
    regions_df : DataFrame
        Region geometries with ``region_id`` and ``region_kind``.
    region_kind : RegionKind
        ``corridor`` or ``hex``.
    stat_names : list[str]
        Statistics to extract from ``RS_ZonalStatsAll`` (mean, std, max supported).
    h3_resolution : int, optional
        Required when ``region_kind`` is ``hex`` for partition column naming.

    Returns
    -------
    DataFrame
        Long-format stats with ``region_kind``, ``scene_id``, ``index_name``,
        requested statistics, and optional ``h3_res8`` partition column.
    """
    joined = rasters_df.crossJoin(F.broadcast(regions_df))
    stats_df = joined.withColumn(
        "stats",
        F.expr("RS_ZonalStatsAll(raster, geometry, 1, true, true)"),
    ).filter(F.col("stats").isNotNull())

    select_exprs: list[str] = [
        "region_id",
        "region_kind",
        "scene_id",
        "index_name",
    ]
    if region_kind == "hex" and h3_resolution is not None:
        select_exprs.append(f"h3_res{h3_resolution}")

    for stat_name in stat_names:
        sedona_field = _stat_field_name(stat_name)
        select_exprs.append(f"stats.{sedona_field} as {stat_name}")

    if "max" not in stat_names:
        select_exprs.append("stats.max as max")

    return stats_df.selectExpr(*select_exprs)


def pivot_index_stats(long_df: DataFrame, stat_names: list[str]) -> DataFrame:
    """
    Pivot long-format index statistics to wide columns per region.

    Parameters
    ----------
    long_df : DataFrame
        Long-format output from ``compute_region_zonal_stats``.
    stat_names : list[str]
        Statistics that were computed.

    Returns
    -------
    DataFrame
        One row per region with ``{index}_{stat}`` columns.
    """
    group_cols = ["region_id", "region_kind", "scene_id"]
    partition_col = next((col for col in long_df.columns if col.startswith("h3_res")), None)
    if partition_col is not None:
        group_cols.append(partition_col)

    agg_exprs = []
    for index_name in ("ndvi", "nbr", "ndwi"):
        for stat_name in stat_names:
            if stat_name in long_df.columns:
                agg_exprs.append(
                    F.max(F.when(F.col("index_name") == index_name, F.col(stat_name))).alias(
                        f"{index_name}_{stat_name}"
                    )
                )
        if "max" in long_df.columns:
            agg_exprs.append(
                F.max(F.when(F.col("index_name") == index_name, F.col("max"))).alias(
                    f"{index_name}_max"
                )
            )
    return long_df.groupBy(*group_cols).agg(*agg_exprs)


def _stat_field_name(stat_name: str) -> str:
    """
    Map config stat names to RS_ZonalStatsAll struct field names.

    Parameters
    ----------
    stat_name : str
        Config statistic name such as ``mean`` or ``std``.

    Returns
    -------
    str
        Sedona struct field name.
    """
    if stat_name == "std":
        return "stddev"
    return stat_name


def run_zonal_stats_job(
    spark: SparkSession,
    *,
    indices_dir: Path,
    regions_df: DataFrame,
    region_kind: RegionKind,
    output_dir: Path,
    stat_names: list[str],
    h3_resolution: int | None = None,
) -> Path:
    """
    Run corridor or hex zonal statistics and write GeoParquet output.

    Parameters
    ----------
    spark : SparkSession
        Active Sedona-enabled Spark session.
    indices_dir : Path
        Directory with per-scene index COGs.
    regions_df : DataFrame
        Region geometries from ``load_corridor_regions`` or ``load_hex_regions``.
    region_kind : RegionKind
        ``corridor`` or ``hex``.
    output_dir : Path
        Output directory for GeoParquet files.
    stat_names : list[str]
        Statistics to compute.
    h3_resolution : int, optional
        H3 resolution when writing hex partitions.

    Returns
    -------
    Path
        Output directory path.
    """
    index_rows = discover_index_cog_paths(indices_dir)
    rasters_df = load_index_rasters(spark, index_rows)
    long_df = compute_region_zonal_stats(
        rasters_df,
        regions_df,
        region_kind=region_kind,
        stat_names=stat_names,
        h3_resolution=h3_resolution,
    )
    wide_df = pivot_index_stats(long_df, stat_names)

    writer = wide_df.write.mode("overwrite").format("geoparquet")
    if region_kind == "hex" and h3_resolution is not None:
        writer = writer.partitionBy(f"h3_res{h3_resolution}")
    writer.save(str(output_dir))
    return output_dir


def stop_spark(spark: SparkSession) -> None:
    """
    Stop an active Spark session.

    Parameters
    ----------
    spark : SparkSession
        Session returned by ``create_sedona_session``.
    """
    spark.stop()
