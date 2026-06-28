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
    # Read GeoTIFF bytes via Spark binaryFile; strip file: prefix for join key match.
    binary_df = spark.read.format("binaryFile").load(cog_paths)
    binary_df = binary_df.withColumn(
        "cog_path",
        F.regexp_replace(F.col("path"), r"^file:", ""),
    )
    return (
        binary_df.join(paths_df, on="cog_path", how="inner")
        .withColumn("raster", F.expr("RS_FromGeoTiff(content)"))
        .select("scene_id", "index_name", "cog_path", "raster")
    )


def _geojson_property_fields(features_df: DataFrame) -> list[str]:
    """Return property field names from an exploded GeoJSON features DataFrame."""
    if "properties" not in features_df.columns:
        return []
    props_type = features_df.schema["properties"].dataType
    field_names = getattr(props_type, "fieldNames", None)
    if callable(field_names):
        return list(field_names())
    return []


def _load_geojson_features(spark: SparkSession, geojson_path: Path) -> DataFrame:
    """
    Load a GeoJSON file as one row per feature with ``geometry`` and ``properties``.

    Sedona's GeoJSON reader returns FeatureCollections with a ``features`` array;
    this helper explodes that array so downstream SQL can reference ``geometry``.
    """
    raw_df = spark.read.format("geojson").load(str(geojson_path))
    if "features" in raw_df.columns:
        return raw_df.select(F.explode("features").alias("feature")).select(
            F.col("feature.geometry").alias("geometry"),
            F.col("feature.properties").alias("properties"),
        )
    if "geometry" in raw_df.columns:
        return raw_df.select("geometry", F.col("properties"))
    msg = f"GeoJSON at {geojson_path} has no features or geometry columns"
    raise ValueError(msg)


def _with_wgs84_srid(features_df: DataFrame) -> DataFrame:
    """Assign EPSG:4326 to geometries loaded from GeoJSON (no embedded SRID)."""
    return features_df.withColumn("geometry", F.expr("ST_SetSRID(geometry, 4326)"))


def load_corridor_regions(
    spark: SparkSession,
    geojson_path: Path,
    *,
    buffer_m: float,
    metric_crs: str,
    clip_bbox_wgs84: tuple[float, float, float, float] | None = None,
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
    clip_bbox_wgs84 : tuple[float, float, float, float], optional
        When set, keep only lines intersecting this WGS84 bbox before buffering.

    Returns
    -------
    DataFrame
        Columns: ``region_id``, ``geometry``, ``region_kind``.
    """
    if not geojson_path.is_file():
        msg = f"Transmission line GeoJSON not found: {geojson_path}"
        raise FileNotFoundError(msg)

    lines_df = _with_wgs84_srid(_load_geojson_features(spark, geojson_path))
    prop_fields = _geojson_property_fields(lines_df)
    # Prefer stable CEC OBJECTID; fall back to monotonic ID when absent.
    if "OBJECTID" in prop_fields:
        lines_df = lines_df.withColumn("region_id", F.col("properties.OBJECTID").cast("string"))
    else:
        lines_df = lines_df.withColumn("region_id", F.monotonically_increasing_id().cast("string"))

    if clip_bbox_wgs84 is not None:
        # Clip lines to study∩scene bbox before buffering to reduce join cost.
        lines_df = filter_regions_to_wgs84_bbox(lines_df, clip_bbox_wgs84)

    # Buffer in metric CRS: WGS84 -> metric -> ST_Buffer -> metric (keep UTM alignment).
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
    metric_crs: str | None = None,
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
    metric_crs : str, optional
        When set, reproject hex polygons to this CRS so they align with UTM index COGs.

    Returns
    -------
    DataFrame
        Columns: ``region_id``, ``geometry``, ``region_kind``, ``h3_res8``.
    """
    if not geojson_path.is_file():
        msg = f"Hex region GeoJSON not found: {geojson_path}"
        raise FileNotFoundError(msg)

    hex_df = _with_wgs84_srid(_load_geojson_features(spark, geojson_path))
    prop_fields = _geojson_property_fields(hex_df)
    if "h3_index" in prop_fields:
        id_expr = F.col("properties.h3_index").cast("string")
    elif "region_id" in prop_fields:
        id_expr = F.col("properties.region_id").cast("string")
    else:
        id_expr = F.monotonically_increasing_id().cast("string")
    partition_col = f"h3_res{h3_resolution}"
    if metric_crs is not None:
        # Reproject hexes to UTM so they align with index COG rasters.
        hex_df = hex_df.withColumn("geometry", F.expr(f"ST_Transform(geometry, '{metric_crs}')"))
    return (
        hex_df.withColumn("region_id", id_expr)
        .withColumn(partition_col, id_expr)
        .withColumn("region_kind", F.lit("hex"))
        .select("region_id", "geometry", "region_kind", partition_col)
    )


def filter_regions_to_wgs84_bbox(
    regions_df: DataFrame,
    bbox_wgs84: tuple[float, float, float, float],
) -> DataFrame:
    """
    Keep only region geometries that intersect a WGS84 bounding box.

    Parameters
    ----------
    regions_df : DataFrame
        Regions with a ``geometry`` column.
    bbox_wgs84 : tuple[float, float, float, float]
        Bounding box as (west, south, east, north).

    Returns
    -------
    DataFrame
        Filtered regions.
    """
    west, south, east, north = bbox_wgs84
    # Build WKT bbox polygon inline for ST_Intersects spatial filter.
    bbox_wkt = (
        f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"
    )
    return regions_df.filter(
        F.expr(f"ST_Intersects(geometry, ST_SetSRID(ST_GeomFromWKT('{bbox_wkt}'), 4326))")
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
    joined = rasters_df.crossJoin(F.broadcast(regions_df)).filter(
        F.expr("ST_Intersects(RS_Envelope(raster), geometry)")
    )
    # RS_ZonalStatsAll args: band=1, allTouched=true, noData=true.
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
        # Always retain max for downstream burn-severity thresholding.
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
    # Pivot the three spectral indices computed in Phase 2 feature engineering.
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
    scene_ids = sorted({row[0] for row in index_rows})
    geom_df = regions_df.select("region_id", "geometry")
    partition_col = f"h3_res{h3_resolution}" if h3_resolution is not None else None

    for scene_index, scene_id in enumerate(scene_ids):
        scene_rows = [row for row in index_rows if row[0] == scene_id]
        long_parts: list[DataFrame] = []
        for index_row in scene_rows:
            rasters_df = load_index_rasters(spark, [index_row])
            long_parts.append(
                compute_region_zonal_stats(
                    rasters_df,
                    regions_df,
                    region_kind=region_kind,
                    stat_names=stat_names,
                    h3_resolution=h3_resolution,
                )
            )
        long_df = long_parts[0]
        for part in long_parts[1:]:
            long_df = long_df.unionByName(part)
        wide_df = pivot_index_stats(long_df, stat_names).join(geom_df, on="region_id", how="left")

        # Overwrite on first scene; append subsequent scenes; partition hex output by h3_res8.
        write_mode = "overwrite" if scene_index == 0 else "append"
        writer = wide_df.write.mode(write_mode).format("geoparquet")
        if region_kind == "hex" and partition_col is not None:
            writer = writer.partitionBy(partition_col)
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
