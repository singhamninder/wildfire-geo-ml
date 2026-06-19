"""
Zonal statistics of spectral indices over H3 hexagons.

Aggregates pixel values within each H3 cell — collapsing thousands of pixels
into one feature row per cell for downstream ML. Uses rasterio masking locally;
corridor-scale aggregation is handled by Sedona RS_ZonalStats in Phase 3.
"""

from contextlib import ExitStack
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.io import DatasetReader, MemoryFile
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from wildfire_geo_ml.features.geopandas_io import empty_gdf, gdf_from_records
from wildfire_geo_ml.features.h3_utils import h3_cells_to_geodataframe


def _empty_stats_gdf() -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame with the expected zonal-stats schema."""
    frame = empty_gdf()
    frame["h3_res8"] = pd.Series(dtype="string")
    frame["pixel_count"] = pd.Series(dtype="int64")
    return frame


def _write_array_to_memory(da: xr.DataArray) -> MemoryFile:
    """
    Write a rioxarray DataArray to an in-memory GeoTIFF.

    Parameters
    ----------
    da : xr.DataArray
        Geo-referenced single-band array.

    Returns
    -------
    MemoryFile
        Open memory file positioned at the start of the raster.
    """
    memfile = MemoryFile()
    da.rio.to_raster(memfile.name)
    return memfile


def _open_index_rasters(
    index_arrays: dict[str, xr.DataArray],
    stack: ExitStack,
) -> dict[str, DatasetReader]:
    """
    Write each index array to an in-memory GeoTIFF once and return open readers.

    Parameters
    ----------
    index_arrays : dict[str, xr.DataArray]
        Mapping of index name to geo-referenced array.
    stack : ExitStack
        Context manager that owns reader lifetime for the scene.

    Returns
    -------
    dict[str, DatasetReader]
        Open rasterio datasets keyed by index name.
    """
    readers: dict[str, DatasetReader] = {}
    for index_name, data_array in index_arrays.items():
        if data_array.rio.crs is None:
            msg = f"Index array '{index_name}' is missing a CRS; cannot mask by geometry"
            raise ValueError(msg)
        memfile = _write_array_to_memory(data_array)
        stack.enter_context(memfile)
        readers[index_name] = stack.enter_context(memfile.open())
    return readers


def _mask_valid_values(src: DatasetReader, geometry: BaseGeometry) -> np.ndarray:
    """
    Extract valid (non-NaN) pixel values under a polygon geometry.

    Parameters
    ----------
    src : DatasetReader
        Open geo-referenced index raster in the same CRS as ``geometry``.
    geometry : shapely geometry
        Polygon in the same CRS as ``src``.

    Returns
    -------
    np.ndarray
        Flattened valid pixel values; empty when no overlap.
    """
    geom_geojson = mapping(geometry)
    if src.crs is None:
        msg = "Index raster is missing a CRS; cannot mask by geometry"
        raise ValueError(msg)
    data, _ = mask(src, [geom_geojson], crop=True, nodata=np.nan, all_touched=True)
    vals = data[0].astype(np.float64).flatten()
    return vals[~np.isnan(vals)]


def _compute_stats(values: np.ndarray, stat_names: list[str]) -> dict[str, float | int | None]:
    """
    Compute requested summary statistics for a 1-D value array.

    Parameters
    ----------
    values : np.ndarray
        Pixel values (may be empty).
    stat_names : list[str]
        Statistics to compute, e.g. ``["mean", "std"]``.

    Returns
    -------
    dict[str, float | int | None]
        Statistic names mapped to values, or None when empty.
    """
    if values.size == 0:
        return dict.fromkeys(stat_names)

    stats: dict[str, float | int | None] = {"pixel_count": int(values.size)}
    if "mean" in stat_names:
        stats["mean"] = float(np.mean(values))
    if "std" in stat_names:
        stats["std"] = float(np.std(values))
    if "min" in stat_names:
        stats["min"] = float(np.min(values))
    if "max" in stat_names:
        stats["max"] = float(np.max(values))
    if "count" in stat_names:
        stats["count"] = int(values.size)
    return stats


def compute_scene_h3_stats(
    index_arrays: dict[str, xr.DataArray],
    h3_cells: list[str],
    stat_names: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute zonal statistics for spectral index rasters over H3 hexagons.

    Parameters
    ----------
    index_arrays : dict[str, xr.DataArray]
        Mapping of index name (``ndvi``, ``nbr``, ``ndwi``) to geo-referenced array.
    h3_cells : list[str]
        H3 cell IDs to summarize.
    stat_names : list[str], optional
        Statistics to compute. Default: ``["mean", "std"]``.

    Returns
    -------
    gpd.GeoDataFrame
        One row per H3 cell with geometry, ``pixel_count``, and per-index stats.
        Rows with zero valid pixels are dropped.
    """
    if stat_names is None:
        stat_names = ["mean", "std"]
    if not h3_cells:
        return _empty_stats_gdf()

    reference = next(iter(index_arrays.values()))
    if reference.rio.crs is None:
        msg = "Index arrays must have a CRS for zonal statistics"
        raise ValueError(msg)

    hex_gdf = h3_cells_to_geodataframe(h3_cells)
    hex_proj = hex_gdf.to_crs(reference.rio.crs)

    records: list[dict[str, Any]] = []
    with ExitStack() as stack:
        index_readers = _open_index_rasters(index_arrays, stack)
        for i in range(len(hex_proj)):
            row = hex_proj.iloc[i]
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            record: dict[str, Any] = {
                "h3_index": row["h3_index"],
                "h3_res8": row["h3_index"],
                "geometry": hex_gdf.iloc[i].geometry,
            }

            pixel_count: int | None = None
            for index_name, src in index_readers.items():
                values = _mask_valid_values(src, geom)
                stats = _compute_stats(values, stat_names)
                raw_count = stats.get("pixel_count")
                if isinstance(raw_count, (int, float)):
                    pixel_count = int(raw_count)
                for stat_name in stat_names:
                    column = f"{index_name}_{stat_name}"
                    record[column] = stats.get(stat_name)

            if pixel_count is None or pixel_count == 0:
                continue

            record["pixel_count"] = pixel_count
            records.append(record)

    if not records:
        return _empty_stats_gdf()

    result = gdf_from_records(records)
    return result


def stats_to_dataframe(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Drop geometry column for tabular inspection or parquet metadata.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Zonal statistics GeoDataFrame.

    Returns
    -------
    pd.DataFrame
        Non-spatial columns only.
    """
    return pd.DataFrame(gdf.drop(columns="geometry"))
