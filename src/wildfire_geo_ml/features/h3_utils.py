"""
H3 hexagonal indexing utilities for spatial feature engineering.

H3 provides a globally consistent spatial index. Polyfilling the study-area bbox
yields the set of hexagonal units for zonal statistics — analogous to a fishnet
grid, but hierarchical and uniform-adjacency.
"""

from typing import Any

import geopandas as gpd
import h3
from h3 import LatLngPoly
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from wildfire_geo_ml.features.geopandas_io import gdf_from_records

H3_DEFAULT_RESOLUTION = 8


def coerce_bbox4(bbox: tuple[float, ...] | list[float]) -> tuple[float, float, float, float]:
    """
    Validate and return a 4-element EPSG:4326 bounding box.

    Parameters
    ----------
    bbox : tuple or list of float
        Bounding box as west, south, east, north.

    Returns
    -------
    tuple[float, float, float, float]
        Validated bbox tuple.

    Raises
    ------
    ValueError
        If bbox does not contain exactly four values.
    """
    if len(bbox) != 4:
        msg = f"bbox must have 4 values [west, south, east, north], got {len(bbox)}"
        raise ValueError(msg)
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _bbox_to_latlng_ring(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """
    Convert an EPSG:4326 bbox to a closed lat/lng ring for H3.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box as (west, south, east, north).

    Returns
    -------
    list[tuple[float, float]]
        Closed ring of (lat, lng) vertices.
    """
    west, south, east, north = bbox
    return [
        (south, west),
        (south, east),
        (north, east),
        (north, west),
        (south, west),
    ]


def polyfill_bbox(
    bbox: tuple[float, float, float, float] | list[float],
    resolution: int = H3_DEFAULT_RESOLUTION,
) -> list[str]:
    """
    Fill a bounding box with H3 hexagons at the specified resolution.

    Parameters
    ----------
    bbox : tuple or list of float
        Bounding box as (west, south, east, north) in EPSG:4326 decimal degrees.
    resolution : int
        H3 resolution level. Default 8 (~461 m edge length).

    Returns
    -------
    list[str]
        Sorted H3 cell IDs covering the bbox.
    """
    bbox_tuple = coerce_bbox4(bbox)
    ring = _bbox_to_latlng_ring(bbox_tuple)
    cells = h3.polygon_to_cells(LatLngPoly(ring), resolution)
    return sorted(cells)


def h3_cell_to_polygon(cell: str) -> Polygon:
    """
    Convert an H3 cell ID to a Shapely polygon in EPSG:4326.

    Parameters
    ----------
    cell : str
        H3 cell ID.

    Returns
    -------
    Polygon
        Hexagon boundary as a GeoJSON-style polygon (x=lng, y=lat).
    """
    boundary = h3.cell_to_boundary(cell)
    coords = [(lng, lat) for lat, lng in boundary]
    return Polygon(coords)


def h3_cells_to_geodataframe(
    cells: list[str],
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Convert H3 cell IDs to a GeoDataFrame of hexagon polygons.

    Parameters
    ----------
    cells : list[str]
        H3 cell IDs from :func:`polyfill_bbox`.
    crs : str
        CRS for the output GeoDataFrame. Default EPSG:4326.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: ``h3_index``, ``h3_res``, ``geometry``.
    """
    records: list[dict[str, Any]] = []
    for cell in cells:
        records.append(
            {
                "h3_index": cell,
                "h3_res": h3.get_resolution(cell),
                "geometry": h3_cell_to_polygon(cell),
            }
        )
    return gdf_from_records(records, crs=crs)


def filter_cells_to_extent(
    cells: list[str],
    scene_bbox: tuple[float, float, float, float] | list[float],
) -> list[str]:
    """
    Keep H3 cells whose centroids fall within a scene bounding box.

    Parameters
    ----------
    cells : list[str]
        Candidate H3 cell IDs.
    scene_bbox : tuple or list of float
        Scene extent as (west, south, east, north) in EPSG:4326.

    Returns
    -------
    list[str]
        Filtered cell IDs intersecting the scene extent.
    """
    bbox_tuple = coerce_bbox4(scene_bbox)
    extent = box(*bbox_tuple)
    filtered: list[str] = []
    for cell in cells:
        poly = h3_cell_to_polygon(cell)
        if extent.intersects(poly):
            filtered.append(cell)
    return filtered


def filter_cells_to_geometry(
    cells: list[str],
    geometry: BaseGeometry,
    geometry_crs: str,
) -> list[str]:
    """
    Keep H3 cells whose hex polygons intersect a scene geometry.

    Parameters
    ----------
    cells : list[str]
        Candidate H3 cell IDs.
    geometry : shapely geometry
        Scene extent geometry in ``geometry_crs`` (e.g. valid-data footprint).
    geometry_crs : str
        CRS of ``geometry`` (e.g. EPSG:32610 for Landsat UTM tiles).

    Returns
    -------
    list[str]
        Filtered cell IDs intersecting ``geometry``.
    """
    if not cells:
        return []
    if geometry is None or geometry.is_empty:
        return []

    hex_gdf = h3_cells_to_geodataframe(cells)
    hex_proj = hex_gdf.to_crs(geometry_crs)
    filtered: list[str] = []
    for i in range(len(hex_proj)):
        cell_geom = hex_proj.iloc[i].geometry
        if cell_geom is None or cell_geom.is_empty:
            continue
        if geometry.intersects(cell_geom):
            filtered.append(str(hex_gdf.iloc[i]["h3_index"]))
    return filtered
