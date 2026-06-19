"""Geopandas helpers with localized type-checker suppressions for incomplete stubs."""

from typing import Any

import geopandas as gpd
import pandas as pd


def gdf_from_records(records: list[dict[str, Any]], *, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame from row records with a geometry column.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Row dicts including a ``geometry`` key.
    crs : str
        Coordinate reference system for the output frame.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with the supplied records.
    """
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)  # ty: ignore[no-matching-overload]


def empty_gdf(*, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Return an empty GeoDataFrame with geometry column initialized.

    Parameters
    ----------
    crs : str
        Coordinate reference system for the output frame.

    Returns
    -------
    gpd.GeoDataFrame
        Empty GeoDataFrame.
    """
    return gpd.GeoDataFrame(  # ty: ignore[no-matching-overload]
        {"h3_index": pd.Series(dtype="string")},
        geometry=gpd.GeoSeries([], dtype=object),
        crs=crs,
    )
