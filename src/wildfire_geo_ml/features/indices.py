"""
Spectral index computation for Landsat-9 Surface Reflectance bands.

NDVI, NBR, and NDWI are the canonical vegetation, burn severity, and moisture
indices used in wildfire risk models. Computing them in rioxarray keeps spatial
metadata (CRS, transform) attached to the output arrays for H3 zonal alignment.
"""

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

# Scale factor for Landsat-9 L2 SR: DN * 0.0000275 + (-0.2) -> reflectance
LANDSAT9_SCALE = 0.0000275
LANDSAT9_OFFSET = -0.2
# Fill value for L2 products
LANDSAT9_FILL = 0

REQUIRED_INDEX_BANDS = frozenset({"B3", "B4", "B5", "B7"})


def dn_to_reflectance(dn: xr.DataArray) -> xr.DataArray:
    """
    Convert Landsat-9 L2 SR digital numbers to surface reflectance.

    Parameters
    ----------
    dn : xr.DataArray
        Raw DN values from a Landsat SR COG.

    Returns
    -------
    xr.DataArray
        Reflectance in approximately [0, 1] with fill pixels set to NaN.
    """
    scaled = dn.where(dn != LANDSAT9_FILL, np.nan)
    return scaled * LANDSAT9_SCALE + LANDSAT9_OFFSET


def load_band(cog_path: Path, band: int = 1) -> xr.DataArray:
    """
    Load a Landsat-9 SR band COG as a scaled reflectance DataArray.

    Parameters
    ----------
    cog_path : Path
        Path to a single-band COG.
    band : int
        Band index to read (1-indexed). Default 1.

    Returns
    -------
    xr.DataArray
        Reflectance values with spatial CRS attached.
    """
    da = xr.open_dataarray(str(cog_path), engine="rasterio", mask_and_scale=False)
    if "band" in da.dims:
        da = da.isel(band=band - 1)
    return dn_to_reflectance(da)


def compute_ndvi(red: xr.DataArray, nir: xr.DataArray) -> xr.DataArray:
    """
    Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red).

    Range: [-1, 1]. Dense healthy vegetation is typically 0.6–0.9.

    Parameters
    ----------
    red : xr.DataArray
        Red band reflectance (B4).
    nir : xr.DataArray
        NIR band reflectance (B5).

    Returns
    -------
    xr.DataArray
        NDVI values retaining source spatial metadata.
    """
    return (nir - red) / (nir + red)


def compute_nbr(nir: xr.DataArray, swir2: xr.DataArray) -> xr.DataArray:
    """
    Normalized Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2).

    Range: [-1, 1]. Post-fire burned areas show strongly negative NBR (< -0.1).

    Parameters
    ----------
    nir : xr.DataArray
        NIR band reflectance (B5).
    swir2 : xr.DataArray
        SWIR2 band reflectance (B7).

    Returns
    -------
    xr.DataArray
        NBR values retaining source spatial metadata.
    """
    return (nir - swir2) / (nir + swir2)


def compute_ndwi(green: xr.DataArray, nir: xr.DataArray) -> xr.DataArray:
    """
    Normalized Difference Water Index: (Green - NIR) / (Green + NIR).

    Range: [-1, 1]. Positive values indicate water or wet vegetation.

    Parameters
    ----------
    green : xr.DataArray
        Green band reflectance (B3).
    nir : xr.DataArray
        NIR band reflectance (B5).

    Returns
    -------
    xr.DataArray
        NDWI values retaining source spatial metadata.
    """
    return (green - nir) / (green + nir)


def compute_all_indices(
    red: xr.DataArray,
    nir: xr.DataArray,
    green: xr.DataArray,
    swir2: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """
    Compute NDVI, NBR, and NDWI in one call.

    Parameters
    ----------
    red : xr.DataArray
        Red band reflectance (B4).
    nir : xr.DataArray
        NIR band reflectance (B5).
    green : xr.DataArray
        Green band reflectance (B3).
    swir2 : xr.DataArray
        SWIR2 band reflectance (B7).

    Returns
    -------
    dict[str, xr.DataArray]
        Keys ``ndvi``, ``nbr``, ``ndwi``. Each array retains source CRS and transform.
    """
    return {
        "ndvi": compute_ndvi(red, nir),
        "nbr": compute_nbr(nir, swir2),
        "ndwi": compute_ndwi(green, nir),
    }


INDEX_NODATA = -9999.0


def write_scene_index_cogs(
    index_arrays: dict[str, xr.DataArray],
    output_dir: Path,
    scene_id: str,
) -> dict[str, Path]:
    """
    Persist NDVI, NBR, and NDWI arrays as float GeoTIFFs for Sedona raster SQL.

    Parameters
    ----------
    index_arrays : dict[str, xr.DataArray]
        Mapping of index name to geo-referenced array.
    output_dir : Path
        Root directory for index COG outputs.
    scene_id : str
        Landsat scene identifier used as a subdirectory name.

    Returns
    -------
    dict[str, Path]
        Mapping of index name to written GeoTIFF path.
    """
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for index_name, data_array in index_arrays.items():
        if data_array.rio.crs is None:
            msg = f"Index array '{index_name}' is missing a CRS; cannot write COG"
            raise ValueError(msg)
        out_path = scene_dir / f"{index_name}.tif"
        values = data_array.astype("float32")
        values = values.where(np.isfinite(values), INDEX_NODATA)
        values = values.rio.write_nodata(INDEX_NODATA)
        values.rio.to_raster(out_path, dtype="float32")
        written[index_name] = out_path
    return written


def load_scene_indices(cog_paths: dict[str, Path]) -> dict[str, xr.DataArray]:
    """
    Load required bands and compute all spectral indices for a scene.

    Parameters
    ----------
    cog_paths : dict[str, Path]
        Mapping of band label (e.g. ``B4``) to local COG path.

    Returns
    -------
    dict[str, xr.DataArray]
        Keys ``ndvi``, ``nbr``, ``ndwi``.

    Raises
    ------
    ValueError
        If any required band is missing from ``cog_paths``.
    """
    missing = REQUIRED_INDEX_BANDS - set(cog_paths.keys())
    if missing:
        msg = f"Missing required bands for spectral indices: {sorted(missing)}"
        raise ValueError(msg)

    green = load_band(cog_paths["B3"])
    red = load_band(cog_paths["B4"])
    nir = load_band(cog_paths["B5"])
    swir2 = load_band(cog_paths["B7"])
    return compute_all_indices(red, nir, green, swir2)
