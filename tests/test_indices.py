"""Tests for Landsat-9 spectral index computation."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.conftest import FEATURE_BANDS, SCENE_ID, write_scaled_sr_cog
from wildfire_geo_ml.features.indices import (
    compute_all_indices,
    compute_nbr,
    compute_ndvi,
    compute_ndwi,
    dn_to_reflectance,
    load_band,
    load_scene_indices,
)
from wildfire_geo_ml.ingest.landsat_paths import local_band_path


def test_dn_to_reflectance_scales_and_masks_fill() -> None:
    da = xr.DataArray(np.array([0, 10000], dtype=np.uint16))
    result = dn_to_reflectance(da)
    assert np.isnan(result.values[0])
    assert result.values[1] == pytest.approx(0.075)


def test_compute_ndvi_known_values() -> None:
    red = xr.DataArray(0.1)
    nir = xr.DataArray(0.5)
    ndvi = compute_ndvi(red, nir)
    assert float(ndvi) == pytest.approx(0.6666666667, rel=1e-6)


def test_compute_nbr_known_values() -> None:
    nir = xr.DataArray(0.5)
    swir2 = xr.DataArray(0.2)
    nbr = compute_nbr(nir, swir2)
    assert float(nbr) == pytest.approx(0.4285714286, rel=1e-6)


def test_compute_ndwi_known_values() -> None:
    green = xr.DataArray(0.3)
    nir = xr.DataArray(0.5)
    ndwi = compute_ndwi(green, nir)
    assert float(ndwi) == pytest.approx(-0.25)


def test_compute_all_indices_keys() -> None:
    arrays = compute_all_indices(
        xr.DataArray(0.1),
        xr.DataArray(0.5),
        xr.DataArray(0.3),
        xr.DataArray(0.2),
    )
    assert set(arrays) == {"ndvi", "nbr", "ndwi"}


def test_load_band_from_cog(tmp_path: Path) -> None:
    cog_path = tmp_path / "B4.tif"
    write_scaled_sr_cog(cog_path, 0.1)
    band = load_band(cog_path)
    assert float(band.mean()) == pytest.approx(0.1, rel=1e-3)
    assert band.rio.crs is not None


def test_load_scene_indices_from_cogs(tmp_path: Path) -> None:
    reflectance = {
        "B3": 0.3,
        "B4": 0.1,
        "B5": 0.5,
        "B7": 0.2,
    }
    cog_paths: dict[str, Path] = {}
    for band, value in reflectance.items():
        cog_path = local_band_path(tmp_path, SCENE_ID, band)
        write_scaled_sr_cog(cog_path, value)
        cog_paths[band] = cog_path

    indices = load_scene_indices(cog_paths)
    assert float(indices["ndvi"].mean()) == pytest.approx(0.6666666667, rel=1e-3)
    assert float(indices["nbr"].mean()) == pytest.approx(0.4285714286, rel=1e-3)
    assert float(indices["ndwi"].mean()) == pytest.approx(-0.25, rel=1e-3)


def test_load_scene_indices_missing_band_raises(tmp_path: Path) -> None:
    cog_paths = {
        "B4": local_band_path(tmp_path, SCENE_ID, "B4"),
        "B5": local_band_path(tmp_path, SCENE_ID, "B5"),
    }
    for band in ("B4", "B5"):
        write_scaled_sr_cog(cog_paths[band], 0.1)

    with pytest.raises(ValueError, match="Missing required bands"):
        load_scene_indices(cog_paths)


def test_feature_bands_constant() -> None:
    assert FEATURE_BANDS == ["B3", "B4", "B5", "B7"]
