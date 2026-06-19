"""Tests for writing per-scene index COGs."""

from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.transform import from_origin

from wildfire_geo_ml.features.indices import write_scene_index_cogs


def test_write_scene_index_cogs_writes_float_geotiff(tmp_path: Path) -> None:
    values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    transform = from_origin(500000.0, 4500000.0, 30.0, 30.0)
    da = xr.DataArray(values, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:32610")
    da = da.rio.write_transform(transform)

    paths = write_scene_index_cogs({"ndvi": da}, tmp_path, "TEST_SCENE")
    assert paths["ndvi"].is_file()
    with rasterio.open(paths["ndvi"]) as src:
        assert src.dtypes[0] == "float32"
        assert src.count == 1
        assert src.crs is not None
