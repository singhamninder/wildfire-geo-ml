"""Tests for Sedona session initialization."""

import pytest

from wildfire_geo_ml.sedona.session import (
    DEFAULT_JAR_PACKAGES,
    SEDONA_VERSION,
    create_sedona_session,
)
from wildfire_geo_ml.sedona.zonal_stats_job import stop_spark


@pytest.mark.slow
def test_create_sedona_session_registers_raster_sql(java17_home: str) -> None:
    spark = create_sedona_session(app_name="wildfire-sedona-test")
    try:
        assert SEDONA_VERSION == "1.9.0"
        assert "sedona-spark-4.0_2.13" in DEFAULT_JAR_PACKAGES
        functions = spark.sql("SHOW FUNCTIONS LIKE 'rs_zonalstats'").collect()
        assert len(functions) >= 1
    finally:
        stop_spark(spark)
