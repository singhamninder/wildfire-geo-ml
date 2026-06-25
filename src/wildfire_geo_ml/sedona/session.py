"""
SedonaContext and SparkSession initialization.

SedonaContext registers RS_* SQL functions and spatial UDTs into Spark. Call
``create_sedona_session`` once per job before any spatial or raster SQL.
"""

from __future__ import annotations

import os
from typing import Any

from pyspark.sql import SparkSession
from sedona.spark import SedonaContext

SEDONA_VERSION = "1.9.0"
SEDONA_SPARK_ARTIFACT = f"org.apache.sedona:sedona-spark-4.0_2.13:{SEDONA_VERSION}"
GEOTOOLS_WRAPPER = "org.datasyslab:geotools-wrapper:1.9.0-33.5"
DEFAULT_JAR_PACKAGES = f"{SEDONA_SPARK_ARTIFACT},{GEOTOOLS_WRAPPER}"

DEFAULT_SHUFFLE_PARTITIONS = "8"
DEFAULT_DRIVER_MEMORY = "8g"


def create_sedona_session(
    app_name: str = "wildfire-veg-risk",
    master: str | None = None,
    extra_config: dict[str, str] | None = None,
    *,
    jar_packages: str = DEFAULT_JAR_PACKAGES,
    shuffle_partitions: str = DEFAULT_SHUFFLE_PARTITIONS,
) -> SparkSession:
    """
    Create a SparkSession with Sedona registered.

    Parameters
    ----------
    app_name : str
        Spark application name for the UI.
    master : str, optional
        Spark master URL. Defaults to ``SPARK_MASTER`` env or ``local[*]``.
        Use ``yarn`` or EMR Serverless application endpoint in production.
    extra_config : dict[str, str], optional
        Additional Spark config key/value pairs.
    jar_packages : str
        Maven coordinates for Sedona and geotools-wrapper JARs.
    shuffle_partitions : str
        Value for ``spark.sql.shuffle.partitions`` (keep modest on laptops).

    Returns
    -------
    SparkSession
        Sedona-enabled SparkSession with RS_* functions registered.

    Notes
    -----
    On EMR Serverless, set ``SPARK_MASTER`` to the cluster endpoint and pass
    Sedona JARs via ``--jars`` in ``submit_job_run`` if not using packages.

    Requires Java 17+ on the driver (PySpark 4.1). Set ``JAVA_HOME`` accordingly
    for local runs.
    """
    resolved_master = master or os.environ.get("SPARK_MASTER", "local[*]")
    builder = (
        SparkSession.builder.master(resolved_master)
        .appName(app_name)
        .config("spark.jars.packages", jar_packages)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.driver.memory", DEFAULT_DRIVER_MEMORY)
    )
    if extra_config:
        for key, value in extra_config.items():
            builder = builder.config(key, value)

    spark = builder.getOrCreate()
    sedona: Any = SedonaContext.create(spark)
    return sedona
