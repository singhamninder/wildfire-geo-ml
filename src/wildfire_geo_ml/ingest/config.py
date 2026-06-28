"""Load and validate ingest settings from pipeline YAML."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class StudyAreaConfig(BaseModel):
    """Geographic and temporal extent for scene discovery and analysis."""

    bbox: list[float] = Field(
        description="Bounding box as [west, south, east, north] in EPSG:4326 degrees."
    )
    datetime: str = Field(
        description="ISO8601 interval for STAC search, e.g. '2024-07-01/2024-08-31'."
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        """
        Ensure bbox has exactly four WGS84 coordinates.

        Parameters
        ----------
        value : list[float]
            Bounding box as [west, south, east, north].

        Returns
        -------
        list[float]
            Validated bbox unchanged.

        Raises
        ------
        ValueError
            If the list length is not four.
        """
        if len(value) != 4:
            msg = f"bbox must have exactly 4 values [west, south, east, north], got {len(value)}"
            raise ValueError(msg)
        return value


class DiscoverConfig(BaseModel):
    """LandsatLook STAC API settings for AOI scene discovery."""

    stac_api_url: str = "https://landsatlook.usgs.gov/stac-server"
    collection: str = "landsat-c2l2-sr"
    platform: str = "LANDSAT_9"
    max_cloud_cover: float = Field(
        default=10.0,
        description="Maximum eo:cloud_cover (percent); scenes must be strictly below this value.",
    )

    @field_validator("max_cloud_cover")
    @classmethod
    def validate_cloud_cover(cls, value: float) -> float:
        """
        Ensure cloud cover threshold is a valid percentage.

        Parameters
        ----------
        value : float
            Maximum eo:cloud_cover percent for STAC discovery.

        Returns
        -------
        float
            Validated threshold unchanged.

        Raises
        ------
        ValueError
            If value is outside [0, 100].
        """
        if not 0 <= value <= 100:
            msg = f"max_cloud_cover must be between 0 and 100, got {value}"
            raise ValueError(msg)
        return value


class IngestConfig(BaseModel):
    """Landsat download settings from ``config/pipeline.yaml``."""

    bucket: str = "usgs-landsat"
    collection_prefix: str = "collection02/level-2/standard/oli-tirs"
    bands: list[str] = Field(default_factory=lambda: ["B2", "B3", "B4", "B5", "B6", "B7"])
    wrs_path: str = "044"


class FeaturesConfig(BaseModel):
    """H3 feature engineering settings from ``config/pipeline.yaml``."""

    h3_resolution: int = Field(
        default=8,
        description="H3 resolution for zonal stats (~461 m edge at res 8).",
    )
    stat_names: list[str] = Field(
        default_factory=lambda: ["mean", "std"],
        description="Zonal statistics to compute per H3 cell.",
    )
    required_bands: list[str] = Field(
        default_factory=lambda: ["B3", "B4", "B5", "B7"],
        description="Bands required for NDVI, NBR, and NDWI.",
    )
    output_dir: str = Field(
        default="data/features/h3_partitioned",
        description="Directory for H3-partitioned GeoParquet output.",
    )
    indices_dir: str = Field(
        default="data/indices",
        description="Directory for per-scene NDVI/NBR/NDWI index COGs consumed by Sedona.",
    )

    @field_validator("h3_resolution")
    @classmethod
    def validate_h3_resolution(cls, value: int) -> int:
        """
        Ensure H3 resolution is within the valid index range.

        Parameters
        ----------
        value : int
            H3 resolution level (0–15).

        Returns
        -------
        int
            Validated resolution unchanged.

        Raises
        ------
        ValueError
            If value is outside [0, 15].
        """
        if not 0 <= value <= 15:
            msg = f"h3_resolution must be between 0 and 15, got {value}"
            raise ValueError(msg)
        return value


class SedonaConfig(BaseModel):
    """Apache Sedona / Spark settings from ``config/pipeline.yaml``."""

    app_name: str = "wildfire-veg-risk"
    jar_packages: str = (
        "org.apache.sedona:sedona-spark-4.0_2.13:1.9.0,org.datasyslab:geotools-wrapper:1.9.0-33.5"
    )
    shuffle_partitions: int = Field(
        default=8,
        description="spark.sql.shuffle.partitions for local laptop runs.",
    )
    corridor_output_dir: str = Field(
        default="data/features/corridor_partitioned",
        description="GeoParquet output for corridor zonal statistics.",
    )
    hex_output_dir: str = Field(
        default="data/features/sedona_hex_partitioned",
        description="Optional Sedona hex zonal statistics output (EMR demo).",
    )


class CorridorsConfig(BaseModel):
    """Transmission-line corridor settings from ``config/pipeline.yaml``."""

    rest_service_url: str = Field(
        description="ArcGIS REST query endpoint for CEC transmission lines.",
    )
    cached_geojson: str = Field(
        default="data/raw/cec_transmission_lines.geojson",
        description="Local cache path for AOI-filtered line GeoJSON.",
    )
    buffer_m: float = Field(
        default=100.0,
        description="Corridor buffer distance in meters (metric CRS).",
    )
    metric_crs: str = Field(
        default="EPSG:32610",
        description="Metric CRS for buffering before zonal stats.",
    )


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    study_area: StudyAreaConfig
    discover: DiscoverConfig
    ingest: IngestConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    sedona: SedonaConfig = Field(default_factory=SedonaConfig)
    corridors: CorridorsConfig | None = None


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    """
    Load pipeline YAML and return a validated config object.

    Parameters
    ----------
    config_path : Path
        Path to ``pipeline.yaml``.

    Returns
    -------
    PipelineConfig
        Validated configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If YAML is missing required sections or fails validation.
    """
    if not config_path.is_file():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    # Parse YAML and validate required top-level sections before Pydantic coercion.
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Config file {config_path} must be a YAML mapping"
        raise ValueError(msg)

    required = ("study_area", "discover", "ingest")
    missing = [key for key in required if key not in raw]
    if missing:
        msg = f"Config file {config_path} missing required sections: {', '.join(missing)}"
        raise ValueError(msg)

    return PipelineConfig.model_validate(raw)
