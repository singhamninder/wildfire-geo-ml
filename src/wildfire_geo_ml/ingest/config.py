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


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    study_area: StudyAreaConfig
    discover: DiscoverConfig
    ingest: IngestConfig


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
