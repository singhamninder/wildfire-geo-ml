"""Load and validate ingest settings from pipeline YAML."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class IngestConfig(BaseModel):
    """Landsat download settings from ``config/pipeline.yaml``."""

    bucket: str = "usgs-landsat"
    collection_prefix: str = "collection02/level-2/standard/oli-tirs"
    bands: list[str] = Field(default_factory=lambda: ["B2", "B3", "B4", "B5", "B6", "B7"])
    wrs_path: str = "044"
    scenes: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

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
        If YAML is missing the ``ingest`` section or fails validation.
    """
    if not config_path.is_file():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "ingest" not in raw:
        msg = f"Config file {config_path} must contain an 'ingest' section"
        raise ValueError(msg)

    return PipelineConfig.model_validate(raw)
