"""
Validate-first COG ingestion for Landsat-9 SR bands.

WHY: USGS Collection 2 already ships as COG — validate at the ingest boundary and
only re-profile non-conforming rasters. Guaranteed-valid COGs in ``data/cog/`` feed
the STAC catalog and Sedona zonal stats in later phases.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import typer
from tqdm import tqdm

from wildfire_geo_ml.ingest.config import load_pipeline_config
from wildfire_geo_ml.ingest.landsat_paths import (
    discover_scenes_on_disk,
    filter_scenes,
    local_band_path,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config/pipeline.yaml")
DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/cog")

COG_PROFILE = "deflate"
OVERVIEW_LEVELS = 6
OVERVIEW_RESAMPLING = "average"

# rio-cogeo validate stdout markers for quick pass/fail parsing.
VALID_COG_MARKER = "is a valid cloud optimized GeoTIFF"
INVALID_COG_MARKER = "is NOT"


@dataclass
class CogFileResult:
    """Outcome for a single band COG ensure operation."""

    scene_id: str
    label: str
    status: str  # validated | converted | skipped | failed
    path: Path | None = None
    error: str | None = None


@dataclass
class CogSceneReport:
    """Aggregated COG results for one scene."""

    scene_id: str
    results: list[CogFileResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CogFileResult]:
        """Band results with status ``failed``."""
        return [r for r in self.results if r.status == "failed"]


def is_valid_cog(path: Path) -> bool:
    """
    Return True if ``path`` passes ``rio cogeo validate``.

    Parameters
    ----------
    path : Path
        GeoTIFF to validate.

    Returns
    -------
    bool
        True when rio-cogeo reports a valid COG.
    """
    if not path.is_file():
        return False

    result = subprocess.run(
        ["rio", "cogeo", "validate", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    # Parse rio-cogeo validate text output rather than relying on exit code alone.
    if INVALID_COG_MARKER in output:
        return False
    return VALID_COG_MARKER in output or result.returncode == 0


def _validate_cog(cog_path: Path) -> None:
    """
    Run ``rio cogeo validate`` and raise on failure.

    Parameters
    ----------
    cog_path : Path
        COG file to validate.

    Raises
    ------
    ValueError
        If validation reports the file is not a valid COG.
    subprocess.CalledProcessError
        If the rio-cogeo CLI exits with an error.
    """
    result = subprocess.run(
        ["rio", "cogeo", "validate", str(cog_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    if INVALID_COG_MARKER in result.stdout:
        msg = f"COG validation failed for {cog_path}:\n{result.stdout}"
        raise ValueError(msg)


def convert_to_cog(input_path: Path, output_path: Path) -> Path:
    """
    Convert a GeoTIFF to Cloud-Optimized GeoTIFF using rio-cogeo.

    WHY: COG internal tiling (512×512 blocks) + overview pyramids allow
    Sedona's RS_FromGeoTiff to fetch only the spatial subset needed over S3,
    avoiding full-scene downloads for corridor-level analysis.

    Parameters
    ----------
    input_path : Path
        Source GeoTIFF (Landsat-9 SR band file).
    output_path : Path
        Destination path for the COG output.

    Returns
    -------
    Path
        Path to the validated COG file.

    Raises
    ------
    subprocess.CalledProcessError
        If rio-cogeo conversion or validation fails.
    ValueError
        If post-conversion validation fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rio",
            "cogeo",
            "create",
            str(input_path),
            str(output_path),
            "--cog-profile",
            COG_PROFILE,
            "--overview-level",
            str(OVERVIEW_LEVELS),
            "--overview-resampling",
            OVERVIEW_RESAMPLING,
        ],
        check=True,
    )
    _validate_cog(output_path)
    return output_path


def ensure_cog(input_path: Path, output_path: Path, *, force: bool = False) -> str:
    """
    Validate, copy, or convert a raster to a guaranteed-valid COG.

    Parameters
    ----------
    input_path : Path
        Source GeoTIFF under ``data/raw/``.
    output_path : Path
        Destination COG path under ``data/cog/``.
    force : bool
        Re-process even when a valid output already exists.

    Returns
    -------
    str
        ``validated``, ``converted``, ``skipped``, or ``failed``.
    """
    if not input_path.is_file():
        logger.error("Input file missing: %s", input_path)
        return "failed"

    try:
        # Validate-first: skip when output already valid; copy when input is valid COG.
        if output_path.is_file() and is_valid_cog(output_path) and not force:
            return "skipped"

        if is_valid_cog(input_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return "validated"

        # Re-profile non-conforming rasters with rio-cogeo deflate + overviews.
        convert_to_cog(input_path, output_path)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        logger.error("COG ensure failed %s -> %s: %s", input_path, output_path, exc)
        return "failed"

    return "converted"


def resolve_input_scenes(
    input_dir: Path,
    wrs_path: str | None = None,
    rows: list[str] | None = None,
    dates: list[str] | None = None,
) -> list[str]:
    """
    List scene IDs with SR bands on disk, optionally filtered by WRS path/row/date.

    Parameters
    ----------
    input_dir : Path
        Raw download root (e.g. ``data/raw``).
    wrs_path : str, optional
        WRS-2 path filter.
    rows : list[str], optional
        WRS-2 row filter.
    dates : list[str], optional
        Acquisition date filter (YYYYMMDD).

    Returns
    -------
    list[str]
        Scene IDs to process.
    """
    return filter_scenes(
        discover_scenes_on_disk(input_dir),
        wrs_path=wrs_path,
        rows=rows,
        dates=dates,
    )


def process_scene(
    scene_id: str,
    input_dir: Path,
    output_dir: Path,
    bands: list[str],
    force: bool = False,
) -> CogSceneReport:
    """
    Ensure COGs for all requested bands in one scene.

    Parameters
    ----------
    scene_id : str
        Landsat scene ID.
    input_dir : Path
        Raw download root.
    output_dir : Path
        COG output root.
    bands : list[str]
        Band names (e.g. ``B4``).
    force : bool
        Re-process existing valid outputs when True.

    Returns
    -------
    CogSceneReport
        Per-band COG outcomes.
    """
    report = CogSceneReport(scene_id=scene_id)
    # Validate-first COG ensure for each SR band in the scene.
    for band in bands:
        input_path = local_band_path(input_dir, scene_id, band)
        output_path = local_band_path(output_dir, scene_id, band)
        status = ensure_cog(input_path, output_path, force=force)
        report.results.append(
            CogFileResult(scene_id=scene_id, label=band, status=status, path=output_path)
        )
    return report


def run_cog_convert(
    input_dir: Path,
    output_dir: Path,
    scene_ids: list[str],
    bands: list[str],
    force: bool = False,
) -> list[CogSceneReport]:
    """
    Process all scenes and return per-scene COG reports.

    Parameters
    ----------
    input_dir : Path
        Raw download root.
    output_dir : Path
        COG output root.
    scene_ids : list[str]
        Scenes to process.
    bands : list[str]
        Bands to convert.
    force : bool
        Re-process existing valid outputs.

    Returns
    -------
    list[CogSceneReport]
        One report per scene.
    """
    if not scene_ids:
        logger.warning("No scenes matched the filters; nothing to convert.")
        return []

    reports: list[CogSceneReport] = []
    for scene_id in tqdm(scene_ids, desc="Scenes", unit="scene"):
        logger.info("Processing scene %s", scene_id)
        report = process_scene(scene_id, input_dir, output_dir, bands, force=force)
        reports.append(report)
        for result in report.results:
            logger.info("  %s %s: %s", result.label, result.status, result.path)

    return reports


def _print_summary(reports: list[CogSceneReport]) -> None:
    """Print a human-readable summary table to stdout."""
    typer.echo("\nCOG summary:")
    typer.echo(f"{'Scene':<45} {'Band':<6} {'Status':<10}")
    typer.echo("-" * 65)
    for report in reports:
        for result in report.results:
            typer.echo(f"{result.scene_id:<45} {result.label:<6} {result.status:<10}")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        exists=True,
        dir_okay=False,
        help="Path to pipeline.yaml",
    ),
    input_dir: Path = typer.Option(
        DEFAULT_INPUT_DIR,
        "--input-dir",
        file_okay=False,
        help="Root directory for per-scene raw downloads",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        file_okay=False,
        help="Root directory for validated COG outputs",
    ),
    wrs_path: str | None = typer.Option(
        None,
        "--path",
        help="Filter by WRS-2 path (e.g. 044); defaults to config wrs_path",
    ),
    rows: list[str] = typer.Option(
        [],
        "--rows",
        help="Filter by WRS-2 row(s), e.g. --rows 032 --rows 033",
    ),
    dates: list[str] = typer.Option(
        [],
        "--dates",
        help="Filter by acquisition date YYYYMMDD",
    ),
    bands: str | None = typer.Option(
        None,
        "--bands",
        help="Comma-separated bands (e.g. B4,B5); defaults to config",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-process files that already validate as COG",
    ),
) -> None:
    """
    Validate-first COG ingestion for Landsat-9 SR bands.

    Processes all scenes with SR band GeoTIFFs under ``data/raw/`` (downloaded
    artifacts only). USGS Collection 2 bands are usually already COG — this step
    validates each band and copies it to ``data/cog/``. Non-conforming rasters
    are re-profiled with rio-cogeo (deflate, 6 overview levels).

    Notes
    -----
    Exits with code 1 when no scenes match filters or any band COG ensure fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = load_pipeline_config(config_path)
    ingest = pipeline.ingest

    path_filter = wrs_path if wrs_path is not None else ingest.wrs_path
    row_filter = list(rows) if rows else None
    date_filter = list(dates) if dates else None

    # Resolve scenes from on-disk raw downloads matching WRS/date filters.
    scene_ids = resolve_input_scenes(
        input_dir,
        wrs_path=path_filter,
        rows=row_filter,
        dates=date_filter,
    )

    if not scene_ids:
        typer.echo(
            f"No scenes with SR bands found under {input_dir} matching filters.",
            err=True,
        )
        raise typer.Exit(1)

    band_list = ingest.bands
    if bands:
        band_list = [b.strip().upper() for b in bands.split(",") if b.strip()]

    reports = run_cog_convert(
        input_dir,
        output_dir,
        scene_ids,
        band_list,
        force=force,
    )
    _print_summary(reports)

    total_failed = sum(len(r.failed) for r in reports)
    if total_failed:
        typer.echo(f"\n{total_failed} file(s) failed.", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
