"""
Download Landsat-9 Collection 2 SR scenes from AWS Open Data (``s3://usgs-landsat``).

WHY: Wildfire mitigation pipelines need reproducible ingestion from the same USGS bucket
that STAC asset hrefs will reference — no manual EarthExplorer steps.

Notes
-----
The ``usgs-landsat`` bucket is **Requester Pays**; configure ``AWS_PROFILE`` (or
equivalent) before running. Data-transfer charges apply to your AWS account.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import boto3
import typer
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from tqdm import tqdm

from wildfire_geo_ml.ingest.config import IngestConfig, load_pipeline_config
from wildfire_geo_ml.ingest.discover_scenes import DiscoveredScene, resolve_scene_list
from wildfire_geo_ml.ingest.landsat_paths import (
    local_band_path,
    local_mtl_path,
    mtl_key,
    sr_band_key,
)

logger = logging.getLogger(__name__)


class S3DownloadClient(Protocol):
    """Minimal S3 client surface used for Landsat downloads."""

    def download_file(
        self,
        bucket: str,
        key: str,
        filename: str,
        ExtraArgs: dict[str, str] | None = None,  # noqa: N803 — boto3 API name
    ) -> None:
        """
        Download an S3 object to a local file path.

        Parameters
        ----------
        bucket : str
            S3 bucket name.
        key : str
            Object key.
        filename : str
            Local destination path.
        ExtraArgs : dict[str, str], optional
            Extra arguments (e.g. ``RequestPayer`` for Requester Pays buckets).
        """
        ...


DEFAULT_CONFIG = Path("config/pipeline.yaml")
DEFAULT_OUTPUT_DIR = Path("data/raw")

# usgs-landsat is a Requester Pays bucket (anonymous access denied).
REQUESTER_PAYS_ARGS = {"RequestPayer": "requester"}


@dataclass
class FileResult:
    """Outcome for a single downloaded object."""

    scene_id: str
    label: str
    status: str  # downloaded | skipped | failed
    path: Path | None = None
    error: str | None = None


@dataclass
class DownloadReport:
    """Aggregated results for one scene download."""

    scene_id: str
    results: list[FileResult] = field(default_factory=list)

    @property
    def failed(self) -> list[FileResult]:
        """File results with status ``failed``."""
        return [r for r in self.results if r.status == "failed"]

    @property
    def downloaded(self) -> list[FileResult]:
        """File results with status ``downloaded``."""
        return [r for r in self.results if r.status == "downloaded"]

    @property
    def skipped(self) -> list[FileResult]:
        """File results with status ``skipped`` (already present locally)."""
        return [r for r in self.results if r.status == "skipped"]


def get_landsat_s3_client() -> S3DownloadClient:
    """
    Create an S3 client for ``s3://usgs-landsat``.

    Uses the default AWS credential chain. The USGS bucket is **Requester Pays**;
    downloads must include ``RequestPayer=requester`` (see ``REQUESTER_PAYS_ARGS``).

    Returns
    -------
    S3DownloadClient
        Boto3 S3 client for us-west-2.

    Raises
    ------
    NoCredentialsError
        If no AWS credentials are configured (profile, env vars, or IAM role).

    Notes
    -----
    Requires: AWS credentials; S3 data-transfer charges apply to your account.
    """
    try:
        return cast(S3DownloadClient, boto3.client("s3", region_name="us-west-2"))
    except NoCredentialsError:
        logger.exception(
            "AWS credentials required for s3://usgs-landsat (Requester Pays bucket). "
            "Set AWS_PROFILE or run `aws configure`."
        )
        raise


def download_file(
    s3: S3DownloadClient,
    bucket: str,
    key: str,
    dest: Path,
    force: bool = False,
) -> str:
    """
    Download one S3 object to a local path.

    Parameters
    ----------
    s3 : S3DownloadClient
        Boto3 S3 client configured for Requester Pays downloads.
    bucket : str
        S3 bucket name.
    key : str
        Object key.
    dest : Path
        Local destination file path.
    force : bool
        If True, re-download even when the file already exists.

    Returns
    -------
    str
        ``downloaded``, ``skipped``, or ``failed``.
    """
    if dest.exists() and dest.stat().st_size > 0 and not force:
        # Skip re-download when a non-empty local copy already exists.
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Requester Pays is required for the usgs-landsat open-data bucket.
        s3.download_file(
            bucket,
            key,
            str(dest),
            ExtraArgs=REQUESTER_PAYS_ARGS,
        )
    except ClientError as exc:
        logger.error("Failed s3://%s/%s -> %s: %s", bucket, key, dest, exc)
        return "failed"
    return "downloaded"


def download_scene(
    s3: S3DownloadClient,
    config: IngestConfig,
    scene_id: str,
    output_dir: Path,
    bands: list[str] | None = None,
    force: bool = False,
) -> DownloadReport:
    """
    Download SR bands and MTL JSON for one Landsat scene.

    Parameters
    ----------
    s3 : S3DownloadClient
        Boto3 S3 client configured for Requester Pays downloads.
    config : IngestConfig
        Ingest settings (bucket, collection prefix, default bands).
    scene_id : str
        Landsat scene ID.
    output_dir : Path
        Root directory for per-scene subfolders.
    bands : list[str], optional
        Bands to download; defaults to ``config.bands``.
    force : bool
        Re-download existing files when True.

    Returns
    -------
    DownloadReport
        Per-file download outcomes.
    """
    band_list = bands if bands is not None else config.bands
    report = DownloadReport(scene_id=scene_id)

    # Download each SR band GeoTIFF, then the MTL JSON metadata sidecar.
    for band in band_list:
        key = sr_band_key(scene_id, band, config.collection_prefix)
        dest = local_band_path(output_dir, scene_id, band)
        status = download_file(s3, config.bucket, key, dest, force=force)
        report.results.append(FileResult(scene_id=scene_id, label=band, status=status, path=dest))

    mtl_dest = local_mtl_path(output_dir, scene_id)
    mtl_s3_key = mtl_key(scene_id, config.collection_prefix)
    mtl_status = download_file(s3, config.bucket, mtl_s3_key, mtl_dest, force=force)
    report.results.append(
        FileResult(scene_id=scene_id, label="MTL", status=mtl_status, path=mtl_dest)
    )

    return report


def run_download(
    config: IngestConfig,
    output_dir: Path,
    scene_ids: list[str],
    bands: list[str] | None = None,
    force: bool = False,
) -> list[DownloadReport]:
    """
    Download all requested scenes and return per-scene reports.

    Parameters
    ----------
    config : IngestConfig
        Ingest settings.
    output_dir : Path
        Root directory for downloads.
    scene_ids : list[str]
        Scene IDs to download.
    bands : list[str], optional
        Band override.
    force : bool
        Re-download existing files.

    Returns
    -------
    list[DownloadReport]
        One report per scene.
    """
    if not scene_ids:
        logger.warning("No scenes matched the filters; nothing to download.")
        return []

    s3 = get_landsat_s3_client()
    reports: list[DownloadReport] = []

    # Download scenes sequentially with per-file progress logging.
    for scene_id in tqdm(scene_ids, desc="Scenes", unit="scene"):
        logger.info("Downloading scene %s", scene_id)
        report = download_scene(s3, config, scene_id, output_dir, bands=bands, force=force)
        reports.append(report)
        for result in report.results:
            logger.info("  %s %s: %s", result.label, result.status, result.path)

    return reports


def _print_discovery_table(scenes: list[DiscoveredScene]) -> None:
    """Print discovered scenes with cloud cover and WRS path/row."""
    typer.echo("\nDiscovered scenes:")
    typer.echo(f"{'Scene ID':<45} {'Cloud%':<8} {'Path/Row':<10} {'Date':<10}")
    typer.echo("-" * 75)
    for scene in scenes:
        path_row = f"{scene.wrs_path}/{scene.wrs_row}"
        cloud = f"{scene.cloud_cover:.1f}" if scene.cloud_cover > 0 else "n/a"
        typer.echo(f"{scene.scene_id:<45} {cloud:<8} {path_row:<10} {scene.acquisition_date:<10}")


def _print_summary(reports: list[DownloadReport]) -> None:
    """Print a human-readable summary table to stdout."""
    typer.echo("\nDownload summary:")
    typer.echo(f"{'Scene':<45} {'File':<6} {'Status':<10}")
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
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        file_okay=False,
        help="Root directory for per-scene downloads",
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
        help="Re-download files that already exist",
    ),
    discover_only: bool = typer.Option(
        False,
        "--discover-only",
        help="Discover scenes over AOI and print table; do not download",
    ),
    max_cloud_cover: float | None = typer.Option(
        None,
        "--max-cloud-cover",
        help="Override discover.max_cloud_cover (percent; scenes must be below this)",
    ),
) -> None:
    """
    Download Landsat-9 SR bands and MTL JSON from s3://usgs-landsat.

    Discovers scenes over the study-area AOI via LandsatLook STAC
    (bbox + datetime + cloud cover from config/pipeline.yaml). Use
    ``--path``/``--rows``/``--dates`` to narrow the download set.

    Requires AWS credentials (Requester Pays bucket); set AWS_PROFILE in .env or shell.

    Notes
    -----
    Exits with code 1 when STAC discovery fails, no scenes match filters,
    or any individual file download fails.
    """
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load config and resolve CLI filter overrides.
    pipeline = load_pipeline_config(config_path)
    ingest = pipeline.ingest

    path_filter = wrs_path if wrs_path is not None else ingest.wrs_path
    row_filter = list(rows) if rows else None
    date_filter = list(dates) if dates else None

    cloud_threshold = (
        max_cloud_cover if max_cloud_cover is not None else pipeline.discover.max_cloud_cover
    )
    logger.info(
        "Scene selection: bbox=%s datetime=%s cloud_cover < %.1f%%",
        pipeline.study_area.bbox,
        pipeline.study_area.datetime,
        cloud_threshold,
    )

    try:
        discovered = resolve_scene_list(
            pipeline,
            max_cloud_cover=max_cloud_cover,
            wrs_path=path_filter,
            rows=row_filter,
            dates=date_filter,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if discover_only:
        _print_discovery_table(discovered)
        typer.echo(f"\n{len(discovered)} scene(s) matched filters.")
        return

    if not discovered:
        typer.echo("No scenes matched the filters; nothing to download.", err=True)
        raise typer.Exit(1)

    for scene in discovered:
        logger.info(
            "Queued %s (cloud=%.1f%%, path/row=%s/%s)",
            scene.scene_id,
            scene.cloud_cover,
            scene.wrs_path,
            scene.wrs_row,
        )

    band_list: list[str] | None = None
    if bands:
        band_list = [b.strip().upper() for b in bands.split(",") if b.strip()]

    scene_ids = [scene.scene_id for scene in discovered]
    reports = run_download(
        ingest,
        output_dir,
        scene_ids,
        bands=band_list,
        force=force,
    )
    _print_summary(reports)

    total_failed = sum(len(r.failed) for r in reports)
    if total_failed:
        typer.echo(f"\n{total_failed} file(s) failed.", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
