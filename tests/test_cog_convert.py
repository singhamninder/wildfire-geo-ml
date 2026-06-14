"""Tests for validate-first COG conversion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tests.conftest import BANDS, SCENE_ID
from wildfire_geo_ml.ingest.cog_convert import (
    discover_scenes_on_disk,
    ensure_cog,
    is_valid_cog,
    main,
    process_scene,
    resolve_input_scenes,
)

VALID_OUTPUT = f"{SCENE_ID}_SR_B4.TIF is a valid cloud optimized GeoTIFF\n"
INVALID_OUTPUT = f"{SCENE_ID}_SR_B4.TIF is NOT a valid cloud optimized GeoTIFF\n"


def _completed_process(stdout: str = "", returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = ""
    result.returncode = returncode
    return result


@patch("wildfire_geo_ml.ingest.cog_convert.subprocess.run")
def test_is_valid_cog_true(mock_run: MagicMock, tmp_path: Path) -> None:
    tif = tmp_path / "band.tif"
    tif.write_bytes(b"fake")
    mock_run.return_value = _completed_process(stdout=VALID_OUTPUT)

    assert is_valid_cog(tif) is True
    mock_run.assert_called_once()


@patch("wildfire_geo_ml.ingest.cog_convert.subprocess.run")
def test_is_valid_cog_false(mock_run: MagicMock, tmp_path: Path) -> None:
    tif = tmp_path / "band.tif"
    tif.write_bytes(b"fake")
    mock_run.return_value = _completed_process(stdout=INVALID_OUTPUT, returncode=1)

    assert is_valid_cog(tif) is False


def test_is_valid_cog_missing_file(tmp_path: Path) -> None:
    assert is_valid_cog(tmp_path / "missing.tif") is False


@patch("wildfire_geo_ml.ingest.cog_convert.shutil.copy2")
@patch("wildfire_geo_ml.ingest.cog_convert.is_valid_cog")
def test_ensure_cog_validates_and_copies(
    mock_valid: MagicMock,
    mock_copy: MagicMock,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw" / "band.tif"
    output_path = tmp_path / "cog" / "band.tif"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"raw")
    mock_valid.side_effect = lambda path: path == input_path

    status = ensure_cog(input_path, output_path)

    assert status == "validated"
    mock_copy.assert_called_once_with(input_path, output_path)


@patch("wildfire_geo_ml.ingest.cog_convert.convert_to_cog")
@patch("wildfire_geo_ml.ingest.cog_convert.is_valid_cog")
def test_ensure_cog_converts_invalid_input(
    mock_valid: MagicMock,
    mock_convert: MagicMock,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw" / "band.tif"
    output_path = tmp_path / "cog" / "band.tif"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"raw")
    mock_valid.return_value = False

    status = ensure_cog(input_path, output_path)

    assert status == "converted"
    mock_convert.assert_called_once_with(input_path, output_path)


@patch("wildfire_geo_ml.ingest.cog_convert.is_valid_cog")
def test_ensure_cog_skips_existing_valid_output(
    mock_valid: MagicMock,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw" / "band.tif"
    output_path = tmp_path / "cog" / "band.tif"
    input_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"raw")
    output_path.write_bytes(b"cog")
    mock_valid.side_effect = lambda path: path == output_path

    status = ensure_cog(input_path, output_path)

    assert status == "skipped"


@patch("wildfire_geo_ml.ingest.cog_convert.is_valid_cog")
def test_ensure_cog_force_reprocesses_valid_output(
    mock_valid: MagicMock,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "raw" / "band.tif"
    output_path = tmp_path / "cog" / "band.tif"
    input_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"raw")
    output_path.write_bytes(b"cog")
    mock_valid.return_value = True

    with patch("wildfire_geo_ml.ingest.cog_convert.shutil.copy2") as mock_copy:
        status = ensure_cog(input_path, output_path, force=True)

    assert status == "validated"
    mock_copy.assert_called_once_with(input_path, output_path)


def test_ensure_cog_missing_input(tmp_path: Path) -> None:
    status = ensure_cog(tmp_path / "missing.tif", tmp_path / "out.tif")
    assert status == "failed"


def test_discover_scenes_on_disk(tmp_path: Path) -> None:
    scene_dir = tmp_path / SCENE_ID
    scene_dir.mkdir()
    (scene_dir / f"{SCENE_ID}_SR_B4.TIF").write_bytes(b"x")
    (tmp_path / "empty").mkdir()

    assert discover_scenes_on_disk(tmp_path) == [SCENE_ID]


def test_resolve_input_scenes_uses_on_disk_only(tmp_path: Path) -> None:
    scene_a = tmp_path / SCENE_ID
    scene_a.mkdir()
    (scene_a / f"{SCENE_ID}_SR_B4.TIF").write_bytes(b"x")

    other_id = "LC09_L2SP_044033_20240715_20240717_02_T1"
    scene_b = tmp_path / other_id
    scene_b.mkdir()
    (scene_b / f"{other_id}_SR_B4.TIF").write_bytes(b"x")
    (tmp_path / "empty").mkdir()

    scenes = resolve_input_scenes(tmp_path)
    assert scenes == [SCENE_ID, other_id]


@patch("wildfire_geo_ml.ingest.cog_convert.ensure_cog")
def test_process_scene_all_bands(mock_ensure: MagicMock, tmp_path: Path) -> None:
    mock_ensure.return_value = "validated"
    raw_dir = tmp_path / "raw"
    cog_dir = tmp_path / "cog"
    scene_dir = raw_dir / SCENE_ID
    scene_dir.mkdir(parents=True)
    for band in BANDS:
        (scene_dir / f"{SCENE_ID}_SR_{band}.TIF").write_bytes(b"x")

    report = process_scene(SCENE_ID, raw_dir, cog_dir, BANDS)

    assert len(report.results) == len(BANDS)
    assert all(r.status == "validated" for r in report.results)
    assert mock_ensure.call_count == len(BANDS)


@patch("wildfire_geo_ml.ingest.cog_convert.run_cog_convert")
def test_cli_smoke(
    mock_run: MagicMock,
    tmp_path: Path,
    pipeline_config_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    cog_dir = tmp_path / "cog"
    scene_dir = raw_dir / SCENE_ID
    scene_dir.mkdir(parents=True)
    (scene_dir / f"{SCENE_ID}_SR_B4.TIF").write_bytes(b"x")

    from wildfire_geo_ml.ingest.cog_convert import CogFileResult, CogSceneReport

    mock_run.return_value = [
        CogSceneReport(
            scene_id=SCENE_ID,
            results=[CogFileResult(scene_id=SCENE_ID, label="B4", status="validated")],
        )
    ]

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(pipeline_config_path),
            "--input-dir",
            str(raw_dir),
            "--output-dir",
            str(cog_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "COG summary" in result.output
    mock_run.assert_called_once()


@patch("wildfire_geo_ml.ingest.cog_convert.run_cog_convert")
def test_cli_exits_on_failure(
    mock_run: MagicMock,
    tmp_path: Path,
    pipeline_config_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    cog_dir = tmp_path / "cog"
    scene_dir = raw_dir / SCENE_ID
    scene_dir.mkdir(parents=True)
    (scene_dir / f"{SCENE_ID}_SR_B4.TIF").write_bytes(b"x")

    from wildfire_geo_ml.ingest.cog_convert import CogFileResult, CogSceneReport

    mock_run.return_value = [
        CogSceneReport(
            scene_id=SCENE_ID,
            results=[CogFileResult(scene_id=SCENE_ID, label="B4", status="failed", error="boom")],
        )
    ]

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(pipeline_config_path),
            "--input-dir",
            str(raw_dir),
            "--output-dir",
            str(cog_dir),
        ],
    )

    assert result.exit_code == 1
    assert "failed" in result.output
