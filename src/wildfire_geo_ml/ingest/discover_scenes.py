"""
Discover Landsat-9 scenes over the study-area AOI via USGS LandsatLook STAC API.

WHY: Scene selection should be geography-driven (bbox + datetime + cloud cover),
not a manually curated ID list. Hardcoded scenes in config remain a reproducible fallback.
"""

import logging
from dataclasses import dataclass

from pystac import Item
from pystac_client import Client

from wildfire_geo_ml.ingest.config import DiscoverConfig, PipelineConfig, StudyAreaConfig
from wildfire_geo_ml.ingest.landsat_paths import filter_scenes, normalize_scene_id, parse_scene_id

logger = logging.getLogger(__name__)

LANDSAT9_SCENE_PREFIX = "LC09_L2SP_"
MISSING_CLOUD_COVER = 100.0


@dataclass(frozen=True)
class DiscoveredScene:
    """A Landsat scene returned from STAC discovery."""

    scene_id: str
    cloud_cover: float

    @property
    def wrs_path(self) -> str:
        return parse_scene_id(self.scene_id).wrs_path

    @property
    def wrs_row(self) -> str:
        return parse_scene_id(self.scene_id).wrs_row

    @property
    def acquisition_date(self) -> str:
        return parse_scene_id(self.scene_id).acquisition_date


def _cloud_cover_from_item(item: Item) -> float:
    """Extract eo:cloud_cover from a STAC item, defaulting to 100% if absent."""
    raw = item.properties.get("eo:cloud_cover", MISSING_CLOUD_COVER)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return MISSING_CLOUD_COVER


def _is_landsat9_l2sp(scene_id: str) -> bool:
    """Return True if scene_id matches Landsat-9 Collection 2 L2SP naming."""
    if not scene_id.startswith(LANDSAT9_SCENE_PREFIX):
        return False
    try:
        parse_scene_id(scene_id)
    except ValueError:
        return False
    return True


def discover_scenes(
    study_area: StudyAreaConfig,
    discover: DiscoverConfig,
    max_cloud_cover: float | None = None,
) -> list[DiscoveredScene]:
    """
    Query LandsatLook STAC for Landsat-9 SR scenes over the AOI under a cloud threshold.

    Parameters
    ----------
    study_area : StudyAreaConfig
        Bounding box and datetime interval for the search.
    discover : DiscoverConfig
        STAC API URL, collection, platform, and default cloud threshold.
    max_cloud_cover : float, optional
        Override ``discover.max_cloud_cover`` (percent; scenes must be strictly below).

    Returns
    -------
    list[DiscoveredScene]
        Deduplicated scenes sorted by cloud cover ascending (clearest first).

    Raises
    ------
    ValueError
        If no scenes pass cloud, platform, and ID filters.
    """
    threshold = discover.max_cloud_cover if max_cloud_cover is None else max_cloud_cover
    if not 0 <= threshold <= 100:
        msg = f"max_cloud_cover must be between 0 and 100, got {threshold}"
        raise ValueError(msg)

    client = Client.open(discover.stac_api_url)
    search = client.search(
        collections=[discover.collection],
        bbox=study_area.bbox,
        datetime=study_area.datetime,
        query={
            "eo:cloud_cover": {"lt": threshold},
            "platform": {"eq": discover.platform},
        },
    )

    by_id: dict[str, DiscoveredScene] = {}
    for item in search.items():
        scene_id = normalize_scene_id(item.id)
        if not _is_landsat9_l2sp(scene_id):
            continue
        cloud = _cloud_cover_from_item(item)
        if cloud >= threshold:
            continue
        existing = by_id.get(scene_id)
        if existing is None or cloud < existing.cloud_cover:
            by_id[scene_id] = DiscoveredScene(scene_id=scene_id, cloud_cover=cloud)

    results = sorted(by_id.values(), key=lambda scene: scene.cloud_cover)
    if not results:
        msg = (
            f"No Landsat-9 scenes found for bbox={study_area.bbox}, "
            f"datetime={study_area.datetime!r}, cloud_cover < {threshold}%"
        )
        raise ValueError(msg)

    logger.info(
        "Discovered %d scene(s) over AOI (cloud < %.1f%%)",
        len(results),
        threshold,
    )
    return results


def resolve_scene_list(
    pipeline: PipelineConfig,
    use_config_scenes: bool,
    max_cloud_cover: float | None = None,
    wrs_path: str | None = None,
    rows: list[str] | None = None,
    dates: list[str] | None = None,
    scene_override: list[str] | None = None,
) -> list[DiscoveredScene]:
    """
    Resolve scenes to download: STAC discovery (default) or config fallback.

    Parameters
    ----------
    pipeline : PipelineConfig
        Full pipeline configuration.
    use_config_scenes : bool
        If True, use ``ingest.scenes`` and skip cloud filtering.
    max_cloud_cover : float, optional
        Override cloud threshold for discovery.
    wrs_path : str, optional
        WRS-2 path filter applied after resolution.
    rows : list[str], optional
        WRS-2 row filter.
    dates : list[str], optional
        Acquisition date filter (YYYYMMDD).
    scene_override : list[str], optional
        Explicit scene IDs from ``--scenes`` CLI flag.

    Returns
    -------
    list[DiscoveredScene]
        Filtered scenes ready for download.
    """
    path_filter = wrs_path if wrs_path is not None else pipeline.ingest.wrs_path

    if scene_override is not None:
        scene_ids = filter_scenes(
            scene_override,
            wrs_path=path_filter,
            rows=rows,
            dates=dates,
        )
        return [DiscoveredScene(scene_id=sid, cloud_cover=0.0) for sid in scene_ids]

    if use_config_scenes:
        logger.info(
            "Using hardcoded scenes from config (--use-config-scenes); cloud filter bypassed"
        )
        scene_ids = filter_scenes(
            list(pipeline.ingest.scenes),
            wrs_path=path_filter,
            rows=rows,
            dates=dates,
        )
        return [DiscoveredScene(scene_id=sid, cloud_cover=0.0) for sid in scene_ids]

    discovered = discover_scenes(
        pipeline.study_area,
        pipeline.discover,
        max_cloud_cover=max_cloud_cover,
    )
    scene_ids = filter_scenes(
        [scene.scene_id for scene in discovered],
        wrs_path=path_filter,
        rows=rows,
        dates=dates,
    )
    cloud_by_id = {scene.scene_id: scene.cloud_cover for scene in discovered}
    return [
        DiscoveredScene(scene_id=sid, cloud_cover=cloud_by_id[sid])
        for sid in scene_ids
        if sid in cloud_by_id
    ]
