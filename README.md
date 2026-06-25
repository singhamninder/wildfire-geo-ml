# wildfire-geo-ml

**End-to-end cloud-native geospatial ML pipeline for vegetation encroachment risk scoring on power line corridors — Landsat-9 → STAC → H3 GeoParquet → Apache Sedona → LightGBM → AWS Lambda.**

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-pytest-yellow.svg)
![Linting](https://img.shields.io/badge/linting-ruff-purple.svg)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES (free/public)                   │
│  s3://usgs-landsat (Landsat-9 L2 SR)  ·  CAL FIRE FRAP Lines        │
│  s3://sentinel-cogs (Sentinel-2 fallback)  ·  NLCD 2021  ·  SRTM   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — STAC CATALOG                                             │
│  rio-cogeo → COG conversion (deflate, 6 overview levels)            │
│  pystac → Catalog / Collection / Item / Asset                       │
│  stac-validator → 0 errors  ·  EO + Projection + View extensions   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — H3 FEATURE ENGINEERING                                   │
│  rioxarray → NDVI / NBR / NDWI per scene                            │
│  h3-py → polyfill study area (resolution 8, ~461 m hexagons)        │
│  exactextract / rasterio.mask → zonal stats per H3 cell             │
│  pyarrow → H3-partitioned GeoParquet to S3/local                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — APACHE SEDONA ZONAL STATS                                │
│  SedonaContext (local[*] or EMR Serverless)                         │
│  RS_FromPath(s3://cog) → RS_ZonalStats(raster, line_buffer, mean)  │
│  Output: GeoParquet partitioned by h3_res8                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — ML RISK SCORER                                           │
│  Features: NDVI, NBR, NDWI, slope, dist_to_line, NLCD class        │
│  Spatial block CV (H3 res-4 blocks ~county-sized folds)             │
│  LightGBM classifier → MLflow tracking → SHAP explanations          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5 — AWS LAMBDA EVENT TRIGGER                                 │
│  S3 PUT event → handler.py                                          │
│  → rio cogeo validate (fail fast)                                   │
│  → boto3 emr-serverless.start_job_run (Sedona job)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Progress

| Phase | Status | Scope |
|---|---|---|
| 0 — Repo setup | Complete | uv, pre-commit, `.env.example` |
| 1 — STAC catalog | Complete | Landsat ingest, validate-first COG, pystac build, stac-validator, notebook round-trip |
| 2 — H3 features | Complete | NDVI/NBR/NDWI, H3 polyfill, GeoParquet, `notebooks/02_*`, `tests/test_{indices,h3_utils,zonal_stats,h3_partition}.py` |
| 3 — Sedona zonal stats | Complete | RS_ZonalStats over corridor buffers + optional H3 hexes; `notebooks/03_*`, `tests/test_{corridors,sedona_config,sedona_session,zonal_stats_job}.py` |
| 4 — ML risk scorer | Planned | LightGBM, spatial block CV, MLflow, SHAP |
| 5 — Lambda trigger | Planned | S3 PUT → COG validate → EMR Serverless |

A committed sample catalog lives under [`stac/`](stac/): collection `landsat-9-northern-ca-2024` with **9 Landsat-9 scenes** (WRS path 044, rows 031–033; summer 2024 acquisitions).

---

## Quick Start

### Prerequisites
- Python 3.12+ (pinned via `.python-version`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- AWS CLI configured (`aws configure` or `AWS_PROFILE` in `.env`) — required for **Phase 1 Landsat download** (Requester Pays bucket) and Phases 3/5
- Docker (optional, for Lambda local testing)

### Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/singhamninder/wildfire-geo-ml.git
cd wildfire-geo-ml

# 2. Install dependencies (CI-parity)
uv sync --locked --all-extras --dev

# 3. Copy and fill environment variables
cp .env.example .env
# Edit .env with your AWS profile / EMR application ID

# 4. Install pre-commit hooks
uv run pre-commit install

# 5. Verify setup
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src
uv run pytest tests -v
```

### Phase 1: Build the STAC Catalog

```bash
# Preview scene discovery over HFTD AOI (free STAC metadata; no download)
# Default: bbox + datetime + cloud cover < 10% from config/pipeline.yaml
uv run python -m wildfire_geo_ml.ingest.download_landsat \
    --config config/pipeline.yaml \
    --discover-only

# Stricter cloud filter (only scenes under 5% cloud)
uv run python -m wildfire_geo_ml.ingest.download_landsat \
    --config config/pipeline.yaml \
    --discover-only \
    --max-cloud-cover 5

# Download discovered scenes from s3://usgs-landsat (Requester Pays — AWS creds required)
# Scenes are selected via LandsatLook STAC over the HFTD AOI in config/pipeline.yaml
# Loads AWS_PROFILE from .env; data-transfer charges apply to your AWS account
uv run python -m wildfire_geo_ml.ingest.download_landsat \
    --config config/pipeline.yaml \
    --output-dir data/raw/

# Optional: filter to Butte County + July 15 only
uv run python -m wildfire_geo_ml.ingest.download_landsat \
    --config config/pipeline.yaml \
    --path 044 --rows 032 --dates 20240715 \
    --output-dir data/raw/

# Validate-first COG ingest (USGS C2 bands are usually already COG — validate, copy,
# and re-profile with rio-cogeo only when validation fails; use --force to re-run)
uv run python -m wildfire_geo_ml.ingest.cog_convert \
    --input-dir data/raw/ \
    --output-dir data/cog/

# Build STAC catalog (asset hrefs default to s3://usgs-landsat; use --local-hrefs
# for offline/tests with local COG + MTL paths instead of S3 URIs)
uv run python -m wildfire_geo_ml.stac_builder.build_catalog \
    --config config/pipeline.yaml \
    --cog-dir data/cog/ \
    --raw-dir data/raw/ \
    --output-dir stac/

# Validate
uv run stac-validator validate stac/catalog.json --recursive

# STAC-client round-trip: read the catalog back like an analyst would
uv run jupyter execute notebooks/01_explore_stac_catalog.ipynb
```

### Phase 2: H3 Feature Engineering

```bash
uv run python -m wildfire_geo_ml.features.h3_partition \
    --config config/pipeline.yaml \
    --cog-dir data/cog/ \
    --output data/features/h3_partitioned/

# Optional: filter to one scene/date
uv run python -m wildfire_geo_ml.features.h3_partition \
    --config config/pipeline.yaml \
    --cog-dir data/cog/ \
    --rows 032 --dates 20240711 \
    --output data/features/h3_partitioned/

# Feature round-trip notebook (self-contained synthetic demo)
uv run jupyter execute notebooks/02_explore_h3_features.ipynb
```

### Phase 3: Sedona Zonal Stats

Requires **Java 17+** (`export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"`).

```bash
# Corridor stats (primary deliverable) — writes data/features/corridor_partitioned
uv run python -m wildfire_geo_ml.sedona.cli \
    --config config/pipeline.yaml \
    --cog-dir data/cog/ \
    --region corridor

# Optional: H3 hex stats (h3_res8-partitioned) — writes data/features/sedona_hex_partitioned
uv run python -m wildfire_geo_ml.sedona.cli \
    --config config/pipeline.yaml \
    --cog-dir data/cog/ \
    --region hex \
    --no-prepare-indices

# Filter to one scene/date
uv run python -m wildfire_geo_ml.sedona.cli \
    --config config/pipeline.yaml \
    --cog-dir data/cog/ \
    --rows 032 --dates 20240711 \
    --region corridor \
    --no-fetch-lines --no-prepare-indices

# Sedona zonal-stats notebook (self-contained synthetic demo)
uv run jupyter execute notebooks/03_sedona_zonal_stats.ipynb

# Slow integration tests (Java 17)
uv run pytest tests/test_sedona_session.py tests/test_zonal_stats_job.py -v -m slow
```

### Phase 4: Train Risk Scorer *(planned — not yet implemented)*

```bash
# Start MLflow UI first (optional)
uv run mlflow ui &

uv run python -m wildfire_geo_ml.ml.train \
    --config config/train.yaml
```

Training parameters live in `config/train.yaml` — feature paths, LightGBM hyperparams, spatial-CV resolution, MLflow run name. The script loads + validates the YAML once at the entry point and passes a typed config object inward.

### Phase 5: Test Lambda Handler Locally *(planned — not yet implemented)*

```bash
uv run pytest tests/test_lambda_handler.py -v
```

### Notebooks

Exploratory notebooks are Jupyter notebooks at `notebooks/*.ipynb`. They import from the `wildfire_geo_ml` package and stay thin — every notebook runs clean top-to-bottom.

```bash
uv run jupyter lab notebooks/01_explore_stac_catalog.ipynb
uv run jupyter lab notebooks/02_explore_h3_features.ipynb
uv run jupyter nbconvert --to html --execute notebooks/01_explore_stac_catalog.ipynb
```

---

## Project Structure

```
wildfire-geo-ml/
├── pyproject.toml           # PEP 621 project + uv dependency groups
├── uv.lock                  # Pinned dependency graph (committed)
├── .python-version          # Pins local interpreter
├── config/
│   └── pipeline.yaml        # Study area, STAC discovery, ingest settings
├── scripts/
│   └── check_no_pge_references.sh
├── src/
│   └── wildfire_geo_ml/     # Import package (src layout)
│       ├── ingest/          # download_landsat, discover_scenes, cog_convert, …
│       ├── stac_builder/    # build_catalog, mtl_parser
│       ├── features/        # indices, h3_utils, zonal_stats, h3_partition CLI
│       ├── sedona/          # SedonaContext, corridor/hex RS_ZonalStats CLI
│       ├── ml/              # (planned) LightGBM + spatial CV + MLflow + SHAP
│       └── lambda_trigger/  # (planned) AWS Lambda S3 event handler
├── tests/
│   ├── test_discover_scenes.py
│   ├── test_download_landsat.py
│   ├── test_cog_convert.py
│   ├── test_stac_builder.py
│   ├── test_indices.py
│   ├── test_h3_utils.py
│   ├── test_zonal_stats.py
│   └── test_h3_partition.py
├── stac/                    # Sample STAC catalog (9 items, committed)
│   ├── catalog.json
│   └── landsat-9-northern-ca-2024/
├── data/                    # Raw + COG imagery (gitignored)
├── notebooks/               # Jupyter notebooks (*.ipynb)
└── .github/workflows/       # CI: ruff + ty + pytest
```

---

## Study area

The default study extent covers **HFTD Tier 3** counties in Northern California: **Butte**, **Plumas**, and **Shasta**. Tier 3 is the highest fire-threat designation in the state. Butte County includes Paradise, CA — the community devastated by the 2018 Camp Fire, the deadliest and most destructive wildfire in California history. That geography motivates corridor-scale vegetation risk modeling on transmission lines in high-threat territory.

---

## License

MIT — open for educational use.

Data sources: Landsat-9 imagery is public domain (USGS). CAL FIRE FRAP data is California Open Data. NLCD is USGS Open Data.
