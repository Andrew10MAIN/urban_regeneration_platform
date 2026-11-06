# Urban Regeneration Assessment Platform

> ⚠️ **BETA** — This repository is under active development as part of ongoing PhD research. Interfaces, data schemas, and pipeline configurations may change without notice.

## 1. Purpose

This repository contains the full experimental pipeline developed for a PhD thesis assessing the Łódź Urban Regeneration Programme (Program Rewitalizacji Łodzi). The thesis addresses two core research questions:

- **Assessment** — Did the programme generate measurable causal effects on key urban indicators in treated city blocks?
- **Policy recommendation** — Given a fixed budget for future regeneration projects, which city blocks should be prioritised to maximise the aggregate causal uplift across multiple outcomes simultaneously?

The platform implements a full data-science pipeline: from raw data ingestion through causal machine learning to an interactive dashboard presenting evidence-based policy recommendations.

## 2. Spatial Unit of Analysis

All statistics are aggregated to **urban blocks** — the smallest aggregation level in the city structure from the perspective of urban planning, delineated within the Łódź regeneration zone (*strefy rewitalizacji*). This granularity enables block-level causal estimation and spatially-aware optimisation.

## 3. Statistical & Methodological Foundation

### 3.1 Causal Machine Learning — Individual Treatment Effects

Treatment effects (uplifts) are estimated using **Causal Forest DML** (EconML), which recovers heterogeneous Individual Treatment Effects (ITE) for each urban block. Two treatment groups are distinguished:

- **Direct treatment** — blocks where regeneration projects were physically implemented
- **Indirect treatment (spatial spillover)** — blocks adjacent to directly treated blocks, capturing neighbourhood-level effects

Outcome models are estimated sequentially, each tracked as a separate MLflow experiment. The set of target variables and their confounders is defined in the run configuration and may change as the research develops.

Variables in `mined.variables` do not all cover the same years or the same blocks, so the loader resolves this per model rather than assuming a common panel. Each variable is read at the configured pre-period where available, otherwise at the nearest earlier year still preceding the post-period; the year actually used is recorded in `results.models` and `results.features`. Y, X and T are then built on the intersection of the block sets of the target, every confounder and the block register, and the optimisation runs on the intersection of the block sets of all models entering the objective. Partial coverage therefore narrows the estimation sample in a logged, reproducible way instead of silently pairing mismatched blocks.

### 3.2 Policy Optimisation — Integer Linear Programming

Future regeneration project allocation is framed as a budget-constrained ILP problem solved with **PuLP** (CBC solver). The objective function maximises a weighted sum of causal uplifts, subject to a budget constraint and a connectivity constraint requiring the selected blocks to form a single contiguous area. Each uplift — one per (target variable, treatment arm) pair — carries its own weight; only uplifts that pass a significance test enter the objective, and their weights are rescaled to sum to 1.

Rather than a single budget, the pipeline sweeps a configurable budget range, solving one ILP per scenario in parallel and recording the objective function value for each. The reference sweep runs from 250 to 500 million PLN in 5-million steps (51 scenarios); for context, the first 8 completed projects cost roughly 1 billion PLN in total. Because each solve is capped by a time limit, a scenario may return a feasible rather than proven-optimal solution, which is why the solver status is stored alongside the objective value.

### 3.3 Regeneration Cost Prediction — Random Forest

Project implementation costs are predicted at block level using a Random Forest regression model trained on features such as block area, vacancy rate, and building stock characteristics. SHAP values are computed to explain cost drivers. Predicted costs feed directly into the ILP budget constraint.

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Containerisation | Docker + Docker Compose (profiles: `etl`, `ml`, `app`) |
| Spatial database | PostGIS 16 |
| ETL orchestration | Apache Airflow 2.10.3 |
| ML experiment tracking | MLflow 2.19 |
| Causal ML | EconML (CausalForestDML) |
| Optimisation | PuLP + CBC solver |
| Cost model | scikit-learn Random Forest + SHAP |
| Dashboard | Streamlit + Folium |
| Language | Python 3.12 |

## 5. Reproducibility

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + Docker Compose (Linux)
- Git

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/<username>/<repo>.git
cd <repo>

# 2. Start the database
docker compose up -d postgis

# 3. Load base data (urban block grid, legacy variables, Census, OSM aggregates)
docker compose --profile ml run --rm ml python scripts/bootstrap/load_initial_data.py

# 4. Start Airflow and run ETL pipelines
docker compose --profile etl up -d
# Open http://localhost:8080 (admin / admin)
# Trigger DAGs in order (see ETL section below)

# 5. Train the cost prediction model (Random Forest)
docker compose --profile ml run --rm rf

# 6. Run causal forest + ILP optimisation pipeline
docker compose --profile ml run --rm ml

# 7. Launch the dashboard
docker compose --profile app up streamlit
# Open http://localhost:8501
```

Steps 3, 5 and 6 build the Python image on first invocation and start their dependencies (`postgis`, `mlflow`) automatically. After changing `requirements.txt`, add `--build` to force a rebuild.

MLflow experiment tracking is available at `http://localhost:5000`.

All service ports are published on the loopback interface only; credentials in `docker-compose.yml` are local development defaults and must be replaced in any non-local deployment.

## 6. Bootstrap Overview

The bootstrap script (`scripts/bootstrap/load_initial_data.py`) loads the foundational data layers that all ETL pipelines depend on. These files are committed to the repository under `data/source/` and do not require external API access.

| Data | Source | Reason pre-loaded |
|---|---|---|
| Urban block grid | AutoCAD export (city planning office) | Defines spatial units for all aggregation |
| Sensitive urban survey data | Field survey aggregated to block level (`df_legacy_vars.parquet`) | Cannot be re-fetched; privacy-protected at block granularity |
| Historical OSM data | Pre-processed OSM snapshots aggregated to blocks | Reproducible historical baseline independent of live OSM state |
| Census 2021 | GUS 100×100 m grid aggregated to urban blocks | Official population statistics reference |
| Apartment transaction prices | Łódź city spatial data portal (historical extract) | Historical series not exposed by the live WFS endpoint |
| Building plot transaction prices | Register of Real Estate Prices (RCN) | No live endpoint; block-level series requires neighbour imputation |
| Public transport stops | Pre-processed OSM snapshots, 2015–2025 | Reproducible historical baseline independent of live OSM state |
| Administrative penalties | Municipal Guard of Łódź, public information request | Not available through any programmatic endpoint |

## 7. Database Overview

The PostGIS database is organised into the following schemas:

| Schema | Content |
|---|---|
| `core` | Urban block attributes (treatment status, year) and block geometries |
| `mined` | Point-level POI data (small catering businesses, apartment prices, building permits, penalties, addresses, buildings), aggregated variables per block per year |
| `osm` | Raw and processed OpenStreetMap POI and polygon layers |
| `meta` | Variable descriptions and metadata |
| `results` | ML model metadata, hyperparameters, feature importance, causal uplifts, RF cost predictions, SHAP values, ILP optimisation outputs (one row per budget scenario, plus the objective weights actually applied) |
| `regeneration` | Regeneration project geometries and activity records |

Schema changes are applied by the numbered files in `docker/postgres/init/`. They run automatically only when the PostGIS volume is created from scratch; on an existing database each migration is applied by hand, e.g.

```bash
docker exec -i urban_platform_db psql -U urban_user -d urban_db < docker/postgres/init/008_optimization_sweep.sql
```

## 8. ETL Pipeline Overview

All pipelines are orchestrated via Apache Airflow. Trigger in the following order after bootstrap:

| DAG | Data loaded | External source |
|---|---|---|
| `etl_build_perm` | Building permits (`mined.Build_perm`) | GUNB WFS (Geoportal) |
| `etl_penalties_addresses` | Administrative penalties + address points | Excel file in `data/source/offenses_penalties/` + EMUiA address service |
| `etl_buildings` | Building footprints with attributes | WFS EGiB (cadastral registry) |
| `etl_app_prices` | Apartment transaction prices (`mined.app_prices`) | WFS RCN — incremental update of the bootstrap extract |
| `etl_osm` | Live OSM POI and polygon layers | Geofabrik (`lodzkie-latest-free.shp.zip`) |

The first four DAGs can be triggered in parallel. `etl_osm` should run last due to file size (~250 MB download).

## 9. Dashboard Overview

The Streamlit application (`http://localhost:8501`) presents results across two analytical sections, switchable via the left sidebar:

### EDA — Exploratory Data Analysis

- **Heatmap** — interactive Folium choropleth of any mined variable for a selected year, with optional overlays of OSM POI categories, administrative penalties, building permits, and apartment price points
- **Correlation** — scatter plot with optional OLS regression line for any pair of mined variables
- **Parallel Trends** — mean outcome trajectories by treatment group across all observed years, with regeneration programme start/end markers

### Causal Inference

- **CATE** — Conditional Average Treatment Effects: scatter plot of uplift vs. confounder value for treated blocks, enabling heterogeneity analysis
- **Policy Recommendations** — ILP-optimised block selection map, total uplift bar chart by treatment type, and budget utilisation summary

All views support export: interactive maps as HTML, charts as PNG.

## 10. Data Provenance & Privacy

Four of the committed source datasets warrant explicit provenance notes.

**Sensitive urban survey data** (`data/source/legacy_variables/df_legacy_vars.parquet`) — socio-economic indicators (average age, educational attainment, unemployment level, wage quantile of inhabitants aged 25+) derived from a field survey. The dataset is committed **already aggregated to urban-block level**: it contains no individual records, no addresses and no personal identifiers, only block–year–variable–value tuples. Aggregation was performed before the data entered this repository and is irreversible. Variable definitions are documented in `meta_var.parquet` and loaded into the `meta` schema.

**Apartment transaction prices** (`data/source/app_prices_historic/`) — the historical extract was downloaded from the Łódź city spatial data portal, <https://nowa.mapa.lodz.pl/dla-profesjonalistow/> (accessed June 2026), and is loaded directly into the database by the bootstrap script rather than through an API. Records describe cadastral premises (premises and building identifiers, transaction date, floor, floor area, gross price); they contain no personal identifiers and are not linked to any natural person. The `etl_app_prices` DAG subsequently extends this extract with newer transactions from the WFS RCN endpoint.

**Building plot transaction prices** (`data/source/build_plots_prices/`) — sourced from the Register of Real Estate Prices (*Rejestr Cen Nieruchomości*, RCN). Records describe cadastral plots (plot identifier, land-use category, gross transaction price, document and record dates) with plot geometry; they contain no personal identifiers and are not linked to any natural person. Block-level values are computed as the mean gross price per m² of plot area, and blocks without a transaction in a given year receive the mean of their neighbouring blocks (iterated until the whole area is covered), so the resulting variable is partly imputed by construction.

**Administrative penalties** (`data/source/offenses_penalties/`) — obtained from the Municipal Guard of Łódź (Straż Miejska w Łodzi) in response to a written public information request submitted on 23 January 2024 (*wniosek o udostępnienie informacji publicznej*). The data were supplied by the Municipal Guard in anonymised form: each record contains only the location, date and legal classification of the offence, with no personal identifiers.

All four datasets enter the pipeline solely as inputs for aggregation to urban-block level; all modelling, causal estimation and optimisation are performed on block-level aggregates. Point-level layers appear in the dashboard only as optional EDA overlays.

**Other sources** — OpenStreetMap extracts are used under the Open Database Licence (ODbL); Census 2021 grids are published by Statistics Poland (GUS); building permits and building footprints originate from the GUNB and EGiB WFS services respectively.

## Licence

MIT — see [LICENSE](LICENSE).
