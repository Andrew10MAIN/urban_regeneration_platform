"""
DAG: etl_buildings
Pipeline ETL dla budynków Łodzi z WFS EGiB + agregacja zmiennych do mined.variables.

Przepływ:
  setup_schema
      ↓
  create_run
      ↓
  fetch_stage_buildings    (WFS EGiB → audit.stg_buildings)
      ↓
  load_buildings           (stg → mined.buildings, ON CONFLICT DO UPDATE)
      ↓
  compute_variables        (mined.buildings → mined.variables, rok bieżący)
      ↓
  ensure_metadata          (meta.var_description)
      ↓
  finalize_run             (audit.etl_log ← status końcowy)

Zmienne:
  bdEnvFtxx_arrt_00000000  — powierzchnia zabudowy  [m²]  = SUM(area)
  bdEnvFRxx_arrt_00000000  — gęstość zabudowy       [m²]  = SUM(area × kondygnacje)

Rok: bieżący rok kalendarzowy → YYYY-01-01.
Tabela pomocnicza zmian: audit.stg_building_vars (append każdego runu).
"""

import sys
import logging
from datetime import timedelta

sys.path.insert(0, "/opt/airflow/urban_platform")
sys.path.insert(0, "/opt/airflow/urban_platform/airflow")

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

from src.config.db import get_engine
from extract.buildings import fetch_buildings
from transform.buildings import transform_buildings
from load.buildings import (
    ensure_tables,
    stage_buildings,
    upsert_buildings,
    aggregate_buildings_to_variables,
    upsert_building_metadata,
)
from load.build_perm import create_run_log, finalize_run_log

log = logging.getLogger(__name__)

DAG_ID = "etl_buildings"

DEFAULT_ARGS = {
    "owner": "urban_platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


@dag(
    dag_id=DAG_ID,
    description="ETL: Budynki WFS EGiB → mined.buildings + mined.variables (bdEnvFtxx / bdEnvFRxx)",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["etl", "buildings", "egib"],
)
def etl_buildings():

    @task
    def setup_schema() -> None:
        engine = get_engine()
        ensure_tables(engine)

    @task
    def create_run() -> int:
        engine = get_engine()
        return create_run_log(engine, dag_id=DAG_ID, source="mapa_lodz/egib")

    @task
    def fetch_stage_buildings(etl_run_id: int) -> dict:
        """Pobiera budynki z WFS EGiB i zapisuje do audit.stg_buildings."""
        import geopandas as gpd
        engine = get_engine()

        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry"
        )
        bbox_2177 = tuple(blocks_gdf.total_bounds)

        raw_gdf = fetch_buildings(bbox_2177)
        transformed = transform_buildings(raw_gdf)
        rows_staged = stage_buildings(transformed, engine, etl_run_id)

        return {"rows_staged": rows_staged}

    @task
    def load_buildings(etl_run_id: int, fetch_stats: dict) -> int:
        """Upsert audit.stg_buildings → mined.buildings."""
        engine = get_engine()
        return upsert_buildings(engine, etl_run_id)

    @task
    def compute_variables(etl_run_id: int, rows_loaded: int) -> int:
        """
        Agreguje powierzchnię zabudowy i gęstość per block → mined.variables.
        Rok = bieżący rok kalendarzowy (YYYY-01-01).
        Audit trail → audit.stg_building_vars.
        """
        engine = get_engine()
        return aggregate_buildings_to_variables(engine, etl_run_id)

    @task
    def ensure_metadata() -> None:
        engine = get_engine()
        upsert_building_metadata(engine)

    @task
    def finalize(
        etl_run_id: int,
        fetch_stats: dict,
        rows_loaded: int,
        rows_vars: int,
    ) -> None:
        engine = get_engine()
        rows_staged = fetch_stats.get("rows_staged", 0)
        finalize_run_log(
            engine,
            run_id=etl_run_id,
            rows_extracted=rows_staged,
            rows_staged=rows_staged,
            rows_new=rows_loaded,
            rows_loaded=rows_loaded,
            status="success",
        )

    # ── Wiring ───────────────────────────────────────────────────────────────
    # setup → run → fetch → load → compute_vars → metadata → finalize
    _setup       = setup_schema()
    etl_run_id   = create_run()
    fetch_stats  = fetch_stage_buildings(etl_run_id)
    rows_loaded  = load_buildings(etl_run_id, fetch_stats)   # sekwencyjnie po fetch
    rows_vars    = compute_variables(etl_run_id, rows_loaded) # sekwencyjnie po load
    _meta        = ensure_metadata()

    _setup >> etl_run_id
    rows_vars >> _meta >> finalize(etl_run_id, fetch_stats, rows_loaded, rows_vars)


etl_buildings()
