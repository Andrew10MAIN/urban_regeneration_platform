"""
DAG: etl_buildings
Pipeline ETL dla budynków Łodzi z WFS EGiB.

Przepływ:
  setup_schema
      ↓
  create_run
      ↓
  fetch_stage_buildings    (WFS EGiB → audit.stg_buildings)
      ↓
  load_buildings           (stg → mined.buildings)
      ↓
  finalize_run             (audit.etl_log ← status końcowy)

Źródło: mapa.lodz.pl/OGC/EGiB — warstwa ms:budynki
Śledzenie: audit.etl_log (wspólne z innymi DAGami)
Staging:   audit.stg_buildings
Docelowa:  mined.buildings
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
from load.buildings import ensure_tables, stage_buildings, upsert_buildings
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
    description="ETL: Budynki Łódź WFS EGiB → mined.buildings",
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
        """
        Pobiera budynki z WFS EGiB dla bbox obszaru badań.
        Transformuje i zapisuje do audit.stg_buildings.
        """
        import geopandas as gpd
        engine = get_engine()

        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry"
        )
        bbox_2177 = tuple(blocks_gdf.total_bounds)  # (minx, miny, maxx, maxy) w EPSG:2177

        raw_gdf = fetch_buildings(bbox_2177)
        transformed = transform_buildings(raw_gdf)
        rows_staged = stage_buildings(transformed, engine, etl_run_id)

        return {"rows_staged": rows_staged}

    @task
    def load_buildings(etl_run_id: int, fetch_stats: dict) -> int:
        """
        Upsert z audit.stg_buildings → mined.buildings.
        Przyjmuje fetch_stats tylko po to, żeby wymusić sekwencję po fetch_stage_buildings.
        """
        engine = get_engine()
        return upsert_buildings(engine, etl_run_id)

    @task
    def finalize(etl_run_id: int, fetch_stats: dict, rows_loaded: int) -> None:
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
    # setup → create_run → fetch_stage → load → finalize
    _setup      = setup_schema()
    etl_run_id  = create_run()
    fetch_stats = fetch_stage_buildings(etl_run_id)
    rows_loaded = load_buildings(etl_run_id, fetch_stats)  # sekwencyjnie po fetch

    _setup >> etl_run_id
    rows_loaded >> finalize(etl_run_id, fetch_stats, rows_loaded)


etl_buildings()
