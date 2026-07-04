"""
DAG: etl_app_prices
Pipeline ETL dla transakcji mieszkaniowych z WFS RCN.

Przepływ:
  setup_schema
      ↓
  create_run
      ↓
  fetch_stage_prices   (WFS RCN ms:lokale → audit.stg_app_prices)
      ↓
  load_prices          (stg → mined.app_prices, ON CONFLICT DO NOTHING)
      ↓
  aggregate_variables  (mined.app_prices → mined.variables, wszystkie lata)
      ↓
  ensure_metadata      (meta.var_description)
      ↓
  finalize_run         (audit.etl_log ← status końcowy)

from_date: dynamicznie max(date) z mined.app_prices lub bieżący rok (fallback).
Zmienna: socVrAS00_avrg_00000000 — średnia cena/m² per block per year.
"""

import sys
import logging
from datetime import timedelta

sys.path.insert(0, "/opt/airflow/urban_platform")
sys.path.insert(0, "/opt/airflow/urban_platform/airflow")

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

from src.config.db import get_engine
from extract.app_prices import fetch_app_prices
from transform.app_prices import transform_app_prices
from load.app_prices import (
    ensure_tables,
    stage_app_prices,
    upsert_app_prices,
    aggregate_prices_to_variables,
    upsert_app_price_metadata,
)
from load.build_perm import create_run_log, finalize_run_log

log = logging.getLogger(__name__)

DAG_ID = "etl_app_prices"

DEFAULT_ARGS = {
    "owner": "urban_platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


@dag(
    dag_id=DAG_ID,
    description="ETL: Transakcje mieszkaniowe WFS RCN → mined.app_prices + mined.variables",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["etl", "app_prices", "rcn"],
)
def etl_app_prices():

    @task
    def setup_schema() -> None:
        engine = get_engine()
        ensure_tables(engine)

    @task
    def create_run() -> int:
        engine = get_engine()
        return create_run_log(engine, dag_id=DAG_ID, source="RCN")

    @task
    def fetch_stage_prices(etl_run_id: int) -> dict:
        """
        Pobiera transakcje z WFS RCN od ostatniej daty w mined.app_prices
        (lub od początku bieżącego roku jako fallback) i zapisuje do stg.
        """
        import geopandas as gpd
        from datetime import datetime
        from sqlalchemy import text

        engine = get_engine()

        # Wyznacz from_date dynamicznie
        with engine.connect() as conn:
            max_date = conn.execute(
                text("SELECT MAX(date) FROM mined.app_prices")
            ).scalar()

        if max_date is not None:
            from_date = max_date.strftime("%Y-%m-%d")
        else:
            from_date = f"{datetime.now().year}-01-01"

        log.info("Pobieranie transakcji od: %s", from_date)

        # Bbox z bloków
        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry",
        )
        bbox_2180 = tuple(blocks_gdf.to_crs("EPSG:2180").total_bounds)

        raw_gdf      = fetch_app_prices(bbox_2180)
        transformed  = transform_app_prices(raw_gdf, blocks_gdf, from_date)
        rows_staged  = stage_app_prices(transformed, engine, etl_run_id)

        return {"rows_staged": rows_staged, "from_date": from_date}

    @task
    def load_prices(etl_run_id: int, fetch_stats: dict) -> int:
        """Upsert audit.stg_app_prices → mined.app_prices."""
        engine = get_engine()
        return upsert_app_prices(engine, etl_run_id)

    @task
    def aggregate_variables(etl_run_id: int, rows_loaded: int) -> int:
        """
        Przelicza socVrAS00_avrg_00000000 ze WSZYSTKICH rekordów mined.app_prices.
        Nadpisuje wartości w mined.variables dla wszystkich lat.
        """
        engine = get_engine()
        return aggregate_prices_to_variables(engine, etl_run_id)

    @task
    def ensure_metadata() -> None:
        engine = get_engine()
        upsert_app_price_metadata(engine)

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
    _setup      = setup_schema()
    etl_run_id  = create_run()
    fetch_stats = fetch_stage_prices(etl_run_id)
    rows_loaded = load_prices(etl_run_id, fetch_stats)        # po fetch
    rows_vars   = aggregate_variables(etl_run_id, rows_loaded) # po load
    _meta       = ensure_metadata()

    _setup >> etl_run_id
    rows_vars >> _meta >> finalize(etl_run_id, fetch_stats, rows_loaded, rows_vars)


etl_app_prices()
