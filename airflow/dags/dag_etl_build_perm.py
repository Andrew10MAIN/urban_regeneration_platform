"""
DAG: etl_build_perm
Pipeline ETL dla pozwoleń na budowę (Geoportal GUNB WFS).

Przepływ:
  create_run
      ↓
  extract_and_stage          (WFS → audit.stg_build_perm)
      ↓
  compute_diff               (stg vs mined.Build_perm)
      ↓
  transform_and_load         (stg → mined.Build_perm)
      ↓
  aggregate_and_load_vars    (mined.Build_perm → mined.variables)
      ↓
  ensure_metadata            (meta.var_description)
      ↓
  finalize_run               (audit.etl_log ← status końcowy)

Uruchamiaj ręcznie (schedule=None) lub ustaw @monthly.
"""

import sys
import logging
from datetime import datetime, timedelta

# Global src/ (config, ml, ...) + airflow/ ETL modules
sys.path.insert(0, "/opt/airflow/urban_platform")
sys.path.insert(0, "/opt/airflow/urban_platform/airflow")

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

from src.config.db import get_engine
from extract.build_perm import extract
from transform.build_perm import transform_pipeline
from load.build_perm import (
    create_run_log,
    stage_raw,
    compute_diff,
    upsert_build_perm,
    aggregate_variables,
    upsert_variables,
    upsert_metadata,
    finalize_run_log,
)

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "urban_platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


@dag(
    dag_id="etl_build_perm",
    description="ETL: Pozwolenia na budowę z Geoportal GUNB → mined.Build_perm + mined.variables",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["etl", "build_perm", "gunb"],
)
def etl_build_perm():

    @task
    def create_run() -> int:
        engine = get_engine()
        return create_run_log(engine, dag_id="etl_build_perm")

    @task
    def extract_and_stage(etl_run_id: int) -> dict:
        """Ekstrakcja z WFS + zapis do staging. Zwraca statystyki."""
        engine = get_engine()

        log.info("=== EXTRACT ===")
        raw_gdf = extract(engine)
        rows_extracted = len(raw_gdf)

        log.info("=== TRANSFORM (pre-stage) ===")
        transformed_gdf = transform_pipeline(raw_gdf, engine)

        log.info("=== STAGE ===")
        rows_staged = stage_raw(transformed_gdf, engine, etl_run_id)

        return {
            "rows_extracted": rows_extracted,
            "rows_staged":    rows_staged,
        }

    @task
    def diff_check(etl_run_id: int) -> int:
        """Liczy nowe rekordy względem istniejących w mined.Build_perm."""
        engine = get_engine()
        return compute_diff(engine, etl_run_id)

    @task
    def load_build_perm(etl_run_id: int) -> int:
        """Czyta dane ze staging (z block_id) i upsertuje do mined.Build_perm."""
        import geopandas as gpd
        engine = get_engine()
        gdf = gpd.read_postgis(
            f"SELECT build_perm_id, build_plot_no, issue_date AS date, description, geometry, block_id "
            f"FROM audit.stg_build_perm WHERE run_id = {etl_run_id}",
            engine,
            geom_col="geometry"
        )
        return upsert_build_perm(gdf, engine)

    @task
    def load_variables(etl_run_id: int = None) -> int:
        """Agreguje ze staging i upsertuje do mined.variables."""
        engine = get_engine()
        df = aggregate_variables(engine, run_id=etl_run_id)
        return upsert_variables(df, engine)

    @task
    def ensure_metadata() -> None:
        engine = get_engine()
        upsert_metadata(engine)

    @task
    def finalize(
        etl_run_id: int = None,
        stage_stats: dict = None,
        rows_new: int = None,
        rows_loaded: int = None,
    ) -> None:
        engine = get_engine()
        finalize_run_log(
            engine,
            run_id=etl_run_id,
            rows_extracted=stage_stats["rows_extracted"],
            rows_staged=stage_stats["rows_staged"],
            rows_new=rows_new,
            rows_loaded=rows_loaded,
            status="success",
        )

    # ── Wiring ──────────────────────────────────────────────────────────────
    etl_run_id  = create_run()
    stage_stats = extract_and_stage(etl_run_id)
    rows_new    = diff_check(etl_run_id)
    rows_loaded = load_build_perm(etl_run_id)
    _vars       = load_variables(etl_run_id)
    _meta       = ensure_metadata()

    stage_stats >> rows_new >> rows_loaded >> [_vars, _meta] >> finalize(
        etl_run_id, stage_stats, rows_new, rows_loaded
    )


etl_build_perm()
