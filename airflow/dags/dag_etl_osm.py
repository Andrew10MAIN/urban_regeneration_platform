"""
DAG: etl_osm
Pipeline ETL dla danych OSM (POI + polygon).

Przepływ:
  setup_schema
      ↓
  create_run
      ↓
  download_geofabrik          (ZIP → /data/source/osm_latest/)
      ↓
  [fetch_stage_poi,           (POI SHP → transform → audit.stg_osm_poi)
   fetch_stage_poly]          (parking SHP + OSMnx scrubs → audit.stg_osm_poly) ← równolegle
      ↓
  [load_poi,                  (DELETE current year + INSERT → osm.poi)
   load_poly]                 (DELETE current year + INSERT → osm.poly) ← równolegle
      ↓
  aggregate_variables         (stg_osm_poi + stg_osm_poly → mined.variables)
      ↓
  ensure_metadata             (meta.var_description)
      ↓
  finalize_run

Logika roku:
  - Dane z Geofabrik/OSMnx pobierane są z datą = dzisiaj (np. 2026-07-05).
  - W osm.poi/osm.poly USUWANE są wszystkie wiersze dla bieżącego roku (YYYY),
    a wstawiane nowe. Dane z poprzednich lat pozostają nienaruszone.
  - W mined.variables rok = YYYY-01-01 (ON CONFLICT DO UPDATE — zawsze odświeżany).
"""

import sys
import logging
from datetime import date, timedelta

sys.path.insert(0, "/opt/airflow/urban_platform")
sys.path.insert(0, "/opt/airflow/urban_platform/airflow")

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

from src.config.db import get_engine
from load.build_perm import create_run_log, finalize_run_log

log = logging.getLogger(__name__)

DAG_ID = "etl_osm"

DEFAULT_ARGS = {
    "owner": "urban_platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


@dag(
    dag_id=DAG_ID,
    description="ETL: OSM (Geofabrik + OSMnx) → osm.poi / osm.poly / mined.variables",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["etl", "osm", "geofabrik", "osmnx"],
)
def etl_osm():

    @task
    def setup_schema() -> None:
        from load.osm_data import ensure_tables
        engine = get_engine()
        ensure_tables(engine)

    @task
    def create_run() -> int:
        engine = get_engine()
        return create_run_log(engine, dag_id=DAG_ID, source="Geofabrik/OSMnx")

    # ── Extract: Geofabrik ────────────────────────────────────────────────────

    @task(execution_timeout=timedelta(minutes=60))
    def download_geofabrik(etl_run_id: int) -> str:
        """Pobiera ZIP z Geofabrik i rozpakowuje POI + traffic_a do katalogu."""
        from extract.osm_data import download_geofabrik as _download, OSM_OUT_DIR
        out = _download(OSM_OUT_DIR)
        return str(out)

    # ── Extract + Stage: POI i Poly ───────────────────────────────────────────

    @task
    def fetch_stage_poi(etl_run_id: int, out_dir: str) -> dict:
        """Wczytuje POI SHP, transformuje, zapisuje do audit.stg_osm_poi."""
        from pathlib import Path
        import geopandas as gpd
        from transform.osm_data import transform_poi
        from load.osm_data import stage_poi

        engine = get_engine()
        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry",
        )
        poi_gdf = transform_poi(
            Path(out_dir) / "gis_osm_pois_free_1.shp",
            blocks_gdf,
            date.today(),
        )
        rows = stage_poi(poi_gdf, engine, etl_run_id)
        return {"rows_staged": rows}

    @task
    def fetch_stage_poly(etl_run_id: int, out_dir: str) -> dict:
        """Wczytuje parking SHP + scrubs z landuse SHP (Geofabrik),
        transformuje, zapisuje do audit.stg_osm_poly."""
        from pathlib import Path
        import geopandas as gpd
        from extract.osm_data import get_scrub_shp_path
        from transform.osm_data import transform_poly
        from load.osm_data import stage_poly

        engine = get_engine()
        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry",
        )
        poly_gdf = transform_poly(
            Path(out_dir) / "gis_osm_traffic_a_free_1.shp",
            get_scrub_shp_path(Path(out_dir)),
            blocks_gdf,
            date.today(),
        )
        rows = stage_poly(poly_gdf, engine, etl_run_id)

        # Usuń landuse po użyciu — plik zbyt duży na repo
        for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
            p = get_scrub_shp_path(Path(out_dir)).with_suffix(ext)
            if p.exists():
                p.unlink()
                log.info("Usunięto: %s", p)

        return {"rows_staged": rows}

    # ── Load → tabele docelowe ────────────────────────────────────────────────

    @task
    def load_poi(etl_run_id: int, _poi_staged: dict) -> int:
        from load.osm_data import load_poi as _load
        engine = get_engine()
        return _load(engine, etl_run_id, date.today().year)

    @task
    def load_poly(etl_run_id: int, _poly_staged: dict) -> int:
        from load.osm_data import load_poly as _load
        engine = get_engine()
        return _load(engine, etl_run_id, date.today().year)

    # ── Agregacja zmiennych ───────────────────────────────────────────────────

    @task
    def aggregate_variables(etl_run_id: int, _poi_loaded: int, _poly_loaded: int) -> int:
        """
        Czyta audit.stg_osm_poi + audit.stg_osm_poly dla bieżącego run_id,
        agreguje count/area_ratio per block i upsertuje do mined.variables.
        """
        import geopandas as gpd
        from transform.osm_data import (
            aggregate_poi_to_variables,
            aggregate_poly_to_variables,
        )
        from load.osm_data import upsert_osm_variables
        import pandas as pd

        engine = get_engine()
        current_year = date.today().year

        poi_gdf = gpd.read_postgis(
            f"SELECT osm_id, fclass, date, block_id, geometry "
            f"FROM audit.stg_osm_poi WHERE run_id = {etl_run_id}",
            engine, geom_col="geometry",
        )
        poly_gdf = gpd.read_postgis(
            f"SELECT osm_id, fclass, date, block_id, geometry "
            f"FROM audit.stg_osm_poly WHERE run_id = {etl_run_id}",
            engine, geom_col="geometry",
        )
        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry",
        )

        df_poi  = aggregate_poi_to_variables(poi_gdf, blocks_gdf, current_year)
        df_poly = aggregate_poly_to_variables(poly_gdf, blocks_gdf, current_year)
        all_vars = pd.concat([df_poi, df_poly], ignore_index=True)

        return upsert_osm_variables(all_vars, engine)

    # ── Metadane ─────────────────────────────────────────────────────────────

    @task
    def ensure_metadata() -> None:
        from load.osm_data import upsert_osm_metadata
        upsert_osm_metadata(get_engine())

    # ── Finalizacja ───────────────────────────────────────────────────────────

    @task
    def finalize(
        etl_run_id: int,
        poi_stats: dict,
        poly_stats: dict,
        rows_poi: int,
        rows_poly: int,
        rows_vars: int,
    ) -> None:
        engine = get_engine()
        staged = (poi_stats or {}).get("rows_staged", 0) + \
                 (poly_stats or {}).get("rows_staged", 0)
        finalize_run_log(
            engine,
            run_id=etl_run_id,
            rows_extracted=staged,
            rows_staged=staged,
            rows_new=staged,
            rows_loaded=(rows_poi or 0) + (rows_poly or 0),
            status="success",
        )

    # ── Wiring ───────────────────────────────────────────────────────────────
    _setup     = setup_schema()
    run_id     = create_run()
    out_dir    = download_geofabrik(run_id)

    poi_stats  = fetch_stage_poi(run_id, out_dir)
    poly_stats = fetch_stage_poly(run_id, out_dir)

    rows_poi   = load_poi(run_id, poi_stats)
    rows_poly  = load_poly(run_id, poly_stats)

    rows_vars  = aggregate_variables(run_id, rows_poi, rows_poly)
    _meta      = ensure_metadata()

    _setup >> run_id
    [rows_vars, _meta] >> finalize(
        run_id, poi_stats, poly_stats, rows_poi, rows_poly, rows_vars
    )


etl_osm()
