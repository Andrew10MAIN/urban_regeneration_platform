"""
DAG: etl_penalties_addresses
Pipeline ETL dla kar straży miejskiej i adresów Łodzi.

Przepływ:
  setup_schema
      ↓
  create_run
      ↓
  fetch_stage_addresses        (WFS EMUiA → audit.stg_addresses)
      ↓
  process_new_penalties        (Excel → transform → audit.stg_penalties)
      ↓
  load_addresses               (stg → mined.adresses)
  load_penalties               (stg → mined.penalties)   ← równolegle
      ↓
  compute_variables            (mined.penalties → mined.variables)
      ↓
  ensure_metadata              (meta.var_description)
      ↓
  finalize_run                 (audit.etl_log ← status końcowy)

Wykrywanie nowych plików Excel:
  DAG sprawdza audit.processed_files. Nowe lub zmienione pliki (inny MD5)
  są przetwarzane; niezmienione pomijane. Dodaj nowy plik Excel do
  data/source/offenses_penalties/ i uruchom DAG — zostanie wykryty automatycznie.
"""

import sys
import logging
from datetime import timedelta

sys.path.insert(0, "/opt/airflow/urban_platform")
sys.path.insert(0, "/opt/airflow/urban_platform/airflow")

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

from src.config.db import get_engine

from extract.penalties_addresses import (
    get_unprocessed_files,
    fetch_osm_parks_squares,
    fetch_addresses,
    fetch_streets,
    load_penalties_excel,
)
from transform.penalties_addresses import (
    transform_addresses,
    build_streets_outside_pattern,
    transform_penalties_sheet,
    aggregate_penalties_to_variables,
    OFFENSE_ARTICLES,
)
from load.penalties_addresses import (
    ensure_tables,
    stage_addresses,
    stage_penalties,
    upsert_addresses,
    upsert_penalties,
    upsert_variables,
    upsert_metadata,
    mark_file_processed,
)
# Reuse run log functions from build_perm (wspólna tabela audit.etl_log)
from load.build_perm import create_run_log, finalize_run_log

log = logging.getLogger(__name__)

DAG_ID = "etl_penalties_addresses"

DEFAULT_ARGS = {
    "owner": "urban_platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}


@dag(
    dag_id=DAG_ID,
    description="ETL: Kary straży miejskiej + adresy EMUiA → mined.penalties / mined.adresses / mined.variables",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["etl", "penalties", "addresses", "city_guard"],
)
def etl_penalties_addresses():

    @task
    def setup_schema() -> None:
        """Migruje / tworzy tabele mined.adresses, mined.penalties, staging."""
        engine = get_engine()
        ensure_tables(engine)

    @task
    def create_run() -> int:
        engine = get_engine()
        return create_run_log(engine, dag_id=DAG_ID, source="mapa_lodz/city_guard")

    # ── Adresy ───────────────────────────────────────────────────────────────

    @task
    def fetch_stage_addresses(etl_run_id: int) -> dict:
        """
        Pobiera pełen snapshot adresów z WFS EMUiA dla bbox obszaru badań.
        Zapisuje do audit.stg_addresses.
        """
        import geopandas as gpd
        engine = get_engine()

        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry"
        )
        bbox_2177 = tuple(blocks_gdf.total_bounds)  # (minx, miny, maxx, maxy) w EPSG:2177

        raw_gdf = fetch_addresses(bbox_2177)
        transformed = transform_addresses(raw_gdf)
        rows_staged = stage_addresses(transformed, engine, etl_run_id)

        return {"rows_staged": rows_staged}

    # ── Kary ─────────────────────────────────────────────────────────────────

    @task
    def process_new_penalties(etl_run_id: int, addr_stats: dict) -> dict:
        """
        Sprawdza nowe pliki Excel. Dla każdego nowego:
        - Pobiera OSMnx (parki/place) + WFS ulice (potrzebne do filtrowania).
        - Czyta adresy ze staging (stg_addresses) do geokodowania.
        - Transformuje arkusze alcohol + offense.
        - Zapisuje do audit.stg_penalties z source_file.
        - Rejestruje plik w audit.processed_files.

        Jeśli brak nowych plików — kończy bez błędu (0 staged).
        addr_stats: wynik fetch_stage_addresses — przyjmujemy go tylko żeby wymusić kolejność.
        """
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
        engine = get_engine()

        new_files = get_unprocessed_files(engine, DAG_ID)
        if not new_files:
            log.info("Brak nowych plików Excel — pomijam przetwarzanie kar")
            return {"files_processed": 0, "rows_staged": 0}

        # Dane wspólne dla wszystkich plików (pobieramy raz)
        log.info("Pobieranie danych wspólnych (OSMnx, WFS ulice, bloki)...")
        parks_gdf = fetch_osm_parks_squares()

        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry"
        )
        # bbox obszaru badań w EPSG:2177 — taki sam CRS jak WFS EMUiA po swap_xy
        minx, miny, maxx, maxy = blocks_gdf.total_bounds
        bbox_2177 = shapely_box(minx, miny, maxx, maxy)

        streets_gdf = fetch_streets()
        streets_outside_pattern = build_streets_outside_pattern(streets_gdf, bbox_2177)

        # Adresy już są w staging z poprzedniego tasku
        addresses_gdf = gpd.read_postgis(
            f"SELECT gml_id, full_adress, geometry FROM audit.stg_addresses WHERE run_id = {etl_run_id}",
            engine, geom_col="geometry"
        )
        if addresses_gdf.crs is None:
            addresses_gdf = addresses_gdf.set_crs("EPSG:4326")

        total_staged = 0
        for file_info in new_files:
            log.info("Przetwarzam plik: %s", file_info["name"])
            sheets = load_penalties_excel(file_info["path"])
            all_gdfs = []

            if "alcohol" in sheets:
                gdf_alc = transform_penalties_sheet(
                    sheets["alcohol"],
                    pen_type="alcohol_consumption",
                    addresses_gdf=addresses_gdf,
                    parks_gdf=parks_gdf,
                    streets_outside_pattern=streets_outside_pattern,
                    article_filter=None,  # arkusz alkohol nie ma filtra artykułowego
                )
                all_gdfs.append(gdf_alc)

            if "offense" in sheets:
                gdf_off = transform_penalties_sheet(
                    sheets["offense"],
                    pen_type="offense",
                    addresses_gdf=addresses_gdf,
                    parks_gdf=parks_gdf,
                    streets_outside_pattern=streets_outside_pattern,
                    article_filter=OFFENSE_ARTICLES,
                )
                all_gdfs.append(gdf_off)

            if all_gdfs:
                import pandas as pd
                combined = gpd.GeoDataFrame(
                    pd.concat(all_gdfs, ignore_index=True),
                    geometry="geometry",
                    crs=parks_gdf.crs,
                )
                staged = stage_penalties(combined, engine, etl_run_id, file_info["name"])
                total_staged += staged
                mark_file_processed(engine, DAG_ID, file_info["name"], file_info["hash"], etl_run_id)
                log.info("Plik %s: %d rekordów staged", file_info["name"], staged)

        return {"files_processed": len(new_files), "rows_staged": total_staged}

    # ── Load ─────────────────────────────────────────────────────────────────

    @task
    def load_addresses(etl_run_id: int) -> int:
        engine = get_engine()
        return upsert_addresses(engine, etl_run_id)

    @task
    def load_penalties(etl_run_id: int) -> int:
        engine = get_engine()
        return upsert_penalties(engine, etl_run_id)

    @task
    def compute_variables() -> int:
        """
        Agreguje WSZYSTKIE kary z mined.penalties (historyczne + bieżące)
        per block × year × pen_type. Zapisuje do mined.variables.
        """
        import geopandas as gpd
        engine = get_engine()

        gdf_penalties = gpd.read_postgis(
            "SELECT pen_id, date, pen_type, geometry FROM mined.penalties",
            engine, geom_col="geometry"
        )
        blocks_gdf = gpd.read_postgis(
            "SELECT block_id, geometry FROM core.urban_blocks_geom",
            engine, geom_col="geometry"
        )
        df_vars = aggregate_penalties_to_variables(gdf_penalties, blocks_gdf)
        return upsert_variables(df_vars, engine)

    @task
    def ensure_metadata() -> None:
        engine = get_engine()
        upsert_metadata(engine)

    @task
    def finalize(
        etl_run_id: int = None,
        addr_stats: dict = None,
        pen_stats: dict = None,
        rows_addr_loaded: int = None,
        rows_pen_loaded: int = None,
    ) -> None:
        engine = get_engine()
        rows_staged = (addr_stats or {}).get("rows_staged", 0) + \
                      (pen_stats or {}).get("rows_staged", 0)
        finalize_run_log(
            engine,
            run_id=etl_run_id,
            rows_extracted=rows_staged,
            rows_staged=rows_staged,
            rows_new=(pen_stats or {}).get("rows_staged", 0),
            rows_loaded=(rows_addr_loaded or 0) + (rows_pen_loaded or 0),
            status="success",
        )

    # ── Wiring ───────────────────────────────────────────────────────────────
    # WAŻNE: process_new_penalties MUSI czekać na fetch_stage_addresses
    # (adresy muszą być w stg_addresses zanim kary zostaną przetransformowane).
    # Przekazujemy addr_stats jako argument → Airflow automatycznie serializes zależność.
    _setup        = setup_schema()
    etl_run_id    = create_run()
    addr_stats    = fetch_stage_addresses(etl_run_id)
    pen_stats     = process_new_penalties(etl_run_id, addr_stats)  # sekwencyjnie po addr

    rows_addr     = load_addresses(etl_run_id)
    rows_pen      = load_penalties(etl_run_id)

    _vars         = compute_variables()
    _meta         = ensure_metadata()

    # Kolejność: setup → run → fetch_addr → process_pen → [load_addr || load_pen] → [vars, meta] → finalize
    _setup >> etl_run_id
    pen_stats >> rows_addr  # load_addresses też po penaltach (etl_run_id z XCom wystarczy, ale dla czytelności)
    pen_stats >> rows_pen
    [rows_addr, rows_pen] >> _vars >> _meta >> finalize(
        etl_run_id, addr_stats, pen_stats, rows_addr, rows_pen
    )


etl_penalties_addresses()
