"""
Load: Penalties + Addresses
Staging, upsert do mined.adresses / mined.penalties / mined.variables.
"""

import logging

import pandas as pd
import geopandas as gpd
from geoalchemy2 import Geometry
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

METADATA = [
    {
        "var_id": "urVibAlPn_coun_00000000",
        "unit": "count",
        "origin": "city_guard",
        "description": (
            "The number of the penalties for alcohol consumption in public space "
            "in a particular urban block in a particular year"
        ),
    },
    {
        "var_id": "urVibOfPn_coun_00000000",
        "unit": "count",
        "origin": "city_guard",
        "description": (
            "The number of the penalties for offences in public space "
            "in a particular urban block in a particular year"
        ),
    },
]


# ─── Schema migration ────────────────────────────────────────────────────────

def ensure_tables(engine: Engine) -> None:
    """
    Tworzy / migruje tabele przy pierwszym uruchomieniu DAGu.
    - mined.adresses i mined.penalties mogą istnieć ze starym schematem (z 003_tables.sql)
      → sprawdza kolumny i odtwarza jeśli nieaktualne.
    - Staging tables tworzone z IF NOT EXISTS (bezpieczne).
    """
    with engine.begin() as conn:

        # ── mined.adresses ──
        has_gml_id = conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'mined' AND table_name = 'adresses'
              AND column_name = 'gml_id'
        """)).fetchone()
        if not has_gml_id:
            log.info("Migracja: odtwarzam mined.adresses z nowym schematem")
            conn.execute(text("DROP TABLE IF EXISTS mined.adresses CASCADE"))
            conn.execute(text("""
                CREATE TABLE mined.adresses (
                    gml_id      TEXT PRIMARY KEY,
                    guid        TEXT,
                    full_adress TEXT,
                    street      TEXT,
                    building_no TEXT,
                    zip_code    TEXT,
                    status      TEXT,
                    geometry    GEOMETRY(Point, 2177)
                )
            """))

        # ── mined.penalties ──
        has_place = conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'mined' AND table_name = 'penalties'
              AND column_name = 'place_of_penalty'
        """)).fetchone()
        if not has_place:
            log.info("Migracja: odtwarzam mined.penalties z nowym schematem")
            conn.execute(text("DROP TABLE IF EXISTS mined.penalties CASCADE"))
            conn.execute(text("""
                CREATE TABLE mined.penalties (
                    pen_id           BIGINT PRIMARY KEY,
                    date             TIMESTAMP,
                    place_of_penalty TEXT,
                    pen_type         TEXT,
                    geometry         GEOMETRY(Point, 2177)
                )
            """))

        # ── audit.stg_addresses ──
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit.stg_addresses (
                id          SERIAL,
                run_id      INTEGER REFERENCES audit.etl_log(run_id),
                gml_id      TEXT,
                guid        TEXT,
                full_adress TEXT,
                street      TEXT,
                building_no TEXT,
                zip_code    TEXT,
                status      TEXT,
                geometry    GEOMETRY(Point, 2177),
                loaded_at   TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_stg_addresses_run
            ON audit.stg_addresses(run_id)
        """))

        # ── audit.stg_penalties ──
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit.stg_penalties (
                id               SERIAL,
                run_id           INTEGER REFERENCES audit.etl_log(run_id),
                source_file      TEXT,
                pen_id           BIGINT,
                date             TIMESTAMP,
                place_of_penalty TEXT,
                pen_type         TEXT,
                geometry         GEOMETRY(Point, 2177),
                loaded_at        TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_stg_penalties_run
            ON audit.stg_penalties(run_id)
        """))

        # ── audit.processed_files ──
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit.processed_files (
                id           SERIAL PRIMARY KEY,
                dag_id       TEXT NOT NULL,
                file_name    TEXT NOT NULL,
                file_hash    TEXT,
                processed_at TIMESTAMP DEFAULT NOW(),
                run_id       INTEGER,
                UNIQUE (dag_id, file_name)
            )
        """))

    log.info("ensure_tables: schemat gotowy")


# ─── Staging ─────────────────────────────────────────────────────────────────

def stage_addresses(gdf: gpd.GeoDataFrame, engine: Engine, run_id: int) -> int:
    """Zapisuje pełny snapshot adresów do audit.stg_addresses."""
    df = gdf.copy()
    df["run_id"] = run_id
    stage = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2177")
    stage.to_postgis(
        "stg_addresses", engine, schema="audit", if_exists="append", index=False,
        dtype={"geometry": Geometry("POINT", srid=2177)},
    )
    log.info("Staged %d adresów (run_id=%d)", len(stage), run_id)
    return len(stage)


def stage_penalties(
    gdf: gpd.GeoDataFrame, engine: Engine, run_id: int, source_file: str
) -> int:
    """Zapisuje kary z jednego pliku Excel do audit.stg_penalties."""
    df = gdf.copy()
    df["run_id"] = run_id
    df["source_file"] = source_file
    cols = ["run_id", "source_file", "pen_id", "date", "place_of_penalty", "pen_type", "geometry"]
    df = df[[c for c in cols if c in df.columns]].copy()
    stage = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2177")
    stage.to_postgis(
        "stg_penalties", engine, schema="audit", if_exists="append", index=False,
        dtype={"geometry": Geometry("POINT", srid=2177)},
    )
    log.info("Staged %d kar (run_id=%d, plik=%s)", len(stage), run_id, source_file)
    return len(stage)


# ─── Load ────────────────────────────────────────────────────────────────────

def upsert_addresses(engine: Engine, run_id: int) -> int:
    """Upsert z audit.stg_addresses → mined.adresses (ON CONFLICT aktualizuje)."""
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO mined.adresses
                (gml_id, guid, full_adress, street, building_no, zip_code, status, geometry)
            SELECT gml_id, guid, full_adress, street, building_no, zip_code, status, geometry
            FROM audit.stg_addresses
            WHERE run_id = :run_id
              AND gml_id IS NOT NULL
            ON CONFLICT (gml_id) DO UPDATE
                SET guid        = EXCLUDED.guid,
                    full_adress = EXCLUDED.full_adress,
                    street      = EXCLUDED.street,
                    building_no = EXCLUDED.building_no,
                    zip_code    = EXCLUDED.zip_code,
                    status      = EXCLUDED.status,
                    geometry    = EXCLUDED.geometry
        """), {"run_id": run_id})
    log.info("Upsert mined.adresses: %d wierszy", result.rowcount)
    return result.rowcount


def upsert_penalties(engine: Engine, run_id: int) -> int:
    """Upsert z audit.stg_penalties → mined.penalties (ON CONFLICT DO NOTHING)."""
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO mined.penalties (pen_id, date, place_of_penalty, pen_type, geometry)
            SELECT pen_id, date, place_of_penalty, pen_type, geometry
            FROM audit.stg_penalties
            WHERE run_id = :run_id
              AND pen_id IS NOT NULL
            ON CONFLICT (pen_id) DO NOTHING
        """), {"run_id": run_id})
    log.info("Upsert mined.penalties: %d wierszy", result.rowcount)
    return result.rowcount


def upsert_variables(df: pd.DataFrame, engine: Engine) -> int:
    """Upsert zagregowanych zmiennych → mined.variables."""
    if df.empty:
        log.info("Brak danych do upsert mined.variables")
        return 0

    rows = df.to_dict("records")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE tmp_vars_pen (
                var_id   TEXT,
                year     TIMESTAMP,
                block_id BIGINT,
                value    DOUBLE PRECISION
            ) ON COMMIT DROP
        """))
        conn.execute(
            text("INSERT INTO tmp_vars_pen VALUES (:var_id, :year, :block_id, :value)"),
            rows,
        )
        result = conn.execute(text("""
            INSERT INTO mined.variables (var_id, year, block_id, value)
            SELECT var_id, year, block_id, value FROM tmp_vars_pen
            ON CONFLICT (var_id, year, block_id) DO UPDATE
                SET value = EXCLUDED.value
        """))
    log.info("Upsert mined.variables: %d wierszy", result.rowcount)
    return result.rowcount


def upsert_metadata(engine: Engine) -> None:
    """Zapewnia wpisy w meta.var_description dla zmiennych penalty/alcohol."""
    with engine.begin() as conn:
        for meta in METADATA:
            conn.execute(text("""
                INSERT INTO meta.var_description (var_id, unit, origin, description)
                VALUES (:var_id, :unit, :origin, :description)
                ON CONFLICT (var_id) DO UPDATE
                    SET unit        = EXCLUDED.unit,
                        origin      = EXCLUDED.origin,
                        description = EXCLUDED.description
            """), meta)
    log.info("Metadane penalty/alcohol zaktualizowane w meta.var_description")


def mark_file_processed(
    engine: Engine, dag_id: str, file_name: str, file_hash: str, run_id: int
) -> None:
    """Rejestruje przetworzony plik Excel w audit.processed_files."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO audit.processed_files (dag_id, file_name, file_hash, run_id)
            VALUES (:dag_id, :file_name, :file_hash, :run_id)
            ON CONFLICT (dag_id, file_name) DO UPDATE
                SET file_hash    = EXCLUDED.file_hash,
                    run_id       = EXCLUDED.run_id,
                    processed_at = NOW()
        """), {"dag_id": dag_id, "file_name": file_name, "file_hash": file_hash, "run_id": run_id})
    log.info("Oznaczono jako przetworzone: %s", file_name)
