"""
Load: OSM data
- ensure_tables: tworzy audit.stg_osm_poi / audit.stg_osm_poly (jeśli brak)
- stage_poi / stage_poly: append do staging per run
- load_poi / load_poly: DELETE current year + INSERT z staging → osm.poi / osm.poly
- upsert_osm_variables: ON CONFLICT DO UPDATE w mined.variables
- upsert_osm_metadata: ON CONFLICT DO NOTHING w meta.var_description
"""

import logging

import geopandas as gpd
import pandas as pd
from geoalchemy2 import Geometry
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

OSM_META = [
    {
        "var_id": "urVibSCBx_coun_00000000",
        "unit": "count",
        "origin": "OSM",
        "description": (
            "The number of the small catering businesses in a particular urban block "
            "in a particular year - urban vibrancy proxy"
        ),
    },
    {
        "var_id": "urVibAOut_coun_00000000",
        "unit": "count",
        "origin": "OSM",
        "description": (
            "The number of the alcohol outlets in a particular urban block "
            "in a particular year."
        ),
    },
    {
        "var_id": "bdEnvPlgr_coun_00000000",
        "unit": "count",
        "origin": "OSM",
        "description": (
            "The number of playgrounds in a particular urban block "
            "in a particular year."
        ),
    },
    {
        "var_id": "bdEnvSchl_coun_00000000",
        "unit": "count",
        "origin": "OSM",
        "description": (
            "The number of schools in a particular urban block "
            "in a particular year."
        ),
    },
    {
        "var_id": "bdEnvKnrg_coun_00000000",
        "unit": "count",
        "origin": "OSM",
        "description": (
            "The number of kindergardens in a particular urban block "
            "in a particular year."
        ),
    },
    {
        "var_id": "bdEnvScrb_arrt_00000000",
        "unit": "arrt",
        "origin": "OSM",
        "description": (
            "The relation between the area of all undevelopped sites in an urban block "
            "and the area of the urban block in a particular year."
        ),
    },
    {
        "var_id": "bdEnvCaPr_arrt_00000000",
        "unit": "arrt",
        "origin": "OSM",
        "description": (
            "The relation between the area of all car parks in an urban block "
            "and the area of the urban block in a particular year."
        ),
    },
]


# ── Schema ────────────────────────────────────────────────────────────────────

def ensure_tables(engine: Engine) -> None:
    """Tworzy audit.stg_osm_poi i audit.stg_osm_poly jeśli nie istnieją."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit.stg_osm_poi (
                id        SERIAL,
                run_id    INTEGER REFERENCES audit.etl_log(run_id),
                osm_id    BIGINT,
                fclass    TEXT,
                date      TIMESTAMP,
                block_id  BIGINT,
                geometry  GEOMETRY(Point, 2177),
                loaded_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_stg_osm_poi_run
            ON audit.stg_osm_poi(run_id)
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit.stg_osm_poly (
                id        SERIAL,
                run_id    INTEGER REFERENCES audit.etl_log(run_id),
                osm_id    BIGINT,
                fclass    TEXT,
                date      TIMESTAMP,
                block_id  BIGINT,
                geometry  GEOMETRY(Geometry, 2177),
                loaded_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_stg_osm_poly_run
            ON audit.stg_osm_poly(run_id)
        """))
    log.info("ensure_tables OSM: gotowe")


# ── Staging ───────────────────────────────────────────────────────────────────

def stage_poi(gdf: gpd.GeoDataFrame, engine: Engine, run_id: int) -> int:
    df = gdf.copy()
    df["run_id"] = run_id
    cols = ["run_id", "osm_id", "fclass", "date", "block_id", "geometry"]
    stage = gpd.GeoDataFrame(
        df[[c for c in cols if c in df.columns]],
        geometry="geometry", crs="EPSG:2177",
    )
    stage.to_postgis(
        "stg_osm_poi", engine, schema="audit", if_exists="append", index=False,
        dtype={"geometry": Geometry("POINT", srid=2177)},
    )
    log.info("Staged %d POI (run_id=%d)", len(stage), run_id)
    return len(stage)


def stage_poly(gdf: gpd.GeoDataFrame, engine: Engine, run_id: int) -> int:
    df = gdf.copy()
    df["run_id"] = run_id
    cols = ["run_id", "osm_id", "fclass", "date", "block_id", "geometry"]
    stage = gpd.GeoDataFrame(
        df[[c for c in cols if c in df.columns]],
        geometry="geometry", crs="EPSG:2177",
    )
    stage.to_postgis(
        "stg_osm_poly", engine, schema="audit", if_exists="append", index=False,
        dtype={"geometry": Geometry(srid=2177)},
    )
    log.info("Staged %d poly (run_id=%d)", len(stage), run_id)
    return len(stage)


# ── Load → tabele docelowe ────────────────────────────────────────────────────

def load_poi(engine: Engine, run_id: int, current_year: int) -> int:
    """
    Zastępuje snapshot roku current_year w osm.poi:
      DELETE WHERE YEAR(date) = current_year
      INSERT z audit.stg_osm_poi WHERE run_id = run_id
    Dane z poprzednich lat pozostają nienaruszone.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM osm.poi
            WHERE EXTRACT(YEAR FROM date)::int = :year
        """), {"year": current_year})

        result = conn.execute(text("""
            INSERT INTO osm.poi (osm_id, fclass, date, geometry)
            SELECT osm_id, fclass, date, geometry
            FROM audit.stg_osm_poi
            WHERE run_id = :run_id
              AND osm_id IS NOT NULL
            ON CONFLICT (osm_id, date) DO NOTHING
        """), {"run_id": run_id})

    log.info("load_poi: %d wierszy dla roku %d", result.rowcount, current_year)
    return result.rowcount


def load_poly(engine: Engine, run_id: int, current_year: int) -> int:
    """
    Zastępuje snapshot roku current_year w osm.poly.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM osm.poly
            WHERE EXTRACT(YEAR FROM date)::int = :year
        """), {"year": current_year})

        result = conn.execute(text("""
            INSERT INTO osm.poly (osm_id, fclass, date, geometry)
            SELECT osm_id, fclass, date, geometry
            FROM audit.stg_osm_poly
            WHERE run_id = :run_id
              AND osm_id IS NOT NULL
            ON CONFLICT (osm_id, date) DO NOTHING
        """), {"run_id": run_id})

    log.info("load_poly: %d wierszy dla roku %d", result.rowcount, current_year)
    return result.rowcount


# ── Variables + metadata ──────────────────────────────────────────────────────

def upsert_osm_variables(df: pd.DataFrame, engine: Engine) -> int:
    if df.empty:
        log.warning("Brak danych do upsert mined.variables (OSM)")
        return 0
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO mined.variables (var_id, year, block_id, value)
                VALUES (:var_id, :year, :block_id, :value)
                ON CONFLICT (var_id, year, block_id) DO UPDATE
                    SET value = EXCLUDED.value
            """),
            df.to_dict("records"),
        )
    log.info("Upsert mined.variables OSM: %d wierszy", result.rowcount)
    return result.rowcount


def upsert_osm_metadata(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO meta.var_description (var_id, unit, origin, description)
                VALUES (:var_id, :unit, :origin, :description)
                ON CONFLICT (var_id) DO NOTHING
            """),
            OSM_META,
        )
    log.info("Metadane OSM: %d wpisów", len(OSM_META))
