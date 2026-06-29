"""
Load: Buildings
Staging i upsert do mined.buildings.
"""

import logging
import geopandas as gpd
from geoalchemy2 import Geometry
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)


# ─── Schema ─────────────────────────────────────────────────────────────────

def ensure_tables(engine: Engine) -> None:
    """Tworzy / migruje mined.buildings i audit.stg_buildings."""
    with engine.begin() as conn:

        # ── mined.buildings ── migracja jeśli brak kolumny building_id
        has_building_id = conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'mined' AND table_name = 'buildings'
              AND column_name = 'building_id'
        """)).fetchone()
        if not has_building_id:
            log.info("Migracja: odtwarzam mined.buildings z nowym schematem")
            conn.execute(text("DROP TABLE IF EXISTS mined.buildings CASCADE"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mined.buildings (
                building_id          TEXT PRIMARY KEY,
                floors_above_ground  INTEGER,
                floors_below_ground  INTEGER,
                geometry             GEOMETRY(Geometry, 2177)
            )
        """))

        # ── audit.stg_buildings ──
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit.stg_buildings (
                id                   SERIAL,
                run_id               INTEGER REFERENCES audit.etl_log(run_id),
                building_id          TEXT,
                floors_above_ground  INTEGER,
                floors_below_ground  INTEGER,
                geometry             GEOMETRY(Geometry, 2177),
                loaded_at            TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_stg_buildings_run
            ON audit.stg_buildings(run_id)
        """))

    log.info("ensure_tables buildings: schemat gotowy")


# ─── Staging ─────────────────────────────────────────────────────────────────

def stage_buildings(gdf: gpd.GeoDataFrame, engine: Engine, run_id: int) -> int:
    """Zapisuje snapshot budynków do audit.stg_buildings."""
    df = gdf.copy()
    df["run_id"] = run_id
    cols = ["run_id", "building_id", "floors_above_ground", "floors_below_ground", "geometry"]
    df = df[[c for c in cols if c in df.columns]].copy()
    stage = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2177")
    stage.to_postgis(
        "stg_buildings", engine, schema="audit", if_exists="append", index=False,
        dtype={"geometry": Geometry(srid=2177)},
    )
    log.info("Staged %d budynków (run_id=%d)", len(stage), run_id)
    return len(stage)


# ─── Load ────────────────────────────────────────────────────────────────────

def upsert_buildings(engine: Engine, run_id: int) -> int:
    """Upsert z audit.stg_buildings → mined.buildings (ON CONFLICT aktualizuje)."""
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO mined.buildings
                (building_id, floors_above_ground, floors_below_ground, geometry)
            SELECT building_id, floors_above_ground, floors_below_ground, geometry
            FROM audit.stg_buildings
            WHERE run_id = :run_id
              AND building_id IS NOT NULL
            ON CONFLICT (building_id) DO UPDATE
                SET floors_above_ground = EXCLUDED.floors_above_ground,
                    floors_below_ground = EXCLUDED.floors_below_ground,
                    geometry            = EXCLUDED.geometry
        """), {"run_id": run_id})
    log.info("Upsert mined.buildings: %d wierszy", result.rowcount)
    return result.rowcount
