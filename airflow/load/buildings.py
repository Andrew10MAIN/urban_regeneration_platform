"""
Load: Buildings
Staging, upsert do mined.buildings i agregacja zmiennych do mined.variables.
"""

import logging
import geopandas as gpd
from geoalchemy2 import Geometry
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

VAR_BUILDINGS = [
    {
        "var_id":      "bdEnvFtxx_arrt_00000000",
        "unit":        "arrt",
        "origin":      "mapa_lodz/egib",
        "description": "Total building footprint area in an urban block (m²)",
        "col":         "footprint_area",
    },
    {
        "var_id":      "bdEnvFRxx_arrt_00000000",
        "unit":        "arrt",
        "origin":      "mapa_lodz/egib",
        "description": "Total floor area (footprint × floors above ground) in an urban block (m²)",
        "col":         "floor_area",
    },
]


# ─── Schema ─────────────────────────────────────────────────────────────────

def ensure_tables(engine: Engine) -> None:
    """Tworzy / migruje mined.buildings, audit.stg_buildings, audit.stg_building_vars."""
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

        # ── audit.stg_building_vars ── tabela pomocnicza zmian zmiennych w ciągu roku
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit.stg_building_vars (
                id           SERIAL PRIMARY KEY,
                run_id       INTEGER REFERENCES audit.etl_log(run_id),
                var_id       TEXT,
                year         TIMESTAMP,
                block_id     BIGINT,
                value        DOUBLE PRECISION,
                computed_at  TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_stg_building_vars_run
            ON audit.stg_building_vars(run_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_stg_building_vars_year
            ON audit.stg_building_vars(var_id, year)
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


# ─── Zmienne per block ───────────────────────────────────────────────────────

def aggregate_buildings_to_variables(engine: Engine, run_id: int) -> int:
    """
    Liczy per block (PostGIS), normalizując przez powierzchnię kwartału:
      bdEnvFtxx_arrt_00000000 = SUM(ST_Area(budynek)) / ST_Area(kwartał)   [-]
      bdEnvFRxx_arrt_00000000 = SUM(ST_Area(budynek) × kondygnacje) / ST_Area(kwartał)  [-]

    Rok = bieżący rok kalendarzowy (zapisywany jako YYYY-01-01).
    Każdy run → audit.stg_building_vars (append).
    mined.variables → ON CONFLICT DO UPDATE (nadpisuje w bieżącym roku).
    """
    from datetime import datetime
    year_ts = f"{datetime.now().year}-01-01"

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT
                b.block_id,
                ST_Area(b.geometry)                                             AS block_area,
                COALESCE(SUM(ST_Area(bldg.geometry)), 0)                        AS footprint_area,
                COALESCE(SUM(ST_Area(bldg.geometry)
                             * COALESCE(bldg.floors_above_ground, 1)), 0)       AS floor_area
            FROM core.urban_blocks_geom b
            LEFT JOIN mined.buildings bldg
                ON ST_Within(ST_Centroid(bldg.geometry), b.geometry)
            GROUP BY b.block_id, b.geometry
        """)).fetchall()

    if not rows:
        log.warning("Brak budynków w mined.buildings — pomijam agregację zmiennych")
        return 0

    stg_rows = []
    var_rows = []
    for row in rows:
        block_area = float(row.block_area) if row.block_area and float(row.block_area) > 0 else None
        if block_area is None:
            log.warning("block_id=%d ma block_area=0 — pomijam", row.block_id)
            continue
        vals = {
            "footprint_area": float(row.footprint_area) / block_area,
            "floor_area":     float(row.floor_area)     / block_area,
        }
        for var in VAR_BUILDINGS:
            rec = {
                "var_id":   var["var_id"],
                "year":     year_ts,
                "block_id": int(row.block_id),
                "value":    vals[var["col"]],
            }
            stg_rows.append({**rec, "run_id": run_id})
            var_rows.append(rec)

    with engine.begin() as conn:
        # Tabela pomocnicza — ślad każdego runu
        conn.execute(
            text("""
                INSERT INTO audit.stg_building_vars (run_id, var_id, year, block_id, value)
                VALUES (:run_id, :var_id, :year, :block_id, :value)
            """),
            stg_rows,
        )
        # mined.variables — aktualizuje wartość w bieżącym roku
        result = conn.execute(
            text("""
                INSERT INTO mined.variables (var_id, year, block_id, value)
                VALUES (:var_id, :year, :block_id, :value)
                ON CONFLICT (var_id, year, block_id) DO UPDATE
                    SET value = EXCLUDED.value
            """),
            var_rows,
        )

    log.info(
        "Zmienne budynkowe: %d bloków × 2 zmienne → mined.variables (rok=%s, run_id=%d)",
        len(rows), year_ts, run_id,
    )
    return result.rowcount


def upsert_building_metadata(engine: Engine) -> None:
    """Zapewnia wpisy w meta.var_description dla zmiennych budynkowych."""
    with engine.begin() as conn:
        for var in VAR_BUILDINGS:
            conn.execute(text("""
                INSERT INTO meta.var_description (var_id, unit, origin, description)
                VALUES (:var_id, :unit, :origin, :description)
                ON CONFLICT (var_id) DO UPDATE
                    SET unit        = EXCLUDED.unit,
                        origin      = EXCLUDED.origin,
                        description = EXCLUDED.description
            """), var)
    log.info("Metadane zmiennych budynkowych zaktualizowane")
