"""
Load: App Prices
Staging, upsert do mined.app_prices, agregacja zmiennych do mined.variables.
"""

import logging
import geopandas as gpd
from geoalchemy2 import Geometry
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

VAR_APP_PRICE = {
    "var_id":      "socVrAS00_avrg_00000000",
    "unit":        "pln",
    "origin":      "RCN",
    "description": "average price of the appartments in urban block",
}


# ─── Schema ─────────────────────────────────────────────────────────────────

def ensure_tables(engine: Engine) -> None:
    """Tworzy / migruje mined.app_prices i audit.stg_app_prices."""
    with engine.begin() as conn:

        # mined.app_prices — migracja jeśli brak kolumny gml_id (stary schemat)
        has_gml_id = conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'mined' AND table_name = 'app_prices'
              AND column_name = 'gml_id'
        """)).fetchone()
        if not has_gml_id:
            log.info("Migracja: odtwarzam mined.app_prices z nowym schematem")
            conn.execute(text("DROP TABLE IF EXISTS mined.app_prices CASCADE"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mined.app_prices (
                gml_id      TEXT PRIMARY KEY,
                res_unit_id TEXT,
                building_id TEXT,
                date        TIMESTAMP,
                floor_no    DOUBLE PRECISION,
                floor_area  DOUBLE PRECISION,
                price_gross DOUBLE PRECISION,
                block_id    BIGINT,
                geometry    GEOMETRY(Point, 2177)
            )
        """))

        # audit.stg_app_prices
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit.stg_app_prices (
                id          SERIAL,
                run_id      INTEGER REFERENCES audit.etl_log(run_id),
                gml_id      TEXT,
                res_unit_id TEXT,
                building_id TEXT,
                date        TIMESTAMP,
                floor_no    DOUBLE PRECISION,
                floor_area  DOUBLE PRECISION,
                price_gross DOUBLE PRECISION,
                block_id    BIGINT,
                geometry    GEOMETRY(Point, 2177),
                loaded_at   TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_stg_app_prices_run
            ON audit.stg_app_prices(run_id)
        """))

    log.info("ensure_tables app_prices: schemat gotowy")


# ─── Staging ─────────────────────────────────────────────────────────────────

def stage_app_prices(gdf: gpd.GeoDataFrame, engine: Engine, run_id: int) -> int:
    """Zapisuje snapshot transakcji do audit.stg_app_prices."""
    if gdf.empty:
        log.info("Brak transakcji do staged (run_id=%d)", run_id)
        return 0

    df = gdf.copy()
    df["run_id"] = run_id
    cols = ["run_id", "gml_id", "res_unit_id", "building_id", "date",
            "floor_no", "floor_area", "price_gross", "block_id", "geometry"]
    df = df[[c for c in cols if c in df.columns]].copy()

    stage = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2177")
    stage.to_postgis(
        "stg_app_prices", engine, schema="audit", if_exists="append", index=False,
        dtype={"geometry": Geometry(srid=2177)},
    )
    log.info("Staged %d transakcji (run_id=%d)", len(stage), run_id)
    return len(stage)


# ─── Load ────────────────────────────────────────────────────────────────────

def upsert_app_prices(engine: Engine, run_id: int) -> int:
    """Upsert audit.stg_app_prices → mined.app_prices (ON CONFLICT DO NOTHING)."""
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO mined.app_prices
                (gml_id, res_unit_id, building_id, date,
                 floor_no, floor_area, price_gross, block_id, geometry)
            SELECT
                gml_id, res_unit_id, building_id, date,
                floor_no, floor_area, price_gross, block_id, geometry
            FROM audit.stg_app_prices
            WHERE run_id = :run_id
              AND gml_id IS NOT NULL
            ON CONFLICT (gml_id) DO NOTHING
        """), {"run_id": run_id})
    log.info("Upsert mined.app_prices: %d nowych wierszy (run_id=%d)", result.rowcount, run_id)
    return result.rowcount


# ─── Agregacja do mined.variables ────────────────────────────────────────────

def aggregate_prices_to_variables(engine: Engine, run_id: int) -> int:
    """
    Liczy średnią cenę/m² per block per year ze WSZYSTKICH rekordów w mined.app_prices.
    socVrAS00_avrg_00000000 → mined.variables (ON CONFLICT DO UPDATE).
    """
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT
                block_id,
                DATE_TRUNC('year', date) AS year,
                AVG(price_gross / NULLIF(floor_area, 0)) AS avg_price_per_m2
            FROM mined.app_prices
            WHERE floor_area > 0
              AND price_gross > 0
              AND block_id IS NOT NULL
            GROUP BY block_id, DATE_TRUNC('year', date)
        """)).fetchall()

    if not rows:
        log.warning("Brak danych w mined.app_prices — pomijam agregację zmiennych")
        return 0

    var_rows = [
        {
            "var_id":   VAR_APP_PRICE["var_id"],
            "year":     row.year,
            "block_id": int(row.block_id),
            "value":    float(row.avg_price_per_m2),
        }
        for row in rows
    ]

    with engine.begin() as conn:
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
        "Zmienne cen: %d wierszy → mined.variables (run_id=%d)",
        result.rowcount, run_id,
    )
    return result.rowcount


def upsert_app_price_metadata(engine: Engine) -> None:
    """Zapewnia wpis w meta.var_description dla zmiennej cen mieszkań (tylko jeśli nie ma)."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO meta.var_description (var_id, unit, origin, description)
            VALUES (:var_id, :unit, :origin, :description)
            ON CONFLICT (var_id) DO NOTHING
        """), VAR_APP_PRICE)
    log.info("Metadane %s: zaktualizowane", VAR_APP_PRICE["var_id"])
