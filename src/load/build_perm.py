"""
Load: Building Permits
Staging, diff tracking, upsert do mined.Build_perm i mined.variables.
"""

import logging
from datetime import datetime

import pandas as pd
import geopandas as gpd
from geoalchemy2 import Geometry
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

VAR_ID       = "urVibBlPr_coun_00000000"
VAR_UNIT     = "count"
VAR_ORIGIN   = "Geoportal GUNB"
VAR_DESC     = "The number of the issued building permits in a particular urban block in a particular year"


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

def create_run_log(engine: Engine, dag_id: str = "etl_build_perm") -> int:
    """Tworzy wpis w audit.etl_log, zwraca run_id."""
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO audit.etl_log (dag_id, source, started_at, status)
                VALUES (:dag_id, 'Geoportal GUNB WFS', NOW(), 'running')
                RETURNING run_id
            """),
            {"dag_id": dag_id}
        )
        run_id = result.scalar()
    log.info("Utworzono run_id=%d", run_id)
    return run_id


def finalize_run_log(
    engine: Engine,
    run_id: int,
    rows_extracted: int,
    rows_staged: int,
    rows_new: int,
    rows_loaded: int,
    status: str = "success",
    error_msg: str = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE audit.etl_log
                SET finished_at    = NOW(),
                    rows_extracted = :rows_extracted,
                    rows_staged    = :rows_staged,
                    rows_new       = :rows_new,
                    rows_loaded    = :rows_loaded,
                    status         = :status,
                    error_msg      = :error_msg
                WHERE run_id = :run_id
            """),
            {
                "run_id": run_id,
                "rows_extracted": rows_extracted,
                "rows_staged": rows_staged,
                "rows_new": rows_new,
                "rows_loaded": rows_loaded,
                "status": status,
                "error_msg": error_msg,
            }
        )
    log.info("Zaktualizowano run_id=%d, status=%s", run_id, status)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def stage_raw(gdf: gpd.GeoDataFrame, engine: Engine, run_id: int) -> int:
    """
    Zapisuje przetransformowane dane do audit.stg_build_perm (z block_id).
    Zwraca liczbę wierszy.
    """
    cols = ["build_perm_id", "build_plot_no", "date", "description", "geometry", "block_id"]
    df = gdf[[c for c in cols if c in gdf.columns]].copy()
    df = df.rename(columns={"date": "issue_date"})
    df["run_id"] = run_id

    gdf_stage = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2177")
    gdf_stage.to_postgis(
        "stg_build_perm",
        engine,
        schema="audit",
        if_exists="append",
        index=False,
        dtype={"geometry": Geometry("POINT", srid=2177)},
    )
    log.info("Staged %d rekordów dla run_id=%d", len(df), run_id)
    return len(df)


def compute_diff(engine: Engine, run_id: int) -> int:
    """
    Liczy nowe rekordy: te które są w staging dla tego run_id
    ale nie istnieją jeszcze w mined.Build_perm.
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*) FROM audit.stg_build_perm s
                WHERE s.run_id = :run_id
                  AND NOT EXISTS (
                    SELECT 1 FROM mined."Build_perm" b
                    WHERE b.build_perm_id = s.build_perm_id
                      AND b.build_plot_no = s.build_plot_no
                  )
            """),
            {"run_id": run_id}
        )
        new_count = result.scalar()
    log.info("Nowych rekordów względem mined.Build_perm: %d", new_count)
    return new_count


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def upsert_build_perm(gdf: gpd.GeoDataFrame, engine: Engine) -> int:
    """
    Upsert do mined.Build_perm.
    ON CONFLICT (build_perm_id, build_plot_no) DO NOTHING — pozwolenia nie zmieniają się.
    """
    # Filtr: wymagamy nie-null PK
    gdf = gdf[gdf["build_perm_id"].notna() & gdf["build_plot_no"].notna()].copy()

    from geoalchemy2.shape import from_shape

    rows = []
    for _, row in gdf.iterrows():
        rows.append({
            "build_perm_id": str(row["build_perm_id"]),
            "build_plot_no": str(row["build_plot_no"]),
            "block_id":      int(row["block_id"]),
            "date":          row["date"],
            "description":   row.get("description"),
            "geom_wkt":      row["geometry"].wkt,
        })

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE tmp_build_perm (
                build_perm_id TEXT,
                build_plot_no TEXT,
                block_id      BIGINT,
                date          TIMESTAMP,
                description   TEXT,
                geom_wkt      TEXT
            ) ON COMMIT DROP
        """))

        conn.execute(
            text("INSERT INTO tmp_build_perm VALUES (:build_perm_id, :build_plot_no, :block_id, :date, :description, :geom_wkt)"),
            rows
        )

        result = conn.execute(text("""
            INSERT INTO mined."Build_perm" (build_perm_id, build_plot_no, block_id, date, description, geometry)
            SELECT
                build_perm_id,
                build_plot_no,
                block_id,
                date,
                description,
                ST_SetSRID(ST_GeomFromText(geom_wkt), 2177)
            FROM tmp_build_perm
            ON CONFLICT (build_perm_id, build_plot_no) DO NOTHING
        """))
        rows_inserted = result.rowcount

    log.info("Załadowano %d nowych rekordów do mined.Build_perm", rows_inserted)
    return rows_inserted


def aggregate_variables(engine: Engine, run_id: int) -> pd.DataFrame:
    """
    Zlicza pozwolenia per block_id per year ze staging (audit.stg_build_perm).
    Dzięki temu liczy też rekordy z null build_perm_id (2020-2025).
    Bloki bez pozwoleń w danym roku dostają value=0.
    """
    df = pd.read_sql(
        """
        SELECT
            b.block_id,
            y.year,
            COALESCE(p.cnt, 0) AS value
        FROM core.urban_blocks_geom b
        CROSS JOIN (
            SELECT DISTINCT DATE_TRUNC('year', issue_date) AS year
            FROM audit.stg_build_perm
            WHERE issue_date IS NOT NULL AND run_id = %(run_id)s
        ) y
        LEFT JOIN (
            SELECT
                block_id,
                DATE_TRUNC('year', issue_date) AS year,
                COUNT(*) AS cnt
            FROM audit.stg_build_perm
            WHERE issue_date IS NOT NULL
              AND block_id IS NOT NULL
              AND run_id = %(run_id)s
            GROUP BY block_id, DATE_TRUNC('year', issue_date)
        ) p ON p.block_id = b.block_id AND p.year = y.year
        ORDER BY b.block_id, y.year
        """,
        engine,
        params={"run_id": run_id}
    )
    df["var_id"] = VAR_ID
    df["year"] = pd.to_datetime(df["year"])
    log.info("Zagregowano %d wierszy (block_id × year, z zerami)", len(df))
    return df[["var_id", "year", "block_id", "value"]]


def upsert_variables(df: pd.DataFrame, engine: Engine) -> int:
    """
    Upsert do mined.variables.
    ON CONFLICT aktualizuje value (count może się zmienić przy nowych danych).
    """
    rows = df.to_dict("records")

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE tmp_variables (
                var_id   TEXT,
                year     TIMESTAMP,
                block_id BIGINT,
                value    DOUBLE PRECISION
            ) ON COMMIT DROP
        """))
        conn.execute(
            text("INSERT INTO tmp_variables VALUES (:var_id, :year, :block_id, :value)"),
            rows
        )
        result = conn.execute(text("""
            INSERT INTO mined.variables (var_id, year, block_id, value)
            SELECT var_id, year, block_id, value FROM tmp_variables
            ON CONFLICT (var_id, year, block_id) DO UPDATE
                SET value = EXCLUDED.value
        """))
        rows_upserted = result.rowcount

    log.info("Upsert mined.variables: %d wierszy", rows_upserted)
    return rows_upserted


def upsert_metadata(engine: Engine) -> None:
    """Zapewnia istnienie wpisu w meta.var_description."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO meta.var_description (var_id, unit, origin, description)
                VALUES (:var_id, :unit, :origin, :description)
                ON CONFLICT (var_id) DO UPDATE
                    SET unit        = EXCLUDED.unit,
                        origin      = EXCLUDED.origin,
                        description = EXCLUDED.description
            """),
            {
                "var_id":      VAR_ID,
                "unit":        VAR_UNIT,
                "origin":      VAR_ORIGIN,
                "description": VAR_DESC,
            }
        )
    log.info("Metadane %s zaktualizowane w meta.var_description", VAR_ID)
