"""
DB data loading with Streamlit caching.
All heavy queries run once; results are cached for the session lifetime.
"""

from __future__ import annotations

import os
import pandas as pd
import geopandas as gpd
import streamlit as st
from sqlalchemy import create_engine, Engine


# ── Engine ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine() -> Engine:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://urban_user:urban_password@postgis:5432/urban_db",
    )
    return create_engine(url, pool_pre_ping=True)


def _qualify(table: str) -> str:
    schema, name = table.split(".", 1)
    return f'{schema}."{name}"'


# ── Core tables ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading variables…")
def load_mined_variables() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM mined.variables", get_engine())


@st.cache_data(ttl=300, show_spinner="Loading urban blocks…")
def load_urban_blocks_ng() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM core.urban_blocks", get_engine())


@st.cache_data(ttl=300, show_spinner="Loading block geometries…")
def load_urban_blocks_gdf() -> gpd.GeoDataFrame:
    return gpd.read_postgis(
        "SELECT * FROM core.urban_blocks_geom",
        get_engine(), geom_col="geometry",
    )


@st.cache_data(ttl=300, show_spinner="Loading variable descriptions…")
def load_meta() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM meta.var_description", get_engine())


# ── Results tables ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner="Loading models…")
def load_models() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM results.models", get_engine())


@st.cache_data(ttl=60, show_spinner="Loading uplifts…")
def load_uplifts() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM results.uplifts", get_engine())


@st.cache_data(ttl=60, show_spinner="Loading features…")
def load_features() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM results.features", get_engine())


@st.cache_data(ttl=60, show_spinner="Loading optimization…")
def load_optimization_summary() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM results.optimization_summary", get_engine())


@st.cache_data(ttl=60, show_spinner="Loading optimization uplifts…")
def load_uplifts_optimization() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM results.uplifts_optimization", get_engine())


@st.cache_data(ttl=60, show_spinner="Loading predicted prices…")
def load_predicted_prices() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM results.predicted_reg_prices", get_engine())


# ── Point / spatial layers ───────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading OSM points…")
def load_osm_poi() -> gpd.GeoDataFrame:
    return gpd.read_postgis(
        f'SELECT * FROM {_qualify("osm.poi")}',
        get_engine(), geom_col="geometry",
    )


@st.cache_data(ttl=300, show_spinner="Loading app prices…")
def load_app_prices() -> gpd.GeoDataFrame:
    return gpd.read_postgis(
        f'SELECT * FROM {_qualify("mined.app_prices")}',
        get_engine(), geom_col="geometry",
    )


@st.cache_data(ttl=300, show_spinner="Loading penalties…")
def load_penalties() -> gpd.GeoDataFrame:
    return gpd.read_postgis(
        f'SELECT * FROM {_qualify("mined.penalties")}',
        get_engine(), geom_col="geometry",
    )


@st.cache_data(ttl=300, show_spinner="Loading building permits…")
def load_build_perm() -> gpd.GeoDataFrame:
    return gpd.read_postgis(
        f'SELECT * FROM {_qualify("mined.Build_perm")}',
        get_engine(), geom_col="geometry",
    )


# ── Derived helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_meta_dict() -> dict[str, str]:
    df = load_meta()
    return df.set_index("var_id")["description"].to_dict()


@st.cache_data(ttl=300)
def get_available_years() -> list[int]:
    df = load_mined_variables()
    return sorted(df["year"].dt.year.unique().tolist())


@st.cache_data(ttl=300)
def get_vars_for_year(year: int) -> list[str]:
    df = load_mined_variables()
    mask = df["year"] == pd.Timestamp(f"{year}-01-01")
    return sorted(df.loc[mask, "var_id"].unique().tolist())


@st.cache_data(ttl=300)
def get_multi_year_vars() -> list[str]:
    """Variables observed in more than one year — for Parallel Trends."""
    df = load_mined_variables()
    counts = df.groupby("var_id")["year"].nunique()
    return sorted(counts[counts > 1].index.tolist())


@st.cache_data(ttl=300)
def get_urban_blocks_treatment(post_period: int) -> gpd.GeoDataFrame:
    """GDF with block_id, geometry, treated_all for the given post_period."""
    ub_ng  = load_urban_blocks_ng()
    ub_gdf = load_urban_blocks_gdf()
    ts = pd.Timestamp(f"{post_period}-01-01")
    treated = (
        ub_ng[ub_ng["year"] == ts][["block_id", "treated_all"]]
        .reset_index(drop=True)
    )
    return ub_gdf[["block_id", "geometry"]].merge(treated, on="block_id", how="left")


@st.cache_data(ttl=60)
def get_cf_models_dict() -> dict[str, str]:
    """Latest CF model per target_id → {model_id: target_id}."""
    df = load_models()
    cf = df[df["model_id"].str.startswith("CF")].copy()
    latest = (
        cf.sort_values("run_at")
        .drop_duplicates("target_id", keep="last")
    )
    return latest.set_index("model_id")["target_id"].to_dict()
