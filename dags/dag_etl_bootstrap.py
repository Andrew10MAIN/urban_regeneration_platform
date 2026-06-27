"""
DAG: ETL Bootstrap
Ładuje dane źródłowe do bazy urban_db.
Odpowiednik scripts/bootstrap/load_initial_data.py — tutaj jako zadanie Airflow.
"""

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from sqlalchemy import create_engine
import pandas as pd
import geopandas as gpd
from shapely.ops import transform


def get_engine():
    conn = BaseHook.get_connection("urban_db")
    url = f"postgresql+psycopg2://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}"
    return create_engine(url)


def load_urban_blocks():
    engine = get_engine()
    df = pd.read_parquet("/opt/airflow/urban_platform/data/source/urban_blocks/core_urban_blocks.parquet")
    df.to_sql("urban_blocks", engine, schema="core", if_exists="append", index=False)


def load_urban_blocks_geom():
    engine = get_engine()
    gdf = gpd.read_file("/opt/airflow/urban_platform/data/source/urban_blocks_geom/geo_urban_blocks.shp")
    gdf = gdf.to_crs(2177)
    gdf["geometry"] = gdf["geometry"].apply(lambda g: transform(lambda x, y, z=None: (x, y), g))
    gdf.to_postgis("urban_blocks_geom", engine, schema="core", if_exists="append")


def load_legacy_vars():
    engine = get_engine()
    df = pd.read_parquet("/opt/airflow/urban_platform/data/source/legacy_variables/df_legacy_vars.parquet")
    df.to_sql("variables", engine, schema="legacy", if_exists="append", index=False)


with DAG(
    dag_id="etl_bootstrap",
    start_date=datetime(2024, 1, 1),
    schedule=None,          # uruchamiaj ręcznie
    catchup=False,
    tags=["etl", "bootstrap"],
) as dag:

    t1 = PythonOperator(task_id="load_urban_blocks",      python_callable=load_urban_blocks)
    t2 = PythonOperator(task_id="load_urban_blocks_geom", python_callable=load_urban_blocks_geom)
    t3 = PythonOperator(task_id="load_legacy_vars",       python_callable=load_legacy_vars)

    t1 >> t2 >> t3
