from sqlalchemy import create_engine
import pandas as pd
import geopandas as gpd
from shapely.ops import transform

engine = create_engine(
    "postgresql+psycopg2://urban_user:urban_password@localhost:5433/urban_db"
)

print("Starting bootstrap...")

df_blocks = pd.read_parquet(
    "data/source/urban_blocks/core_urban_blocks.parquet"
)

df_blocks.to_sql(
    "urban_blocks",
    engine,
    schema="core",
    if_exists="append",
    index=False
)

print("urban_blocks loaded")

gdf_geom = gpd.read_file(
    "data/source/urban_blocks_geom/geo_urban_blocks.shp"
)

gdf_geom = gdf_geom.to_crs(2177)
gdf_geom["geometry"] = gdf_geom["geometry"].apply(
    lambda g: transform(lambda x, y, z=None: (x, y), g)
)

gdf_geom.to_postgis(
    "urban_blocks_geom",
    engine,
    schema="core",
    if_exists="append"
)

print("urban_blocks_geom loaded")

gdf_regen = gpd.read_file(
    "data/source/regeneration_actions/regeneration_actions.shp"
)

gdf_regen = gdf_regen.to_crs(2177)
gdf_regen.columns = gdf_regen.columns.str.lower()

gdf_regen["geometry"] = gdf_regen["geometry"].apply(
    lambda g: transform(lambda x, y, z=None: (x, y), g)
)

gdf_regen.to_postgis(
    "actions",
    engine,
    schema="regeneration",
    if_exists="append"
)

print("regeneration_actions loaded")


df_legacy_vars = pd.read_parquet(
    "data/source/legacy_variables/df_legacy_vars.parquet"
)

df_legacy_vars.to_sql(
    "variables",
    engine,
    schema="legacy",
    if_exists="append",
    index=False
)

print("legacy_variables loaded")

df_meta_vars_description = pd.read_parquet(
    "data/source/legacy_variables/meta_var.parquet"
)

df_meta_vars_description.to_sql(
    "var_description",
    engine,
    schema="meta",
    if_exists="append",
    index=False
)

print("legacy_variables loaded")