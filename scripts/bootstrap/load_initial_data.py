"""
Bootstrap: pełne załadowanie danych startowych.

Uruchamiać Z KATALOGU PROJEKTU:
    cd C:\\Users\\andre\\Desktop\\IDS\\02_VS_code\\00_urban_regeneration_platform
    python scripts/bootstrap/load_initial_data.py

Idempotentny — bezpieczne do ponownego uruchomienia.
Czyści dane i ładuje od nowa (NIE dotyka tabel audit/etl_log).

Ładuje:
  1. core.urban_blocks
  2. core.urban_blocks_geom
  3. regeneration.actions
  4. mined.variables       ← legacy zmienne (z parquet)
  5. meta.var_description  ← metadane zmiennych (z parquet)
  6. mined.variables       ← Census 2021 socVrPopt_coun_00000000
  7. meta.var_description  ← metadane Census
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.ops import transform
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://urban_user:urban_password@localhost:5433/urban_db"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

print("=" * 60)
print("Bootstrap: start")
print("=" * 60)

# ─── Czyszczenie (kolejność ważna — FK constraints) ──────────────
print("\n[0/7] Czyszczenie tabel...")
with engine.begin() as conn:
    conn.execute(text("DELETE FROM mined.variables"))
    conn.execute(text('DELETE FROM mined."Build_perm"'))
    conn.execute(text("DELETE FROM mined.adresses"))
    conn.execute(text("DELETE FROM mined.penalties"))
    conn.execute(text("DELETE FROM mined.buildings"))
    conn.execute(text("DELETE FROM meta.var_description"))
    conn.execute(text("DELETE FROM regeneration.actions"))
    conn.execute(text("DELETE FROM core.urban_blocks_geom"))
    conn.execute(text("DELETE FROM core.urban_blocks"))
print("      → tabele wyczyszczone")

# ─── 1. core.urban_blocks ───────────────────────────────────────
print("\n[1/7] core.urban_blocks...")
df_blocks = pd.read_parquet("data/source/urban_blocks/core_urban_blocks.parquet")
for col in ["treated_all", "treated_d1nq", "treated_1nq"]:
    if col in df_blocks.columns:
        df_blocks[col] = df_blocks[col].astype(bool)
df_blocks.to_sql("urban_blocks", engine, schema="core", if_exists="append", index=False)
print(f"      → {len(df_blocks)} wierszy")

# ─── 2. core.urban_blocks_geom ──────────────────────────────────
print("\n[2/7] core.urban_blocks_geom...")
gdf_geom = gpd.read_file("data/source/urban_blocks_geom/geo_urban_blocks.shp")
gdf_geom = gdf_geom.to_crs(2177)
gdf_geom["geometry"] = gdf_geom["geometry"].apply(
    lambda g: transform(lambda x, y, z=None: (x, y), g)
)
gdf_geom.to_postgis("urban_blocks_geom", engine, schema="core", if_exists="append")
print(f"      → {len(gdf_geom)} bloków")

# ─── 3. regeneration.actions ────────────────────────────────────
print("\n[3/7] regeneration.actions...")
gdf_regen = gpd.read_file("data/source/regeneration_actions/regeneration_actions.shp")
gdf_regen = gdf_regen.to_crs(2177)
gdf_regen.columns = gdf_regen.columns.str.lower()
gdf_regen = gdf_regen.rename(columns={
    "regen_id":   "reg_id",
    "regen_type": "type",
    "regen_star": "started_at",
    "regen_end":  "finished_at",
    "price_pln":  "costs",
})
keep = ["reg_id", "block_id", "type", "started_at", "finished_at", "costs", "geometry"]
gdf_regen = gdf_regen[[c for c in keep if c in gdf_regen.columns]].copy()
if "costs" in gdf_regen.columns:
    gdf_regen["costs"] = pd.to_numeric(gdf_regen["costs"], errors="coerce")
if "reg_id" in gdf_regen.columns:
    gdf_regen["reg_id"] = pd.to_numeric(gdf_regen["reg_id"], errors="coerce")
    gdf_regen = gdf_regen.dropna(subset=["reg_id"])
    gdf_regen["reg_id"] = gdf_regen["reg_id"].astype(int)
gdf_regen = gpd.GeoDataFrame(gdf_regen, geometry="geometry", crs=gdf_regen.crs)
gdf_regen["geometry"] = gdf_regen["geometry"].apply(
    lambda g: transform(lambda x, y, z=None: (x, y), g)
)
gdf_regen.to_postgis("actions", engine, schema="regeneration", if_exists="append")
print(f"      → {len(gdf_regen)} wierszy")

# ─── 4. mined.variables ← legacy ────────────────────────────────
print("\n[4/7] mined.variables ← legacy parquet...")
df_legacy = pd.read_parquet("data/source/legacy_variables/df_legacy_vars.parquet")
df_legacy = df_legacy[["var_id", "year", "block_id", "value"]].copy()
df_legacy["year"]     = pd.to_datetime(df_legacy["year"])
df_legacy["block_id"] = df_legacy["block_id"].astype(int)
df_legacy["value"]    = df_legacy["value"].astype(float)
with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO mined.variables (var_id, year, block_id, value)
            VALUES (:var_id, :year, :block_id, :value)
            ON CONFLICT (var_id, year, block_id) DO UPDATE SET value = EXCLUDED.value
        """),
        df_legacy.to_dict("records"),
    )
print(f"      → {len(df_legacy)} wierszy → mined.variables")

# ─── 5. meta.var_description ────────────────────────────────────
print("\n[5/7] meta.var_description ← parquet...")
df_meta = pd.read_parquet("data/source/legacy_variables/meta_var.parquet")
with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO meta.var_description (var_id, unit, origin, description)
            VALUES (:var_id, :unit, :origin, :description)
            ON CONFLICT (var_id) DO UPDATE
                SET unit = EXCLUDED.unit, origin = EXCLUDED.origin,
                    description = EXCLUDED.description
        """),
        df_meta.to_dict("records"),
    )
print(f"      → {len(df_meta)} metadanych")

# ─── 6. mined.variables ← Census 2021 ───────────────────────────
print("\n[6/7] Census 2021 — populacja per block...")

CENSUS_GRID_PATH = "data/source/poptot_grid125_census_2021/grid125poptot.geojson"

ldz_grid   = gpd.read_file(CENSUS_GRID_PATH)
ldz_blocks = gpd.read_postgis(
    "SELECT block_id, geometry FROM core.urban_blocks_geom",
    engine, geom_col="geometry",
)

lodz = ox.geocode_to_gdf("Łódź, Poland")
parks = ox.features_from_polygon(lodz.geometry.iloc[0], {"leisure": "park"})
parks_lodz = parks.reset_index().to_crs(ldz_blocks.crs)
parks_lodz = parks_lodz[parks_lodz["name"].isin([
    "Park Staromiejski", "Park Helenów", "Park Źródliska II",
    "Park Źródliska I", "Park im. ks. Józefa Poniatowskiego",
    "Park nad Jasieniem", "Park im. Legionów",
])]

join_park = gpd.sjoin(
    ldz_blocks[["block_id", "geometry"]],
    parks_lodz[["geometry"]],
    how="left", predicate="intersects",
)
blocks_with_park = join_park.loc[~join_park["index_right"].isna(), "block_id"].unique()
ldz_blocks["is_park"] = ldz_blocks["block_id"].isin(blocks_with_park).astype(int)

ldz_blocks_proj = (
    ldz_blocks.to_crs(ldz_grid.crs) if ldz_grid.crs != ldz_blocks.crs else ldz_blocks
)
ldz_grid_split = gpd.overlay(
    ldz_grid, ldz_blocks_proj[["block_id", "is_park", "geometry"]], how="identity"
)
ldz_grid_split2 = ldz_grid_split[~ldz_grid_split["block_id"].isna()].copy()
ldz_grid_split2["block_id"] = ldz_grid_split2["block_id"].astype(int)
ldz_grid_split2["is_park"]  = ldz_grid_split2["is_park"].astype(int)
ldz_grid_split2["POP_split"] = (
    ldz_grid_split2.area / 15625 * ldz_grid_split2["TOT"]
).fillna(0)

code_total_pop = ldz_grid_split2.groupby("CODE")["POP_split"].transform("sum")
code_nblocks   = ldz_grid_split2.groupby("CODE")["block_id"].transform("nunique")
pop_alloc = ldz_grid_split2["POP_split"].where(
    ldz_grid_split2["is_park"] == 0, code_total_pop / code_nblocks
)
population_by_block = (
    ldz_grid_split2.assign(pop_alloc=pop_alloc)
    .groupby("block_id", as_index=False)["pop_alloc"].sum()
    .rename(columns={"pop_alloc": "population"})
)
population_by_block["population"] = np.floor(population_by_block["population"]).astype(int)

with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO mined.variables (var_id, year, block_id, value)
            VALUES (:var_id, :year, :block_id, :value)
            ON CONFLICT (var_id, year, block_id) DO UPDATE SET value = EXCLUDED.value
        """),
        [
            {
                "var_id":   "socVrPopt_coun_00000000",
                "year":     "2021-01-01",
                "block_id": int(r["block_id"]),
                "value":    float(r["population"]),
            }
            for _, r in population_by_block.iterrows()
        ],
    )
print(f"      → {len(population_by_block)} bloków → mined.variables (rok=2021-01-01)")

# ─── 7. meta.var_description ← Census ───────────────────────────
print("\n[7/7] meta.var_description ← Census 2021...")
with engine.begin() as conn:
    conn.execute(text("""
        INSERT INTO meta.var_description (var_id, unit, origin, description)
        VALUES ('socVrPopt_coun_00000000', 'count', 'Census 2021',
                'Total population Census 2021')
        ON CONFLICT (var_id) DO UPDATE
            SET unit = EXCLUDED.unit, origin = EXCLUDED.origin,
                description = EXCLUDED.description
    """))
print("      → socVrPopt_coun_00000000 → meta.var_description")

print("\n" + "=" * 60)
print("Bootstrap: DONE")
print("=" * 60)
