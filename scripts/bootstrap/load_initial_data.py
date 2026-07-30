"""
Bootstrap: pełne załadowanie danych startowych.

Uruchamiać Z KATALOGU PROJEKTU:
    cd C:\\Users\\andre\\Desktop\\IDS\\02_VS_code\\00_urban_regeneration_platform
    python scripts/bootstrap/load_initial_data.py

Idempotentny — bezpieczne do ponownego uruchomienia.
Czyści dane i ładuje od nowa.

Ładuje:
   1. core.urban_blocks
   2. core.urban_blocks_geom
   3. regeneration.actions
   4. mined.variables       ← legacy zmienne (parquet)
   5. meta.var_description  ← metadane legacy
   6. mined.variables       ← Census 2021 (populacja)
   7. meta.var_description  ← metadane Census
   8. mined.app_prices      ← historyczne transakcje mieszkaniowe
   9. mined.variables       ← socVrAS00_avrg_00000000
  10. osm.poi + osm.poly    ← historyczne dane OSM
  11. mined.variables       ← zmienne OSM poi (count per block/year)
  12. mined.variables       ← zmienne OSM poly (area ratio per block/year)
  13. meta.var_description  ← metadane OSM
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.ops import transform
from sqlalchemy import create_engine, text, Boolean
from geoalchemy2 import Geometry

import os
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://urban_user:urban_password@localhost:5433/urban_db"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

print("=" * 60)
print("Bootstrap: start")
print("=" * 60)

# ─── Czyszczenie ─────────────────────────────────────────────────
print("\n[0/13] Czyszczenie tabel...")
with engine.begin() as conn:
    conn.execute(text("DELETE FROM mined.variables"))
    conn.execute(text('DELETE FROM mined."Build_perm"'))
    conn.execute(text("DELETE FROM mined.adresses"))
    conn.execute(text("DELETE FROM mined.penalties"))
    conn.execute(text("DELETE FROM mined.buildings"))
    conn.execute(text("DELETE FROM mined.app_prices"))
    conn.execute(text("DELETE FROM meta.var_description"))
    conn.execute(text("DELETE FROM regeneration.actions"))
    conn.execute(text("DELETE FROM core.urban_blocks_geom"))
    conn.execute(text("DELETE FROM core.urban_blocks"))
    conn.execute(text("DELETE FROM osm.poi"))
    conn.execute(text("DELETE FROM osm.poly"))
    # Audit — czyścimy żeby ETL mógł ponownie przetworzyć pliki
    conn.execute(text("DELETE FROM audit.processed_files"))
    conn.execute(text("DELETE FROM audit.stg_penalties"))
    conn.execute(text("DELETE FROM audit.stg_addresses"))
    conn.execute(text("DELETE FROM audit.stg_build_perm"))
    conn.execute(text("DELETE FROM audit.stg_buildings"))
    conn.execute(text("DELETE FROM audit.stg_building_vars"))
    conn.execute(text("DELETE FROM audit.stg_app_prices"))
print("      → tabele wyczyszczone")

# ─── 1. core.urban_blocks ────────────────────────────────────────
print("\n[1/13] core.urban_blocks...")
df_blocks = pd.read_parquet("data/source/urban_blocks/core_urban_blocks.parquet")
df_blocks.to_sql("urban_blocks", engine, schema="core", if_exists="append", index=False)
print(f"      → {len(df_blocks)} wierszy")

# ─── 2. core.urban_blocks_geom ───────────────────────────────────
print("\n[2/13] core.urban_blocks_geom...")
gdf_geom = gpd.read_file("data/source/urban_blocks_geom/geo_urban_blocks.shp")
gdf_geom = gdf_geom.to_crs(2177)
gdf_geom["geometry"] = gdf_geom["geometry"].apply(
    lambda g: transform(lambda x, y, z=None: (x, y), g)
)
gdf_geom.to_postgis("urban_blocks_geom", engine, schema="core", if_exists="append")
print(f"      → {len(gdf_geom)} bloków")

# ─── 3. regeneration.actions ─────────────────────────────────────
print("\n[3/13] regeneration.actions...")
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

# ─── 4. mined.variables ← legacy ─────────────────────────────────
print("\n[4/13] mined.variables ← legacy parquet...")
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

# ─── 5. meta.var_description ← legacy ───────────────────────────
print("\n[5/13] meta.var_description ← parquet...")
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

# ─── 6. mined.variables ← Census 2021 ────────────────────────────
print("\n[6/13] Census 2021 — populacja per block...")

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
print("\n[7/13] meta.var_description ← Census 2021...")
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

# ─── 8. mined.app_prices ← dane historyczne ──────────────────────
print("\n[8/13] mined.app_prices ← historyczne transakcje mieszkaniowe...")

APP_PRICES_PATH = "data/source/app_prices_historic/app_prices_2015_25.geojson"

gdf_app = gpd.read_file(APP_PRICES_PATH, driver="GeoJSON", encoding="utf-8")
gdf_app["date"] = pd.to_datetime(gdf_app["date"])

ldz_blocks_2177 = gpd.read_postgis(
    "SELECT block_id, geometry FROM core.urban_blocks_geom",
    engine, geom_col="geometry",
)
gdf_app = gdf_app.to_crs(ldz_blocks_2177.crs)

joined = gpd.sjoin(
    gdf_app,
    ldz_blocks_2177[["block_id", "geometry"]],
    how="left",
    predicate="within",
)
gdf_app = (
    joined[~joined["block_id"].isna()]
    .drop(columns=["index_right"])
    .reset_index(drop=True)
    .copy()
)
gdf_app["block_id"]    = gdf_app["block_id"].astype(int)
gdf_app["floor_no"]    = pd.to_numeric(gdf_app["floor_no"],    errors="coerce")
gdf_app["floor_area"]  = pd.to_numeric(gdf_app["floor_area"],  errors="coerce")
gdf_app["price_gross"] = pd.to_numeric(gdf_app["price_gross"], errors="coerce")

keep_cols = ["gml_id", "res_unit_id", "building_id", "date",
             "floor_no", "floor_area", "price_gross", "block_id", "geometry"]
gdf_app = gpd.GeoDataFrame(
    gdf_app[[c for c in keep_cols if c in gdf_app.columns]],
    geometry="geometry", crs="EPSG:2177",
)
gdf_app.to_postgis("app_prices", engine, schema="mined", if_exists="append", index=False)
print(f"      → {len(gdf_app)} transakcji → mined.app_prices")

# ─── 9. mined.variables ← ceny mieszkań ─────────────────────────
print("\n[9/13] mined.variables ← średnia cena/m² per block/year...")
with engine.connect() as conn:
    rows_ap = conn.execute(text("""
        SELECT
            block_id,
            DATE_TRUNC('year', date) AS year,
            AVG(price_gross / NULLIF(floor_area, 0)) AS avg_price_per_m2
        FROM mined.app_prices
        WHERE floor_area > 0 AND price_gross > 0 AND block_id IS NOT NULL
        GROUP BY block_id, DATE_TRUNC('year', date)
    """)).fetchall()

with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO mined.variables (var_id, year, block_id, value)
            VALUES (:var_id, :year, :block_id, :value)
            ON CONFLICT (var_id, year, block_id) DO UPDATE SET value = EXCLUDED.value
        """),
        [{"var_id": "socVrAS00_avrg_00000000", "year": r.year,
          "block_id": int(r.block_id), "value": float(r.avg_price_per_m2)}
         for r in rows_ap],
    )
    conn.execute(text("""
        INSERT INTO meta.var_description (var_id, unit, origin, description)
        VALUES ('socVrAS00_avrg_00000000', 'pln', 'RCN',
                'average price of the appartments in urban block')
        ON CONFLICT (var_id) DO NOTHING
    """))
print(f"      → {len(rows_ap)} wierszy → mined.variables (socVrAS00_avrg_00000000)")

# ─── 10. osm.poi + osm.poly ← historyczne dane OSM ──────────────
print("\n[10/13] osm.poi + osm.poly ← dane historyczne OSM...")

OSM_PATH = "data/source/osm_historic"

# POI
all_poi_gdf = gpd.GeoDataFrame(pd.concat([
    gpd.read_file(f"{OSM_PATH}/scb_poi.geojson",      driver="GeoJSON", encoding="utf-8"),
    gpd.read_file(f"{OSM_PATH}/school_poi.geojson",   driver="GeoJSON", encoding="utf-8"),
    gpd.read_file(f"{OSM_PATH}/kindergar_poi.geojson",driver="GeoJSON", encoding="utf-8"),
    gpd.read_file(f"{OSM_PATH}/playgrnd_poi.geojson", driver="GeoJSON", encoding="utf-8"),
], ignore_index=True), geometry="geometry", crs="EPSG:2177")

all_poi_gdf = all_poi_gdf[["osm_id", "fclass", "year", "block_id", "geometry"]].copy()
all_poi_gdf = all_poi_gdf.drop_duplicates(["osm_id", "year"])
all_poi_gdf["year"] = pd.to_datetime(all_poi_gdf["year"].astype(str) + "-12-31")
all_poi_gdf["block_id"] = pd.to_numeric(all_poi_gdf["block_id"], errors="coerce")
all_poi_gdf["count"] = 1

osm_poi_df = all_poi_gdf[["osm_id", "fclass", "year", "geometry"]].rename(
    columns={"year": "date"}
).copy()
osm_poi_gdf = gpd.GeoDataFrame(osm_poi_df, geometry="geometry", crs="EPSG:2177")
osm_poi_gdf.to_postgis(
    "poi", engine, schema="osm", if_exists="append", index=False,
    dtype={"geometry": Geometry("POINT", srid=2177)},
)
print(f"      → {len(osm_poi_gdf)} wierszy → osm.poi")

# POLY
all_poly_gdf = gpd.GeoDataFrame(pd.concat([
    gpd.read_file(f"{OSM_PATH}/car_parks_poly.geojson", driver="GeoJSON", encoding="utf-8"),
    gpd.read_file(f"{OSM_PATH}/scrubs_poly.geojson",    driver="GeoJSON", encoding="utf-8"),
], ignore_index=True), geometry="geometry", crs="EPSG:2177")

all_poly_gdf = all_poly_gdf[["osm_id", "fclass", "year", "block_id", "geometry"]].copy()
all_poly_gdf = all_poly_gdf.drop_duplicates(["osm_id", "year"])
all_poly_gdf["year"] = pd.to_datetime(all_poly_gdf["year"].astype(str) + "-12-31")
all_poly_gdf["block_id"] = pd.to_numeric(all_poly_gdf["block_id"], errors="coerce")
all_poly_gdf["area"] = all_poly_gdf.area   # m² — dane już w EPSG:2177

osm_poly_df = all_poly_gdf[["osm_id", "fclass", "year", "geometry"]].rename(
    columns={"year": "date"}
).copy()
osm_poly_gdf = gpd.GeoDataFrame(osm_poly_df, geometry="geometry", crs="EPSG:2177")
osm_poly_gdf.to_postgis(
    "poly", engine, schema="osm", if_exists="append", index=False,
    dtype={"geometry": Geometry(srid=2177)},
)
print(f"      → {len(osm_poly_gdf)} wierszy → osm.poly")

# ─── Bloki z powierzchniami (dla poly ratio) ─────────────────────
ldz_blocks_full = gpd.read_postgis(
    "SELECT block_id, geometry FROM core.urban_blocks_geom",
    engine, geom_col="geometry",
)
ldz_blocks_full["block_area"] = ldz_blocks_full.area   # m²

# ─── 11. mined.variables ← OSM poi ──────────────────────────────
print("\n[11/13] mined.variables ← OSM poi (count per block/year)...")

DICT_OSM_POI = {
    "urVibSCBx_coun_00000000": ["arts_centre","bakery","bank","bar","beaty_shop","beverages",
        "bicycle_shop","bier_garten","book_shop","butcher","cafe","car_dealership","car_wash",
        "chemist","cinema","clinic","clothes","computer_shop","convenience","dentist",
        "department_store","doctors","fast_food","florist","furniture_shop","gift_shop",
        "greengrocer","guest_house","hairdresser","hostel","hotel","motel","jeweler","kiosk",
        "laundry","mobile_phone_shop","newsagent","nightclub","optician","outdoor_shop",
        "pharmacy","pub","restaurant","shoe_shop","sports_centre","stanionery","supermerket",
        "swimming_pool","toy_shop","travel_agent","veterinary","video_shop"],
    "urVibAOut_coun_00000000": ["convenience","restaurant","pub","bar","nightclub","beverages"],
    "bdEnvPlgr_coun_00000000": ["playground"],
    "bdEnvSchl_coun_00000000": ["school"],
    "bdEnvKnrg_coun_00000000": ["kindergarten"],
}

years_poi = pd.DataFrame({"year": all_poi_gdf["year"].unique()})
blocks_years_poi = ldz_blocks_full[["block_id"]].merge(years_poi, how="cross")

poi_vars_list = []
for var_id, fclasses in DICT_OSM_POI.items():
    agg = (
        all_poi_gdf[all_poi_gdf["fclass"].isin(fclasses)]
        .groupby(["block_id", "year"])["count"].sum()
        .reset_index()
        .rename(columns={"count": "value", "year": "year"})
    )
    merged = blocks_years_poi.merge(agg, on=["block_id", "year"], how="left")
    merged["value"]  = merged["value"].fillna(0).astype(float)
    merged["var_id"] = var_id
    poi_vars_list.append(merged[["var_id", "year", "block_id", "value"]])

osm_poi_vars = pd.concat(poi_vars_list, ignore_index=True)
osm_poi_vars["block_id"] = osm_poi_vars["block_id"].astype(int)
# osm.poi przechowuje YYYY-12-31; mined.variables zawsze YYYY-01-01
osm_poi_vars["year"] = osm_poi_vars["year"].apply(
    lambda d: pd.Timestamp(f"{d.year}-01-01")
)

with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO mined.variables (var_id, year, block_id, value)
            VALUES (:var_id, :year, :block_id, :value)
            ON CONFLICT (var_id, year, block_id) DO UPDATE SET value = EXCLUDED.value
        """),
        osm_poi_vars.to_dict("records"),
    )
print(f"      → {len(osm_poi_vars)} wierszy → mined.variables (OSM poi)")

# ─── 12. mined.variables ← OSM poly ─────────────────────────────
print("\n[12/13] mined.variables ← OSM poly (area ratio per block/year)...")

DICT_OSM_POLY = {
    "bdEnvScrb_arrt_00000000": ["scrub"],
    "bdEnvCaPr_arrt_00000000": ["parking"],
}

years_poly = pd.DataFrame({"year": all_poly_gdf["year"].unique()})
blocks_years_poly = ldz_blocks_full[["block_id", "block_area"]].merge(
    years_poly, how="cross"
)

poly_vars_list = []
for var_id, fclasses in DICT_OSM_POLY.items():
    agg = (
        all_poly_gdf[all_poly_gdf["fclass"].isin(fclasses)]
        .groupby(["block_id", "year"])["area"].sum()
        .reset_index()
        .rename(columns={"area": "value"})
    )
    merged = blocks_years_poly.merge(agg, on=["block_id", "year"], how="left")
    merged["value"]  = merged["value"].fillna(0).astype(float)
    merged["value"]  = merged["value"] / merged["block_area"]   # ratio [-]
    merged["var_id"] = var_id
    poly_vars_list.append(merged[["var_id", "year", "block_id", "value"]])

osm_poly_vars = pd.concat(poly_vars_list, ignore_index=True)
osm_poly_vars["block_id"] = osm_poly_vars["block_id"].astype(int)
# osm.poly przechowuje YYYY-12-31; mined.variables zawsze YYYY-01-01
osm_poly_vars["year"] = osm_poly_vars["year"].apply(
    lambda d: pd.Timestamp(f"{d.year}-01-01")
)

with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO mined.variables (var_id, year, block_id, value)
            VALUES (:var_id, :year, :block_id, :value)
            ON CONFLICT (var_id, year, block_id) DO UPDATE SET value = EXCLUDED.value
        """),
        osm_poly_vars.to_dict("records"),
    )
print(f"      → {len(osm_poly_vars)} wierszy → mined.variables (OSM poly)")

# ─── 13. meta.var_description ← OSM ─────────────────────────────
print("\n[13/13] meta.var_description ← OSM...")

OSM_META = [
    {"var_id": "urVibSCBx_coun_00000000", "unit": "count", "origin": "OSM",
     "description": "The number of the small catering businesses in a particular urban block "
                    "in a particular year - urban vibrancy proxy"},
    {"var_id": "urVibAOut_coun_00000000", "unit": "count", "origin": "OSM",
     "description": "The number of the alcohol outlets in a particular urban block "
                    "in a particular year."},
    {"var_id": "bdEnvPlgr_coun_00000000", "unit": "count", "origin": "OSM",
     "description": "The number of playgrounds in a particular urban block in a particular year."},
    {"var_id": "bdEnvSchl_coun_00000000", "unit": "count", "origin": "OSM",
     "description": "The number of schools in a particular urban block in a particular year."},
    {"var_id": "bdEnvKnrg_coun_00000000", "unit": "count", "origin": "OSM",
     "description": "The number of kindergardens in a particular urban block in a particular year."},
    {"var_id": "bdEnvScrb_arrt_00000000", "unit": "arrt", "origin": "OSM",
     "description": "The relation between the area of all undevelopped sites in an urban block "
                    "and the area of the urban block in a particular year."},
    {"var_id": "bdEnvCaPr_arrt_00000000", "unit": "arrt", "origin": "OSM",
     "description": "The relation between the area of all car parks in an urban block "
                    "and the area of the urban block in a particular year."},
]

with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO meta.var_description (var_id, unit, origin, description)
            VALUES (:var_id, :unit, :origin, :description)
            ON CONFLICT (var_id) DO UPDATE
                SET unit = EXCLUDED.unit, origin = EXCLUDED.origin,
                    description = EXCLUDED.description
        """),
        OSM_META,
    )
print(f"      → {len(OSM_META)} metadanych OSM → meta.var_description")

print("\n" + "=" * 60)
print("Bootstrap: DONE")
print("=" * 60)
