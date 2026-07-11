"""
Transform: OSM data
- POI z Geofabrik → filtrowanie fclass + spatial join z blokami
- Poly (parking Geofabrik + scrub OSMnx) → spatial join + obszar
- Agregacja POI count i poly area_ratio → mined.variables DataFrames
"""

import logging
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)


def to_mined_year(d) -> pd.Timestamp:
    """
    Przelicza datę na rok referencyjny dla mined.variables (zawsze YYYY-01-01).

    Reguła:
      - data = dokładnie YYYY-01-01  →  (YYYY-1)-01-01   (dane z początku roku
        reprezentują stan końca poprzedniego roku)
      - data = dowolna inna          →  YYYY-01-01

    Przykłady:
      2015-12-31  →  2015-01-01
      2016-01-01  →  2015-01-01
      2026-07-11  →  2026-01-01
      2026-12-31  →  2026-01-01
      2027-01-01  →  2026-01-01
    """
    ts = pd.Timestamp(d)
    if ts.month == 1 and ts.day == 1:
        return pd.Timestamp(f"{ts.year - 1}-01-01")
    return pd.Timestamp(f"{ts.year}-01-01")


# ── Listy fclass ──────────────────────────────────────────────────────────────

SCB_LIST = [
    "arts_centre", "bakery", "bank", "bar", "beauty_shop", "beverages",
    "bicycle_shop", "biergarten", "book_shop", "butcher", "cafe",
    "car_dealership", "car_wash", "chemist", "cinema", "clinic", "clothes",
    "computer_shop", "convenience", "dentist", "department_store", "doctors",
    "fast_food", "florist", "furniture_shop", "gift_shop", "greengrocer",
    "guest_house", "hairdresser", "hostel", "hotel", "motel", "jeweler",
    "kiosk", "laundry", "mobile_phone_shop", "newsagent", "nightclub",
    "optician", "outdoor_shop", "pharmacy", "pub", "restaurant", "shoe_shop",
    "sports_centre", "stationery", "supermarket", "swimming_pool", "toy_shop",
    "travel_agent", "veterinary", "video_shop",
]
AOUT_LIST   = ["convenience", "restaurant", "pub", "bar", "nightclub", "beverages"]
SCH_LIST    = ["school"]
KIND_LIST   = ["kindergarten"]
PLGRND_LIST = ["playground"]
SCRUB_LIST  = ["scrub"]
PARKING_LIST = ["parking"]

ALL_POI_FCLASSES  = list(set(SCB_LIST + SCH_LIST + KIND_LIST + PLGRND_LIST))
ALL_POLY_FCLASSES = SCRUB_LIST + PARKING_LIST

# var_id → lista fclass
DICT_OSM_POI = {
    "urVibSCBx_coun_00000000": SCB_LIST,
    "urVibAOut_coun_00000000": AOUT_LIST,
    "bdEnvPlgr_coun_00000000": PLGRND_LIST,
    "bdEnvSchl_coun_00000000": SCH_LIST,
    "bdEnvKnrg_coun_00000000": KIND_LIST,
}
DICT_OSM_POLY = {
    "bdEnvScrb_arrt_00000000": SCRUB_LIST,
    "bdEnvCaPr_arrt_00000000": PARKING_LIST,
}


# ── Transform ─────────────────────────────────────────────────────────────────

def transform_poi(
    poi_shp_path: Path,
    blocks_gdf: gpd.GeoDataFrame,
    run_date: date,
) -> gpd.GeoDataFrame:
    """
    Wczytuje POI z Geofabrik, filtruje fclass, spatial join z blokami.
    Zwraca GDF: osm_id, fclass, date, block_id, geometry (EPSG:2177).
    """
    gdf = gpd.read_file(poi_shp_path)
    gdf = gdf.to_crs(blocks_gdf.crs)
    gdf = gdf[gdf["fclass"].isin(ALL_POI_FCLASSES)].copy()
    gdf["date"]   = pd.Timestamp(run_date)
    gdf["osm_id"] = pd.to_numeric(gdf["osm_id"], errors="coerce")
    gdf = gdf.dropna(subset=["osm_id"])
    gdf["osm_id"] = gdf["osm_id"].astype("int64")

    joined = gpd.sjoin(
        gdf[["osm_id", "fclass", "date", "geometry"]],
        blocks_gdf[["block_id", "geometry"]],
        how="left", predicate="within",
    )
    result = joined[joined["block_id"].notna()].drop(columns=["index_right"]).copy()
    result["block_id"] = result["block_id"].astype("int64")

    log.info("POI transform: %d rekordów (z block_id)", len(result))
    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:2177")


def transform_poly(
    parking_shp_path: Path,
    scrub_shp_path: Path,
    blocks_gdf: gpd.GeoDataFrame,
    run_date: date,
) -> gpd.GeoDataFrame:
    """
    Łączy parking i scrubs z plików Geofabrik, spatial join z blokami.
    Zwraca GDF: osm_id, fclass, date, block_id, geometry (EPSG:2177).
    """
    # Parking (z gis_osm_traffic_a_free_1.shp)
    parking = gpd.read_file(parking_shp_path)
    parking = parking[parking["fclass"] == "parking"].copy()
    parking = parking.to_crs(blocks_gdf.crs)
    parking["date"]   = pd.Timestamp(run_date)
    parking["osm_id"] = pd.to_numeric(parking["osm_id"], errors="coerce")
    parking = parking.dropna(subset=["osm_id"])
    parking["osm_id"] = parking["osm_id"].astype("int64")

    # Scrubs (z gis_osm_landuse_a_free_1.shp, fclass='scrub')
    scrub = gpd.read_file(scrub_shp_path)
    scrub = scrub[scrub["fclass"] == "scrub"].copy()
    scrub = scrub.to_crs(blocks_gdf.crs)
    scrub["date"]   = pd.Timestamp(run_date)
    scrub["osm_id"] = pd.to_numeric(scrub["osm_id"], errors="coerce")
    scrub = scrub.dropna(subset=["osm_id"])
    scrub["osm_id"] = scrub["osm_id"].astype("int64")

    combined = gpd.GeoDataFrame(
        pd.concat(
            [
                parking[["osm_id", "fclass", "date", "geometry"]],
                scrub[["osm_id", "fclass", "date", "geometry"]],
            ],
            ignore_index=True,
        ),
        geometry="geometry",
        crs="EPSG:2177",
    )

    joined = gpd.sjoin(
        combined,
        blocks_gdf[["block_id", "geometry"]],
        how="left", predicate="intersects",
    )
    result = joined[joined["block_id"].notna()].drop(columns=["index_right"]).copy()
    result["block_id"] = result["block_id"].astype("int64")

    log.info("Poly transform: %d rekordów (z block_id)", len(result))
    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:2177")


# ── Agregacja do mined.variables ──────────────────────────────────────────────

def aggregate_poi_to_variables(
    poi_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    run_date: date,
) -> pd.DataFrame:
    """Count POI per block dla każdego var_id. Brakujące bloki → 0."""
    year_ts = to_mined_year(run_date)
    blocks_df = blocks_gdf[["block_id"]].drop_duplicates()

    rows = []
    for var_id, fclasses in DICT_OSM_POI.items():
        subset = poi_gdf[poi_gdf["fclass"].isin(fclasses)]
        agg = (
            subset.groupby("block_id").size()
            .reset_index(name="value")
        )
        merged = blocks_df.merge(agg, on="block_id", how="left")
        merged["value"]  = merged["value"].fillna(0).astype(float)
        merged["var_id"] = var_id
        merged["year"]   = year_ts
        rows.append(merged[["var_id", "year", "block_id", "value"]])

    return pd.concat(rows, ignore_index=True)


def aggregate_poly_to_variables(
    poly_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    run_date: date,
) -> pd.DataFrame:
    """SUM(area_poly) / block_area per block dla każdego var_id. Brakujące → 0."""
    year_ts = to_mined_year(run_date)

    poly = poly_gdf.copy()
    poly["area"] = poly.geometry.area  # m² — EPSG:2177

    blocks = blocks_gdf.copy()
    blocks["block_area"] = blocks.geometry.area

    rows = []
    for var_id, fclasses in DICT_OSM_POLY.items():
        subset = poly[poly["fclass"].isin(fclasses)]
        agg = (
            subset.groupby("block_id")["area"].sum()
            .reset_index(name="poly_area")
        )
        merged = blocks[["block_id", "block_area"]].merge(agg, on="block_id", how="left")
        merged["poly_area"] = merged["poly_area"].fillna(0)
        merged["value"]     = merged["poly_area"] / merged["block_area"]
        merged["var_id"]    = var_id
        merged["year"]      = year_ts
        rows.append(merged[["var_id", "year", "block_id", "value"]])

    return pd.concat(rows, ignore_index=True)
