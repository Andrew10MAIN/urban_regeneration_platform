"""
Transform: App Prices
Filtruje, przemianowuje i przypisuje block_id transakcjom mieszkaniowym z WFS RCN.
"""

import logging
import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)


def transform_app_prices(
    raw_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    from_date: str,
) -> gpd.GeoDataFrame:
    """
    Parametry:
      raw_gdf    — surowe dane z WFS ms:lokale (EPSG:2180)
      blocks_gdf — core.urban_blocks_geom (EPSG:2177)
      from_date  — ISO date string, np. '2026-01-01'; filtruje transakcje od tej daty

    Zwraca GeoDataFrame w EPSG:2177 z kolumnami:
      gml_id, res_unit_id, building_id, date, floor_no, floor_area, price_gross, block_id, geometry
    """
    gdf = raw_gdf.copy()

    # 1. Filtr: tylko mieszkalne z datą i ceną
    gdf = gdf[gdf["lok_funkcja"] == "mieszkalna"].copy()
    gdf = gdf.dropna(subset=["dok_data", "lok_cena_brutto"])
    log.info("Po filtrze mieszkalne+data+cena: %d rekordów", len(gdf))

    # 2. Rename i cast
    gdf["building_id"] = gdf["lok_id_lokalu"].str.extract(r"(.*_BUD)")
    gdf["dok_data"]    = pd.to_datetime(gdf["dok_data"].str[:10])

    gdf = gdf.rename(columns={
        "lok_id_lokalu":   "res_unit_id",
        "lok_nr_kond":     "floor_no",
        "lok_pow_uzyt":    "floor_area",
        "lok_cena_brutto": "price_gross",
        "dok_data":        "date",
    })

    keep = ["gml_id", "res_unit_id", "building_id", "date",
            "floor_no", "floor_area", "price_gross", "geometry"]
    gdf = gdf[[c for c in keep if c in gdf.columns]].copy()

    gdf["floor_no"]    = pd.to_numeric(gdf["floor_no"],    errors="coerce")
    gdf["floor_area"]  = pd.to_numeric(gdf["floor_area"],  errors="coerce")
    gdf["price_gross"] = pd.to_numeric(gdf["price_gross"], errors="coerce")

    # 3. Filtr daty — tylko nowe transakcje
    gdf = gdf[gdf["date"] >= pd.Timestamp(from_date)].copy()
    log.info("Po filtrze daty (>= %s): %d rekordów", from_date, len(gdf))

    if gdf.empty:
        log.info("Brak nowych transakcji — zwracam pusty GeoDataFrame")
        return gdf

    # 4. Reprojekcja 2180 → 2177
    gdf = gdf.to_crs(blocks_gdf.crs)

    # 5. Spatial join → block_id
    joined = gpd.sjoin(
        gdf,
        blocks_gdf[["block_id", "geometry"]],
        how="left",
        predicate="within",
    )
    result = (
        joined[~joined["block_id"].isna()]
        .drop(columns=["index_right"])
        .reset_index(drop=True)
        .copy()
    )
    result["block_id"] = result["block_id"].astype(int)
    log.info("Po spatial join: %d rekordów z block_id", len(result))

    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:2177")
