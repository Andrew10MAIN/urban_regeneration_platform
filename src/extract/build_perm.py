"""
Extract: Building Permits
Źródło: Geoportal GUNB WFS
"""

import logging
import pandas as pd
import geopandas as gpd
from owslib.wfs import WebFeatureService
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

WFS_URL = "https://mapy.geoportal.gov.pl/wss/ext/GlownyUrzadNadzoruBudowlanego/RWDZ-WFS"
WFS_VERSION = "2.0.0"
SRSNAME = "EPSG:2180"

YEAR_LAYERS = {
    2020: "ms:pozwolenia_2020",
    2021: "ms:pozwolenia_2021",
    2022: "ms:pozwolenia_2022",
    2023: "ms:pozwolenia_2023",
    2024: "ms:pozwolenia_2024",
    2025: "ms:pozwolenia_2025",
}
OTHER_LAYER = "ms:pozwolenia_pozostale"
OTHER_YEARS = [2016, 2017, 2018, 2019]


def get_bbox(engine: Engine) -> tuple:
    """Pobiera bbox urban_blocks_geom z bazy (EPSG:2180)."""
    gdf = gpd.read_postgis(
        "SELECT geometry FROM core.urban_blocks_geom",
        engine,
        geom_col="geometry"
    )
    gdf_2180 = gdf.to_crs("EPSG:2180")
    minx, miny, maxx, maxy = gdf_2180.total_bounds
    log.info("BBox (EPSG:2180): %.2f %.2f %.2f %.2f", minx, miny, maxx, maxy)
    return (minx, miny, maxx, maxy)


def _connect_wfs() -> WebFeatureService:
    log.info("Łączenie z WFS: %s", WFS_URL)
    return WebFeatureService(url=WFS_URL, version=WFS_VERSION, timeout=120)


def _fetch_layer(wfs: WebFeatureService, layer: str, bbox: tuple) -> gpd.GeoDataFrame:
    # WFS 2.0.0 wymaga bbox jako 5-elementowej krotki: (minx, miny, maxx, maxy, CRS)
    bbox_with_crs = bbox + ("urn:ogc:def:crs:EPSG::2180",)
    response = wfs.getfeature(typename=layer, bbox=bbox_with_crs)
    gdf = gpd.read_file(response)
    if gdf.crs is None:
        gdf = gdf.set_crs(SRSNAME)
    return gdf


def fetch_year_layers(wfs: WebFeatureService, bbox: tuple) -> gpd.GeoDataFrame:
    """Pobiera warstwy roczne 2020–2025."""
    gdfs = []
    for year, layer in YEAR_LAYERS.items():
        log.info("Pobieranie warstwy %s ...", layer)
        gdf = _fetch_layer(wfs, layer, bbox)
        gdf["year"] = year
        gdfs.append(gdf)
        log.info("  → %d rekordów", len(gdf))
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=SRSNAME)


def fetch_other_years(wfs: WebFeatureService, bbox: tuple) -> gpd.GeoDataFrame:
    """Pobiera pozwolenia z warstwy 'pozostale' dla lat 2016–2019."""
    log.info("Pobieranie warstwy %s ...", OTHER_LAYER)
    gdf = _fetch_layer(wfs, OTHER_LAYER, bbox)
    if gdf.crs is None:
        gdf = gdf.set_crs(SRSNAME)
    gdf["data_wplywu_wniosku_do_urzedu"] = pd.to_datetime(
        gdf["data_wplywu_wniosku_do_urzedu"]
    )

    gdfs = []
    for year in OTHER_YEARS:
        start = pd.Timestamp(f"{year}-01-01")
        end   = pd.Timestamp(f"{year}-12-31")
        if year == 2019:
            mask = gdf["data_wplywu_wniosku_do_urzedu"] >= start
        else:
            mask = (
                (gdf["data_wplywu_wniosku_do_urzedu"] >= start) &
                (gdf["data_wplywu_wniosku_do_urzedu"] <= end)
            )
        chunk = gdf[mask].copy()
        chunk["year"] = year
        log.info("  → rok %d: %d rekordów", year, len(chunk))
        gdfs.append(chunk)

    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=SRSNAME)


def extract(engine: Engine) -> gpd.GeoDataFrame:
    """
    Pełna ekstrakcja z WFS.
    Zwraca surowy GeoDataFrame w EPSG:2180 z kolumną 'year'.
    """
    bbox = get_bbox(engine)
    wfs  = _connect_wfs()

    gdf_years = fetch_year_layers(wfs, bbox)
    gdf_other = fetch_other_years(wfs, bbox)

    gdf_all = gpd.GeoDataFrame(
        pd.concat([gdf_years, gdf_other], ignore_index=True),
        crs=SRSNAME
    )
    log.info("Łącznie wyekstrahowano: %d rekordów", len(gdf_all))
    return gdf_all
