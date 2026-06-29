"""
Transform: Buildings
Zamiana osi XY, standaryzacja kolumn, reprojekcja do EPSG:2177.
"""

import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString

log = logging.getLogger(__name__)

COLUMN_MAP = {
    "ID_BUDYNKU":             "building_id",
    "KONDYGNACJE_NADZIEMNE":  "floors_above_ground",
    "KONDYGNACJE_PODZIEMNE":  "floors_below_ground",
}

DROP_COLS = ["gml_id", "RODZAJ"]


# ─── Helpers ────────────────────────────────────────────────────────────────

def _swap_xy(geom):
    """
    Zamienia X↔Y — WFS EGiB zwraca geometrie w EPSG:2177 z przestawionymi osiami (Y,X).
    Obsługuje Point, LineString, MultiLineString, Polygon, MultiPolygon.
    """
    if geom is None:
        return None
    t = geom.geom_type
    if t == "Point":
        return Point(geom.y, geom.x)
    elif t == "LineString":
        return LineString([(y, x) for x, y in geom.coords])
    elif t == "MultiLineString":
        return MultiLineString([[(y, x) for x, y in ln.coords] for ln in geom.geoms])
    elif t == "Polygon":
        ext = [(y, x) for x, y in geom.exterior.coords]
        inn = [[(y, x) for x, y in ring.coords] for ring in geom.interiors]
        return Polygon(ext, inn)
    elif t == "MultiPolygon":
        polys = []
        for p in geom.geoms:
            ext = [(y, x) for x, y in p.exterior.coords]
            inn = [[(y, x) for x, y in ring.coords] for ring in p.interiors]
            polys.append(Polygon(ext, inn))
        return MultiPolygon(polys)
    return geom


# ─── Transform ──────────────────────────────────────────────────────────────

def transform_buildings(raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Czyści budynki z WFS EGiB.
    - Zamienia osie XY (WFS 1.1.0 zwraca Y,X dla EPSG:2177)
    - Ustawia CRS na EPSG:2177
    - Przemianowuje kolumny wg COLUMN_MAP
    - Usuwa zbędne kolumny
    Zwraca GeoDataFrame w EPSG:2177 — spójne z urban_blocks_geom i build_perm.
    """
    gdf = raw_gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(_swap_xy)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:2177")

    gdf = gdf.rename(columns=COLUMN_MAP)
    drop = [c for c in DROP_COLS if c in gdf.columns]
    gdf = gdf.drop(columns=drop)

    if "floors_above_ground" in gdf.columns:
        gdf["floors_above_ground"] = pd.to_numeric(gdf["floors_above_ground"], errors="coerce").astype("Int64")
    if "floors_below_ground" in gdf.columns:
        gdf["floors_below_ground"] = pd.to_numeric(gdf["floors_below_ground"], errors="coerce").astype("Int64")

    keep = ["building_id", "floors_above_ground", "floors_below_ground", "geometry"]
    existing = [c for c in keep if c in gdf.columns]
    log.info("Przetransformowano %d budynków (EPSG:2177)", len(gdf))
    return gdf[existing].copy()
