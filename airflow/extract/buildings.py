"""
Extract: Buildings
Budynki z WFS Łódź EGiB (mapa.lodz.pl/OGC/EGiB).
"""

import logging
import geopandas as gpd

log = logging.getLogger(__name__)

BUILDINGS_WFS = (
    "https://mapa.lodz.pl/OGC/EGiB"
    "?service=WFS&version=1.1.0&request=GetFeature"
    "&typename=ms:budynki"
)


def fetch_buildings(bbox_2177: tuple) -> gpd.GeoDataFrame:
    """
    Pobiera budynki z WFS EGiB dla podanego bbox (EPSG:2177).
    Zwraca surowy GeoDataFrame (osie XY wymagają zamiany).
    """
    minx, miny, maxx, maxy = bbox_2177
    url = f"{BUILDINGS_WFS}&bbox={float(minx)},{float(miny)},{float(maxx)},{float(maxy)}"
    log.info("WFS budynki: pobieranie (bbox=%.0f,%.0f,%.0f,%.0f)...", minx, miny, maxx, maxy)
    gdf = gpd.read_file(url)
    log.info("WFS budynki: %d rekordów", len(gdf))
    return gdf
