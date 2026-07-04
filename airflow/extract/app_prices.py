"""
Extract: App Prices (WFS RCN)
Pobiera transakcje mieszkaniowe z Rejestru Cen Nieruchomości.
"""

import io
import logging
import requests
import geopandas as gpd

log = logging.getLogger(__name__)

WFS_URL = "https://mapy.geoportal.gov.pl/wss/service/rcn"
LAYER   = "ms:lokale"


def fetch_app_prices(bbox_2180: tuple) -> gpd.GeoDataFrame:
    """
    Pobiera transakcje z WFS RCN dla podanego bbox (EPSG:2180).
    bbox_2180: (minx, miny, maxx, maxy) z ldz_blocks.to_crs('EPSG:2180').total_bounds

    MapServer wymaga GET — budujemy URL ręcznie.
    WFS 2.0.0 axis order dla EPSG:2180: (miny, minx, maxy, maxx).
    """
    minx, miny, maxx, maxy = bbox_2180
    bbox_str = f"{miny},{minx},{maxy},{maxx},EPSG:2180"

    params = {
        "SERVICE":   "WFS",
        "VERSION":   "2.0.0",
        "REQUEST":   "GetFeature",
        "TYPENAMES": LAYER,
        "BBOX":      bbox_str,
        "COUNT":     "200000",
    }

    log.info("GET WFS RCN: %s (bbox=%s)", WFS_URL, bbox_str)
    resp = requests.get(WFS_URL, params=params, timeout=120)
    resp.raise_for_status()

    content = resp.content
    log.info("WFS response (pierwsze 400 znaków): %s",
             content[:400].decode("utf-8", errors="replace"))

    gdf = gpd.read_file(io.BytesIO(content))
    log.info("Pobrano %d rekordów z WFS RCN", len(gdf))
    return gdf
