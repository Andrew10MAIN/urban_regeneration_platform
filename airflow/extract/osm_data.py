"""
Extract: OSM data
- POI + parking polygons: Geofabrik (lodzkie-latest-free.shp.zip)
- Scrubs: OSMnx API
"""

import io
import logging
import zipfile
from pathlib import Path

import requests
import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

GEOFABRIK_URL  = "https://download.geofabrik.de/europe/poland/lodzkie-latest-free.shp.zip"
KEEP_LAYERS    = ["gis_osm_pois_free_1", "gis_osm_traffic_a_free_1", "gis_osm_landuse_a_free_1"]
ALLOWED_EXT    = {".shp", ".dbf", ".shx", ".prj", ".cpg"}
OSM_OUT_DIR    = Path("/opt/airflow/urban_platform/data/source/osm_latest")


def download_geofabrik(out_dir: Path = OSM_OUT_DIR) -> Path:
    """Pobiera lodzkie-latest-free.shp.zip i rozpakowuje wybrane warstwy."""
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Pobieranie Geofabrik ZIP: %s", GEOFABRIK_URL)
    resp = requests.get(GEOFABRIK_URL, stream=True, timeout=600)
    resp.raise_for_status()

    log.info("Rozpakowywanie warstw: %s", KEEP_LAYERS)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for member in z.namelist():
            if any(member.startswith(layer) for layer in KEEP_LAYERS):
                z.extract(member, out_dir)

    # Usuń zbędne formaty
    for path in out_dir.rglob("*"):
        if path.is_file() and path.suffix not in ALLOWED_EXT:
            path.unlink()

    log.info("Geofabrik gotowe: %s", out_dir)
    return out_dir


def get_scrub_shp_path(out_dir: Path = OSM_OUT_DIR) -> Path:
    """Zwraca ścieżkę do warstwy landuse (zawiera fclass='scrub')."""
    return out_dir / "gis_osm_landuse_a_free_1.shp"
