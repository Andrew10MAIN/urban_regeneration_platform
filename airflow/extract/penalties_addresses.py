"""
Extract: Penalties + Addresses
- Adresy:  WFS Łódź EMUiA
- Kary:    pliki Excel z data/source/offenses_penalties/
- OSM:     parki i place z OSMnx
"""

import os
import hashlib
import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from sqlalchemy.engine import Engine
from sqlalchemy import text

log = logging.getLogger(__name__)

URBAN_PLATFORM_PATH = os.environ.get("URBAN_PLATFORM_PATH", "/opt/airflow/urban_platform")
PENALTIES_DIR = Path(URBAN_PLATFORM_PATH) / "data" / "source" / "offenses_penalties"

ADDRESS_WFS = (
    "https://mapa.lodz.pl/OGC/EMUiA"
    "?service=WFS&version=1.1.0&request=GetFeature"
    "&typename=ms:punkty_adresowe"
)
STREETS_WFS = (
    "https://mapa.lodz.pl/OGC/EMUiA"
    "?service=WFS&version=1.1.0&request=GetFeature"
    "&typename=ms:ulice"
)


# ─── File tracking ──────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_unprocessed_files(engine: Engine, dag_id: str) -> list[dict]:
    """
    Zwraca listę plików Excel w PENALTIES_DIR, których jeszcze nie przetworzono
    lub których hash się zmienił (tj. zostały zaktualizowane).
    """
    all_files = sorted(PENALTIES_DIR.glob("*.xlsx"))
    if not all_files:
        log.warning("Brak plików Excel w %s", PENALTIES_DIR)
        return []

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT file_name, file_hash
                FROM audit.processed_files
                WHERE dag_id = :dag_id
            """),
            {"dag_id": dag_id},
        ).fetchall()
    processed = {r.file_name: r.file_hash for r in rows}

    to_process = []
    for fp in all_files:
        fhash = _file_hash(str(fp))
        if fp.name not in processed or processed[fp.name] != fhash:
            to_process.append({"path": str(fp), "name": fp.name, "hash": fhash})
            log.info("Do przetworzenia: %s", fp.name)
        else:
            log.info("Pominięto (już przetworzone, hash bez zmian): %s", fp.name)

    return to_process


# ─── OSMnx ─────────────────────────────────────────────────────────────────

def fetch_osm_parks_squares() -> gpd.GeoDataFrame:
    """Pobiera parki i place z OSMnx dla Łodzi (EPSG:4326)."""
    import osmnx as ox  # lazy import — nie blokuje parsowania DAGu
    log.info("OSMnx: pobieranie granic Łodzi...")
    lodz = ox.geocode_to_gdf("Łódź, Poland")
    geom = lodz.geometry.iloc[0]

    log.info("OSMnx: pobieranie parków...")
    parks = ox.features_from_polygon(geom, {"leisure": "park"}).reset_index()

    log.info("OSMnx: pobieranie placów/deptaków...")
    squares = ox.features_from_polygon(
        geom, {"place": "square", "highway": "pedestrian"}
    ).reset_index()

    combined = pd.concat(
        [parks[["name", "geometry"]], squares[["name", "geometry"]]]
    ).reset_index(drop=True)
    gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    log.info("OSMnx: %d obiektów (parki + place)", len(gdf))
    return gdf


# ─── WFS ────────────────────────────────────────────────────────────────────

def fetch_addresses(bbox_2177: tuple) -> gpd.GeoDataFrame:
    """
    Pobiera adresy z WFS Łódź EMUiA w podanym bbox (EPSG:2177).
    Zwraca surowy GeoDataFrame (CRS zależy od WFS — zazwyczaj lat/lon do zamiany).
    """
    minx, miny, maxx, maxy = bbox_2177
    url = f"{ADDRESS_WFS}&bbox={minx},{miny},{maxx},{maxy}"
    log.info("WFS adresy: pobieranie...")
    gdf = gpd.read_file(url)
    log.info("WFS adresy: %d rekordów", len(gdf))
    return gdf


def fetch_streets() -> gpd.GeoDataFrame:
    """Pobiera ulice Łodzi z WFS (cała Łódź, bez filtra bbox)."""
    log.info("WFS ulice: pobieranie...")
    gdf = gpd.read_file(STREETS_WFS)
    log.info("WFS ulice: %d rekordów", len(gdf))
    return gdf


# ─── Excel ──────────────────────────────────────────────────────────────────

def load_penalties_excel(filepath: str) -> dict[str, pd.DataFrame]:
    """
    Wczytuje Excel z karami. Wykrywa arkusze po nazwie.
    Zwraca {"alcohol": df, "offense": df} — klucz może nie istnieć.
    """
    log.info("Excel: wczytywanie %s", filepath)
    result: dict[str, pd.DataFrame] = {}
    xl = pd.ExcelFile(filepath)

    for sheet in xl.sheet_names:
        s = sheet.lower().strip()
        if "alkohol" in s or "spożywanie" in s:
            result["alcohol"] = xl.parse(sheet)
            log.info("  → arkusz alkohol (%s): %d wierszy", sheet, len(result["alcohol"]))
        elif "wykroczeni" in s or "porządkow" in s:
            result["offense"] = xl.parse(sheet)
            log.info("  → arkusz wykroczenia (%s): %d wierszy", sheet, len(result["offense"]))

    if not result:
        log.warning("Nie rozpoznano żadnego arkusza w %s — dostępne: %s", filepath, xl.sheet_names)
    return result
