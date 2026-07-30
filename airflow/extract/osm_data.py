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
    """Pobiera lodzkie-latest-free.shp.zip i rozpakowuje wybrane warstwy.

    Jeśli SHP-y lub ZIP już istnieją (pre-download na hoście) — pomija pobieranie.
    """
    import time

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "lodzkie-latest-free.shp.zip"

    # ── 1. Sprawdź czy SHP-y już rozpakowane ──────────────────────────────────
    poi_shp = out_dir / "gis_osm_pois_free_1.shp"
    landuse_shp = out_dir / "gis_osm_landuse_a_free_1.shp"
    traffic_shp = out_dir / "gis_osm_traffic_a_free_1.shp"
    if poi_shp.exists() and landuse_shp.exists() and traffic_shp.exists():
        log.info("SHP-y już istnieją – pomijam pobieranie i rozpakowywanie.")
        return out_dir

    # ── 2. Sprawdź czy ZIP już pobrany (pre-download na hoście) ───────────────
    MIN_ZIP_SIZE = 250 * 1024 * 1024  # 250 MB
    if zip_path.exists() and zip_path.stat().st_size >= MIN_ZIP_SIZE:
        log.info(
            "ZIP już istnieje (%.1f MB) – pomijam pobieranie.",
            zip_path.stat().st_size / 1e6,
        )
    else:
        # ── 3. Pobierz z retry + resume ───────────────────────────────────────
        CHUNK = 512 * 1024        # 512 KB
        SEG_SIZE = 90 * 1024 * 1024  # 90 MB per segment
        MAX_SEG_RETRIES = 10

        # Pobierz total size
        head = requests.head(GEOFABRIK_URL, timeout=120, allow_redirects=True)
        head.raise_for_status()
        total = int(head.headers["content-length"])
        log.info("Rozmiar pliku: %.1f MB", total / 1e6)

        # Segmenty: 0-89MB, 90-179MB, 180-264MB
        segments = []
        start = 0
        while start < total:
            end = min(start + SEG_SIZE - 1, total - 1)
            segments.append((start, end))
            start = end + 1

        log.info("Pobieranie w %d segmentach po ~%.0f MB", len(segments), SEG_SIZE / 1e6)

        with open(zip_path, "wb") as f_out:
            downloaded_total = 0
            for seg_idx, (seg_start, seg_end) in enumerate(segments, 1):
                seg_size = seg_end - seg_start + 1
                log.info(
                    "Segment %d/%d: %.0f–%.0f MB",
                    seg_idx, len(segments), seg_start / 1e6, seg_end / 1e6,
                )

                seg_downloaded = 0
                for attempt in range(1, MAX_SEG_RETRIES + 1):
                    byte_start = seg_start + seg_downloaded
                    headers = {"Range": f"bytes={byte_start}-{seg_end}"}
                    try:
                        with requests.get(
                            GEOFABRIK_URL, stream=True, timeout=120, headers=headers
                        ) as resp:
                            if resp.status_code not in (200, 206):
                                resp.raise_for_status()
                            for chunk in resp.iter_content(chunk_size=CHUNK):
                                f_out.write(chunk)
                                seg_downloaded += len(chunk)
                                downloaded_total += len(chunk)
                                log.info(
                                    "Pobrano %.1f / %.1f MB",
                                    downloaded_total / 1e6, total / 1e6,
                                )
                        break  # segment OK
                    except Exception as exc:
                        log.warning(
                            "Segment %d, próba %d błąd: %s", seg_idx, attempt, exc
                        )
                        if attempt == MAX_SEG_RETRIES:
                            raise
                        time.sleep(5)

        log.info("Pobieranie zakończone: %.1f MB", zip_path.stat().st_size / 1e6)

    log.info("Rozpakowywanie warstw: %s", KEEP_LAYERS)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if any(member.startswith(layer) for layer in KEEP_LAYERS):
                z.extract(member, out_dir)

    # Usuń ZIP i zbędne formaty
    zip_path.unlink()
    for path in out_dir.rglob("*"):
        if path.is_file() and path.suffix not in ALLOWED_EXT:
            path.unlink()

    log.info("Geofabrik gotowe: %s", out_dir)
    return out_dir


def get_scrub_shp_path(out_dir: Path = OSM_OUT_DIR) -> Path:
    """Zwraca ścieżkę do warstwy landuse (zawiera fclass='scrub')."""
    return out_dir / "gis_osm_landuse_a_free_1.shp"
