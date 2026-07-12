#!/usr/bin/env python3
"""
Pre-ETL Connectivity & Readiness Pipeline
==========================================
Checks all external APIs, WFS services, local files and database schema
required by the Urban Regeneration Platform Airflow ETL DAGs.

DAGs covered:
  - etl_osm              Geofabrik shapefile download
  - etl_build_perm       GUNB WFS (building permits)
  - etl_app_prices       Geoportal RCN WFS (apartment prices)
  - etl_penalties_addr   Lodz EMUiA WFS + Overpass API + local Excel files
  - etl_buildings        Lodz EGiB WFS (buildings)

Usage (from project root, PowerShell):
  python tests\\etl\\pre_etl.py

Exit codes:
  0  all checks passed
  1  one or more checks failed
"""

import os
import sys
import datetime
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from sqlalchemy import create_engine, text

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
PENALTIES_DIR = ROOT / "data" / "source" / "offenses_penalties"

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://urban_user:urban_password@localhost:5433/urban_db",
)

TIMEOUT = 20  # seconds per HTTP request

# ── External endpoints ─────────────────────────────────────────────────────────
#
# Each entry: (dag_label, check_label, url, method, params, expected_layers)
#   expected_layers: list of strings that must appear in GetCapabilities XML

ENDPOINTS = [
    {
        "dag":    "etl_osm",
        "label":  "Geofabrik shapefile (HEAD)",
        "url":    "https://download.geofabrik.de/europe/poland/lodzkie-latest-free.shp.zip",
        "method": "HEAD",
        "params": None,
        "layers": [],
        "note":   "Expected ~264 MB ZIP",
    },
    {
        "dag":    "etl_build_perm",
        "label":  "GUNB WFS GetCapabilities",
        "url":    "https://mapy.geoportal.gov.pl/wss/ext/GlownyUrzadNadzoruBudowlanego/RWDZ-WFS",
        "method": "GET",
        "params": {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetCapabilities"},
        "layers": [
            "ms:pozwolenia_2020", "ms:pozwolenia_2021", "ms:pozwolenia_2022",
            "ms:pozwolenia_2023", "ms:pozwolenia_2024", "ms:pozwolenia_2025",
            "ms:pozwolenia_pozostale",
        ],
        "note": "Building permits – year layers 2020-2025 + legacy",
    },
    {
        "dag":    "etl_app_prices",
        "label":  "Geoportal RCN WFS GetCapabilities",
        "url":    "https://mapy.geoportal.gov.pl/wss/service/rcn",
        "method": "GET",
        "params": {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetCapabilities"},
        "layers": ["ms:lokale"],
        "note":   "Apartment transaction prices",
    },
    {
        "dag":    "etl_penalties_addr",
        "label":  "Lodz EMUiA WFS GetCapabilities",
        "url":    "https://mapa.lodz.pl/OGC/EMUiA",
        "method": "GET",
        "params": {"SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetCapabilities"},
        "layers": ["ms:punkty_adresowe", "ms:ulice"],
        "note":   "Lodz address points and streets",
    },
    {
        "dag":    "etl_buildings",
        "label":  "Lodz EGiB WFS GetCapabilities",
        "url":    "https://mapa.lodz.pl/OGC/EGiB",
        "method": "GET",
        "params": {"SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetCapabilities"},
        "layers": ["ms:budynki"],
        "note":   "Lodz cadastral buildings",
    },
    {
        "dag":    "etl_penalties_addr",
        "label":  "Nominatim geocoding API",
        "url":    "https://nominatim.openstreetmap.org/search",
        "method": "GET",
        "params": {"q": "Lodz Poland", "format": "json", "limit": "1"},
        "layers": [],
        "note":   "OSMnx dependency – geocoding",
    },
    {
        "dag":    "etl_penalties_addr",
        "label":  "Overpass API status",
        "url":    "https://overpass-api.de/api/status",
        "method": "GET",
        "params": None,
        "layers": [],
        "note":   "OSMnx dependency – feature fetch",
    },
]

# ── Required DB tables ─────────────────────────────────────────────────────────

REQUIRED_TABLES = [
    # schema,           table,                   dag
    ("core",            "urban_blocks",           "all"),
    ("core",            "urban_blocks_geom",      "all"),
    ("mined",           "variables",              "all"),
    ("meta",            "var_description",        "all"),
    ("osm",             "poi",                    "etl_osm"),
    ("osm",             "poly",                   "etl_osm"),
    ("mined",           "Build_perm",             "etl_build_perm"),
    ("mined",           "adresses",               "etl_penalties_addr"),
    ("mined",           "app_prices",             "etl_app_prices"),
    ("mined",           "buildings",              "etl_buildings"),
    ("mined",           "penalties",              "etl_penalties_addr"),
    ("regeneration",    "actions",                "all"),
    ("audit",           "etl_log",                "all"),
    ("audit",           "processed_files",        "etl_penalties_addr"),
    ("audit",           "stg_osm_poi",            "etl_osm"),
    ("audit",           "stg_osm_poly",           "etl_osm"),
    ("results",         "optimization",           "all"),
    ("results",         "predicted_reg_price",    "all"),
    ("results",         "uplifts",                "all"),
    ("results",         "feature_spec",           "all"),
    ("results",         "hypp_spec",              "all"),
    ("results",         "model_setups",           "all"),
]

# ── Result collector ───────────────────────────────────────────────────────────

results: list[dict] = []


def record(category: str, label: str, passed: bool, detail: str = "") -> None:
    results.append(
        {"category": category, "label": label, "passed": passed, "detail": detail}
    )
    status = "PASS" if passed else "FAIL"
    detail_str = f"  -> {detail}" if detail else ""
    print(f"  [{status}]  {label}{detail_str}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _layers_in_xml(content: bytes, layer_names: list[str]) -> tuple[bool, list[str]]:
    """Returns (all_found, missing_layers) by scanning raw XML bytes."""
    text_content = content.decode("utf-8", errors="replace")
    missing = [ln for ln in layer_names if ln not in text_content]
    return (len(missing) == 0), missing


def _size_mb(response: requests.Response) -> str:
    cl = response.headers.get("Content-Length") or response.headers.get("content-length")
    if cl:
        return f"{int(cl) / 1e6:.1f} MB"
    return "size unknown"


# ── Check functions ────────────────────────────────────────────────────────────

def check_db_connection() -> bool:
    print("\n[DATABASE]")
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()
        record("DATABASE", "PostgreSQL connectivity", True, version.split(",")[0])
        return True
    except Exception as exc:
        record("DATABASE", "PostgreSQL connectivity", False, str(exc))
        return False


def check_db_tables(db_ok: bool) -> None:
    if not db_ok:
        for schema, table, dag in REQUIRED_TABLES:
            record("DB SCHEMA", f"{schema}.{table}", False, "skipped – DB unreachable")
        return

    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        for schema, table, dag in REQUIRED_TABLES:
            try:
                row_count = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_schema = :s AND table_name = :t"
                    ),
                    {"s": schema, "t": table},
                ).scalar()
                exists = row_count > 0
                detail = f"dag: {dag}" if not exists else f"dag: {dag}"
                record("DB SCHEMA", f"{schema}.{table}", exists,
                       "table missing" if not exists else "")
            except Exception as exc:
                record("DB SCHEMA", f"{schema}.{table}", False, str(exc))


def check_endpoints() -> None:
    print("\n[EXTERNAL ENDPOINTS]")
    headers = {
        "User-Agent": "UrbanPlatform-PreETL-Check/1.0",
        "Accept": "application/xml, text/xml, */*",
    }
    for ep in ENDPOINTS:
        label = f"{ep['dag']} / {ep['label']}"
        try:
            if ep["method"] == "HEAD":
                resp = requests.head(
                    ep["url"], timeout=TIMEOUT, allow_redirects=True, headers=headers
                )
            else:
                resp = requests.get(
                    ep["url"], params=ep["params"], timeout=TIMEOUT,
                    allow_redirects=True, headers=headers
                )

            http_ok = resp.status_code < 400

            if not http_ok:
                record("ENDPOINT", label, False,
                       f"HTTP {resp.status_code} ({ep['note']})")
                continue

            # Extra: file size for Geofabrik HEAD
            if ep["method"] == "HEAD":
                record("ENDPOINT", label, True,
                       f"HTTP {resp.status_code}, {_size_mb(resp)}")
                continue

            # Extra: layer presence check in WFS XML
            if ep["layers"]:
                all_found, missing = _layers_in_xml(resp.content, ep["layers"])
                if all_found:
                    record("ENDPOINT", label, True,
                           f"HTTP {resp.status_code}, all {len(ep['layers'])} layers present")
                else:
                    record("ENDPOINT", label, False,
                           f"HTTP {resp.status_code}, missing layers: {missing}")
                continue

            record("ENDPOINT", label, True, f"HTTP {resp.status_code}")

        except requests.exceptions.ConnectionError as exc:
            record("ENDPOINT", label, False, f"Connection refused / DNS failure")
        except requests.exceptions.Timeout:
            record("ENDPOINT", label, False, f"Timeout after {TIMEOUT}s")
        except Exception as exc:
            record("ENDPOINT", label, False, str(exc)[:120])


def check_local_files() -> None:
    print("\n[LOCAL FILES]")
    try:
        xlsx_files = list(PENALTIES_DIR.glob("*.xlsx"))
        exists = PENALTIES_DIR.exists()
        if not exists:
            record("LOCAL FILES", "Penalties Excel directory", False,
                   f"Directory not found: {PENALTIES_DIR}")
        elif not xlsx_files:
            record("LOCAL FILES", "Penalties Excel files", False,
                   f"No .xlsx files in {PENALTIES_DIR}")
        else:
            record("LOCAL FILES", "Penalties Excel files", True,
                   f"{len(xlsx_files)} file(s): {', '.join(f.name for f in xlsx_files)}")
    except Exception as exc:
        record("LOCAL FILES", "Penalties Excel files", False, str(exc))

    # OSM output directory (may or may not have files yet)
    osm_dir = ROOT / "data" / "source" / "osm_latest"
    shp_files = list(osm_dir.glob("*.shp")) if osm_dir.exists() else []
    record("LOCAL FILES", "OSM output directory (osm_latest)",
           osm_dir.exists(),
           f"{len(shp_files)} .shp file(s) present" if osm_dir.exists()
           else f"Not yet created – will be populated by etl_osm")


# ── Banner / summary ───────────────────────────────────────────────────────────

def print_banner(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}")
    print(f" {title}")
    print(f" Urban Regeneration Platform")
    print(f" Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(line)


def print_summary() -> None:
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    line   = "=" * 64

    print(f"\n{line}")
    print(f" SUMMARY")
    print(f" Total checks : {total}")
    print(f" Passed       : {passed}")
    print(f" Failed       : {failed}")
    print(line)

    if failed:
        print("\nFailed checks:")
        for r in results:
            if not r["passed"]:
                detail = f"  -> {r['detail']}" if r["detail"] else ""
                print(f"  [FAIL]  [{r['category']}] {r['label']}{detail}")
    else:
        print("\nAll checks passed. ETL pipelines are ready to run.")

    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    print_banner("PRE-ETL CONNECTIVITY & READINESS PIPELINE")

    print("\n[DATABASE]")
    db_ok = check_db_connection()
    check_db_tables(db_ok)
    check_endpoints()
    check_local_files()
    print_summary()

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
