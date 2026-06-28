"""
Transform: Building Permits
Filtrowanie, czyszczenie, spatial join z urban_blocks_geom.
"""

import logging
import geopandas as gpd
from shapely.ops import transform
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

VALID_ORGAN = "Prezydent Miasta Łódź"

VALID_CATEGORIES = {
    "I", "II", "III", "V",
    "IX", "X", "XI", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII"
}

# Frazy wykluczające z nazwy zamierzenia budowlanego
_EXCLUSION_SUBSTRINGS = [
    "ROZBIÓRKA", "ozbiórka",
    "KANAŁ", "WYMIANA",
    "Wewnętrzna instalacja gazowa",
    "BOISKO WIELOFUNKCYJNE,REMONT OGRODZENIA I TRYBUN,  UTWARDZ.TERENU",
    "Budowa banerów reklamowych na poł. i zach.scianie pawilonu",
    "BUDOWA RYNKU NOWEGO CENTRUM ŁODZI Z PARKINGIEM PODZIEMNYM",
    "BUDOWA SIECI CIEPŁOWNICZEJ WRAZ Z PRZYŁĄCZEM DO BUDYNKÓW",
    "BUDOWA STACJI TRANSFORMATOROWEJ 15/0,4 kV, LINII KABLOWYCH 15 I 0,4 kV, ZŁĄCZ KABLOWYCH ORAZ KANALIZACJI KABLOWEJ",
    "budynek węzła ciepł. z przyłączami dla bud. wielorodzinnego",
    "myjnia samochodowa w Galerii Łódzkiej",
    "PERZEBUDOWA FRAGM.DROGI",
    "POSZERZENIE PRZEJŚCIA W WEWNĘTRZNEJ NOŚNEJ ŚCIANIE MUROWANEJ POPRZEZ WYKONANIE NOWEGO NADPROŻA STALOWEGO W KLINICE STOMATOLOGICZNEJ",
    "PRZEBUDOWA ELEMENTÓW KONSTRUKCYJNYCH",
    "MONTAŻ INSTALACJI GAZOWEJ DLA POTRZEB",
    "ZMIANA DECYZJI",
    "PRZEBUDOWA CZĘŚCI STROPU DREWNIANEGO",
    "WYKONANIE ŚCIANY MIĘDZY LOKALAMI 14 I 4/5",
]

_EXCLUSION_SUBSTRINGS_CONDITIONAL = [
    # wykluczamy rozbiórki bez budowy
    ("ROZBIÓRKA", ["budowa", "BUDOWA", "PRZEBUD", "WIELORODZINNY",
                   "ielorodzinny", "BUDYNEK BIUROWY Z GARAŻEM I USŁUGAMI",
                   "BUDYNEK WIELOR. Z URZĄDZ.BUDOWL.",
                   "BUDYNEK USŁUG-HANDL.M-C POSTOJ.URZĄDZ.BUDWL",
                   "BUDYNEK O FUNKCJI USŁUG.Z",
                   "ROZBUD.BUDYNKU USŁUG-BIUR."]),
]

COLUMN_MAP = {
    "numer_ewidencyjny_urzad":         "build_perm_id",
    "numer_dzialki":                   "build_plot_no",
    "data_wplywu_wniosku_do_urzedu":   "date",
    "nazwa_zam_budowlanego":           "description",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_excluded(name: str) -> bool:
    """True jeśli nazwa zamierzenia pasuje do listy wykluczeń."""
    if not isinstance(name, str):
        return False

    import re
    if re.search(r"PROJEKT\s*[1-8]", name):
        return True

    for substr in _EXCLUSION_SUBSTRINGS:
        if substr in name:
            # sprawdź warunki warunkowe
            for trigger, exceptions in _EXCLUSION_SUBSTRINGS_CONDITIONAL:
                if trigger in name:
                    if any(exc in name for exc in exceptions):
                        return False  # wyjątek — nie wykluczamy
            return True

    return False


# ---------------------------------------------------------------------------
# Kroki transformacji
# ---------------------------------------------------------------------------

def filter_organ(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = gdf[gdf["nazwa_organu"] == VALID_ORGAN].copy()
    log.info("Po filtrze organu: %d → %d", len(gdf), len(result))
    return result


def filter_categories(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = gdf[gdf["kategoria_txt"].isin(VALID_CATEGORIES)].copy()
    log.info("Po filtrze kategorii: %d → %d", len(gdf), len(result))
    return result


def filter_exclusions(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    mask = gdf["nazwa_zam_budowlanego"].apply(_is_excluded)
    result = gdf[~mask].copy()
    log.info("Po filtrze wykluczeń: %d → %d", len(gdf), len(result))
    return result


def to_centroids(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:2177") -> gpd.GeoDataFrame:
    """Reprojekcja → centroid → EPSG:2177."""
    gdf = gdf.to_crs(target_crs)
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: transform(lambda x, y, z=None: (x, y), g)
    )
    gdf["geometry"] = gdf.centroid
    return gdf


def rename_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    import pandas as pd
    gdf = gdf.rename(columns=COLUMN_MAP)
    gdf["date"] = pd.to_datetime(gdf["date"], errors="coerce")
    gdf.columns = gdf.columns.str.lower()

    # Fallback: jeśli date jest null, użyj roku z nazwy warstwy (kolumna 'year')
    mask_null = gdf["date"].isna() & gdf["year"].notna()
    if mask_null.any():
        gdf.loc[mask_null, "date"] = pd.to_datetime(
            gdf.loc[mask_null, "year"].astype(int).astype(str) + "-01-01"
        )
        log.info("Uzupełniono %d brakujących dat z kolumny year", mask_null.sum())

    return gdf[["build_perm_id", "build_plot_no", "date", "description", "geometry", "year"]].copy()


def spatial_join_blocks(gdf: gpd.GeoDataFrame, engine: Engine) -> gpd.GeoDataFrame:
    """Spatial join z core.urban_blocks_geom → dodaje block_id."""
    blocks = gpd.read_postgis(
        "SELECT block_id, geometry FROM core.urban_blocks_geom",
        engine,
        geom_col="geometry"
    ).to_crs("EPSG:2177")

    result = gpd.sjoin(gdf, blocks[["block_id", "geometry"]], how="left", predicate="within")
    result = result.drop(columns="index_right", errors="ignore")
    result = result[~result["block_id"].isna()].copy()
    result["block_id"] = result["block_id"].astype(int)
    log.info("Po spatial join: %d rekordów z block_id", len(result))
    return result


def transform_pipeline(raw_gdf: gpd.GeoDataFrame, engine: Engine) -> gpd.GeoDataFrame:
    """Pełny pipeline transformacji. Wejście: surowy GDF z WFS."""
    gdf = filter_organ(raw_gdf)
    gdf = filter_categories(gdf)
    gdf = filter_exclusions(gdf)
    gdf = to_centroids(gdf)
    gdf = rename_columns(gdf)
    gdf = spatial_join_blocks(gdf, engine)
    return gdf
