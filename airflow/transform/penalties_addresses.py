"""
Transform: Penalties + Addresses
Filtrowanie, geokodowanie, reprojekcja, agregacja do zmiennych.
"""

import re
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)


# ─── Konfiguracja ───────────────────────────────────────────────────────────

OFFENSE_ARTICLES = [
    "51§1 k.w. - zakłócanie spokoju lub porządku publicznego, spoczynku nocnego",
    "51§2 k.w. - zakłócanie spokoju i porządku publicznego, spoczynku nocnego będąc pod wpływem alkoholu",
]

# Słowa kluczowe — rekord musi zawierać przynajmniej jedno, żeby był geograficznie interpretowalny
KEYWORDS_GEO = ["PARK", "PL\\.", "PLAC", "AL\\.", "ALEJA", "PASAŻ", "SKWER", "DWORZEC", "STAW", "/"]
PATTERN_GEO = r"(\d+|" + "|".join(KEYWORDS_GEO) + r")"

PREFIXES_TO_STRIP = [
    "AL. ", "AL. ALEJA ADAMA ", "ALEJA T. ", "ANDRZEJA ",
    "PŁK. DR. ST. ", "MIKOŁAJA ", "LUDWIKA ", "ALEJA ADAMA ",
]

# Ręczna korekta nazw miejsc przed geokodowaniem
GEO_RENAME = {
    "PLAC BARLICKIEGO": "PL. NORBERTA BARLICKIEGO",
    "PARK MONIUSZKI": "PARK IM. STANISŁAWA MONIUSZKI",
    "28 PUŁKU STRZELCÓW KANIOWSKICH 63": "28 PUŁKU STRZELCÓW KANIOWSKICH 61/63",
}

# Miejsca niemapowalne na adres → mapowane na konkretny adres
RANDOM_ADDRESS_MAPPING = {
    "RODZINY POZNAŃSKICH": "KILIŃSKIEGO 66",
    "DWORZEC ŁÓDŹ FABRYCZNA": "KILIŃSKIEGO 66",
    "PIOTRKOWSKA, / STRUGA": "PIOTRKOWSKA 97",
    "PL. NORBERTA BARLICKIEGO": "MAŁA 5",
    "REWOLUCJI 1905 R., WSCHODNIA": "REWOLUCJI 1905 R. 16",
    "PIOTRKOWSKA 113/115": "PIOTRKOWSKA 113",
}

# Klucz: jak pojawia się w Excelu → Wartość: nazwa w OSMnx
PARK_SQUARES_MAPPING = {
    "ARTURA RUBINSTEINA": "Pasaż Artura Rubinsteina",
    "PARK IM. STANISŁAWA MONIUSZKI": "Park Moniuszki",
    "PIOTRKOWSKA, AL. ARTURA RUBINSTEINA": "Pasaż Artura Rubinsteina",
    "SKWER POWSTANIA WĘGIERSKIEGO 1956 ROKU": "Skwer Powstania Węgierskiego 1956 r.",
    "PARK SIENKIEWICZA": "Park Sienkiewicza",
    "PARK IM. STANISŁAWA STASZICA": "Park Staszica",
    "PL. KOMUNY PARYSKIEJ": "Plac Komuny Paryskiej",
    "LEONA SCHILLERA": "Aleja Leona Schillera",
    "PASAŻ RÓŻY": "Pasaż Róży",
    "PL. WOLNOŚCI": "Plac Wolności",
}

ADDRESS_STATUS_MAP = {
    "istniejący": "existing",
    "w trakcie budowy": "in_construction",
    "prognozowany": "estimated",
}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _swap_xy(geom):
    """Zamienia X↔Y — WFS EMUiA zwraca (lat, lon) zamiast (lon, lat)."""
    if geom is None:
        return None
    if geom.geom_type == "Point":
        return Point(geom.y, geom.x)
    elif geom.geom_type == "LineString":
        return LineString([(y, x) for x, y in geom.coords])
    elif geom.geom_type == "MultiLineString":
        return MultiLineString([LineString([(y, x) for x, y in ln.coords]) for ln in geom.geoms])
    return geom


def _strip_prefix(value: str, prefixes: list[str]) -> str:
    if not isinstance(value, str):
        return value
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix):].strip()
    return value


def _random_point_in_geom(geom):
    """Losowy punkt wewnątrz geometrii (dla parków/placów)."""
    if geom is None:
        return None
    if geom.geom_type == "Point":
        return geom
    if geom.geom_type == "LineString":
        return geom.interpolate(np.random.random(), normalized=True)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        minx, miny, maxx, maxy = geom.bounds
        for _ in range(500):
            p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if geom.contains(p):
                return p
        return geom.centroid
    return geom.centroid


# ─── Adresy ─────────────────────────────────────────────────────────────────

def transform_addresses(raw_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Czyści adresy z WFS Łódź EMUiA.
    Zamienia osie XY, standaryzuje nazwy kolumn, mapuje statusy.
    Zwraca GeoDataFrame w EPSG:2177 (spójne z urban_blocks_geom i build_perm).
    """
    gdf = raw_gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(_swap_xy)
    # WFS EMUiA zwraca EPSG:2177 z zamienionymi osiami → po swap_xy mamy poprawne 2177.
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:2177")

    gdf["ulica_nazwa"] = gdf["ulica_nazwa"].str.upper().str.strip()
    gdf["full_adress"] = (
        gdf["ulica_nazwa"].astype(str) + " " + gdf["numer_porzadkowy"].astype(str)
    )

    gdf = gdf.rename(columns={
        "ulica_nazwa": "street",
        "numer_porzadkowy": "building_no",
        "kod_pocztowy": "zip_code",
        "punkt_status": "status",
    })
    gdf["status"] = gdf["status"].replace(ADDRESS_STATUS_MAP)

    keep = ["gml_id", "guid", "full_adress", "street", "building_no", "zip_code", "status", "geometry"]
    existing = [c for c in keep if c in gdf.columns]
    log.info("Przetransformowano %d adresów", len(gdf))
    return gdf[existing].copy()


# ─── Ulice poza obszarem (do filtrowania kar) ────────────────────────────────

def build_streets_outside_pattern(streets_raw_gdf: gpd.GeoDataFrame, bbox_2177) -> str:
    """
    Buduje regex pattern dla nazw ulic SPOZA bbox obszaru badań.
    bbox_2177: Shapely geometry w EPSG:2177.

    WFS EMUiA zwraca geometrie w EPSG:2177 ale z zamienionymi osiami (Y,X).
    Po swap_xy mamy poprawne EPSG:2177 — porównujemy bezpośrednio z bbox bloków.
    """
    gdf = streets_raw_gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(_swap_xy)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:2177")

    outside = gdf[~gdf.geometry.intersects(bbox_2177)]
    names = outside["nazwa"].dropna().str.upper().str.strip().unique().tolist()
    log.info("Ulic poza bbox: %d", len(names))

    if not names:
        return r"UNMATCHABLE_XYZ_PATTERN"
    escaped = [re.escape(n) for n in names]
    return r"\b(" + "|".join(escaped) + r")\b"


# ─── Kary ───────────────────────────────────────────────────────────────────

def _filter_raw_penalties(
    df: pd.DataFrame,
    streets_outside_pattern: str,
    article_filter: list[str] | None,
    label: str = "",
) -> pd.DataFrame:
    """
    Wspólny filtr dla obu arkuszy:
    1. (opcjonalnie) artykuł
    2. musi zawierać słowo kluczowe geograficzne (numer / park / plac / ...)
    3. nie może zawierać ulicy spoza bbox
    Standaryzuje kolumny i czyści nazwy miejsc.
    """
    log.info("[%s] wejście: %d wierszy", label, len(df))

    if article_filter:
        df = df[df["Artykuły"].isin(article_filter)].copy()
        log.info("[%s] po filtrze artykułów: %d", label, len(df))

    n1 = len(df)
    df = df[
        df["Miejsce wykroczenia"]
        .str.upper()
        .str.contains(PATTERN_GEO, regex=True, na=False)
    ].copy()
    log.info("[%s] po filter1 (geo): %d → %d", label, n1, len(df))

    n2 = len(df)
    df = df[
        ~df["Miejsce wykroczenia"]
        .str.upper()
        .str.contains(streets_outside_pattern, regex=True, na=False)
    ].copy()
    log.info("[%s] po filter2 (poza bbox): %d → %d", label, n2, len(df))

    df = df.rename(columns={
        "Lp.": "pen_id",
        "Data wystawienia": "date",
        "Miejsce wykroczenia": "place_of_penalty",
    })
    df = df[["pen_id", "date", "place_of_penalty"]].copy()

    df["place_of_penalty"] = (
        df["place_of_penalty"]
        .str.upper()
        .str.strip()
        .apply(lambda x: _strip_prefix(x, PREFIXES_TO_STRIP))
        .replace(GEO_RENAME)
    )
    return df


def transform_penalties_sheet(
    raw_df: pd.DataFrame,
    pen_type: str,
    addresses_gdf: gpd.GeoDataFrame,
    parks_gdf: gpd.GeoDataFrame,
    streets_outside_pattern: str,
    article_filter: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Transformuje jeden arkusz Excel z karami.

    pen_type: 'alcohol_consumption' | 'offense'
    """
    df = _filter_raw_penalties(raw_df, streets_outside_pattern, article_filter, label=pen_type)
    df["pen_type"] = pen_type

    # Podziel: park/plac (geometria z OSMnx) vs. adresy (geometria z WFS)
    is_park = df["place_of_penalty"].isin(PARK_SQUARES_MAPPING.keys())
    df_parks = df[is_park].reset_index(drop=True).copy()
    df_addrs = df[~is_park].reset_index(drop=True).copy()
    log.info("[%s] po podziale: parki=%d, adresy=%d", pen_type, len(df_parks), len(df_addrs))

    # ── Parki / place ──
    df_parks["place_of_penalty"] = df_parks["place_of_penalty"].replace(PARK_SQUARES_MAPPING)
    parks_sel = (
        parks_gdf[parks_gdf["name"].isin(PARK_SQUARES_MAPPING.values())]
        .rename(columns={"name": "place_of_penalty"})
        [["place_of_penalty", "geometry"]]
    )
    merged_parks = df_parks.merge(parks_sel, on="place_of_penalty", how="left")
    merged_parks["geometry"] = merged_parks["geometry"].apply(_random_point_in_geom)
    gdf_parks = gpd.GeoDataFrame(merged_parks, geometry="geometry", crs=parks_gdf.crs)
    gdf_parks = gdf_parks.to_crs("EPSG:2177")  # OSMnx → 4326, reprojekcja do 2177
    log.info("[%s] parki po geokodowaniu: %d", pen_type, len(gdf_parks))

    # ── Adresy ──
    df_addrs["place_of_penalty"] = df_addrs["place_of_penalty"].replace(RANDOM_ADDRESS_MAPPING)
    # Normalizacja whitespace przed mergem (ochrona przed rozbieżnościami DB roundtrip)
    df_addrs["place_of_penalty"] = df_addrs["place_of_penalty"].str.strip()
    addr_lookup = addresses_gdf[["full_adress", "geometry"]].rename(
        columns={"full_adress": "place_of_penalty"}
    )
    addr_lookup["place_of_penalty"] = addr_lookup["place_of_penalty"].str.strip()
    n_before = len(df_addrs)
    merged_addrs = df_addrs.merge(addr_lookup, on="place_of_penalty", how="left")
    merged_addrs = merged_addrs.dropna(subset=["geometry"])
    log.info("[%s] adresy: %d → po merge+dropna: %d (utracono: %d)",
             pen_type, n_before, len(merged_addrs), n_before - len(merged_addrs))
    gdf_addrs = gpd.GeoDataFrame(merged_addrs, geometry="geometry", crs=addresses_gdf.crs)

    # ── Połącz ──
    combined = pd.concat([gdf_addrs, gdf_parks], ignore_index=True)
    # Losowy offset pen_id (oryginalne id z Excela nie jest unikalne między plikami)
    combined["pen_id"] = combined["pen_id"].astype(int) + np.random.randint(1, 10_000_001, size=len(combined))
    combined["date"] = pd.to_datetime(combined["date"])

    gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:2177")
    log.info("Przetransformowano %d rekordów kar (typ: %s)", len(gdf), pen_type)
    return gdf


# ─── Agregacja do mined.variables ───────────────────────────────────────────

VAR_IDS = {
    "alcohol_consumption": "urVibAlPn_coun_00000000",
    "offense": "urVibOfPn_coun_00000000",
}


def aggregate_penalties_to_variables(
    gdf_penalties: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Agreguje kary per block_id × year × pen_type → mined.variables.
    Bloki bez kar w danym roku dostają value = 0.
    Czyta z mined.penalties (wszystkie historyczne dane, nie tylko bieżący run).
    """
    if gdf_penalties.empty:
        log.warning("mined.penalties jest puste — zwracam pusty DataFrame zmiennych")
        return pd.DataFrame(columns=["var_id", "year", "block_id", "value"])

    gdf = gdf_penalties.copy()
    # Reprojekcja do CRS bloków (EPSG:2177) żeby sjoin działał poprawnie
    gdf = gdf.to_crs(blocks_gdf.crs)
    gdf["year_int"] = pd.to_datetime(gdf["date"]).dt.year

    joined = gpd.sjoin(
        gdf,
        blocks_gdf[["block_id", "geometry"]],
        how="left",
        predicate="within",
    ).dropna(subset=["block_id"])

    blocks = blocks_gdf["block_id"].unique()
    years = sorted(gdf["year_int"].dropna().unique())
    full_index = pd.MultiIndex.from_product([blocks, years], names=["block_id", "year_int"])

    dfs = []
    for pen_type, var_id in VAR_IDS.items():
        subset = joined[joined["pen_type"] == pen_type]
        agg = (
            subset.groupby(["block_id", "year_int"])
            .size()
            .reset_index(name="value")
            .set_index(["block_id", "year_int"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )
        agg["var_id"] = var_id
        agg["year"] = pd.to_datetime(agg["year_int"].astype(str) + "-01-01")
        agg["block_id"] = agg["block_id"].astype(int)
        dfs.append(agg[["var_id", "year", "block_id", "value"]])

    result = pd.concat(dfs, ignore_index=True)
    log.info("Zagregowano %d wierszy (blok × rok × pen_type, z zerami)", len(result))
    return result
