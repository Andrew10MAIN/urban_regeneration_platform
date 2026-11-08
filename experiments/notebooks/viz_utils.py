"""
Funkcje wizualizacyjne wspoldzielone przez notebooki w tym katalogu.

Uzycie w notebooku lezacym obok tego pliku:

    from viz_utils import (TILE_SOURCES, make_cmap, plot_block_map,
                           plot_block_category_map, plot_hbar, plot_scatter,
                           distribution_table, causal_effects_table,
                           shap_summary)

Kolory podawane sa jako stringi hex ('#0257a3') — VS Code renderuje przy nich
kwadracik z probnikiem koloru. Mapy wymagaja contextily: pip install contextily.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Wspolne ustawienia i helpery wizualizacji
#
# Kolory podawane sa jako stringi hex ('#2a78d6'). VS Code renderuje przy nich
# kwadracik z probnikiem koloru, wiec zmiana koloru to klikniecie w argument.
# ─────────────────────────────────────────────────────────────────────────────
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import contextily as cx
    _CX_AVAILABLE = True
except ImportError:          # pragma: no cover
    _CX_AVAILABLE = False
    warnings.warn("contextily nie jest zainstalowane — mapy bez kafelkow. "
                  "Instalacja: pip install contextily")

WEB_MERCATOR = 3857

# ── Zrodla kafelkow ──────────────────────────────────────────────────────────
# Klucz podawany w argumencie `tiles=`. None wylacza kafelki.
TILE_SOURCES: dict[str, object] = {
    # jasna mapa graficzna — najlepsze tlo pod warstwy kolorystyczne
    "positron": (cx.providers.CartoDB.Positron          if _CX_AVAILABLE else None),
    # mapa z ulicami, nazwami i POI — kontekst miejski
    "osm":      (cx.providers.OpenStreetMap.Mapnik      if _CX_AVAILABLE else None),
    # zobrazowanie satelitarne, dziala wszedzie
    "esri":     (cx.providers.Esri.WorldImagery         if _CX_AVAILABLE else None),
    # ortofotomapa GUGiK — najwyzsza rozdzielczosc nad Lodzia, bywa wolna
    "ortho_pl": (
        "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMTS/StandardResolution"
        "?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTOFOTOMAPA"
        "&STYLE=default&FORMAT=image/jpeg&tileMatrixSet=EPSG:3857"
        "&tileMatrix=EPSG:3857:{z}&tileRow={y}&tileCol={x}"
    ),
}


def make_cmap(color_low: str = "#ffffff",
              color_high: str = "#0257a3",
              color_mid: str | None = None) -> LinearSegmentedColormap:
    """
    Buduje mape kolorow z 2 lub 3 kolorow hex.

    color_mid podany -> paleta rozbiezna (np. ujemne/dodatnie uplifty),
    uzywana razem z diverging=True, ktore ustawia 0 w srodku skali.
    """
    cols = [color_low, color_high] if color_mid is None else [color_low, color_mid, color_high]
    return LinearSegmentedColormap.from_list("custom", cols, N=256)


def _fit_extent(ax, bounds, aspect: tuple[int, int] | None, pad: float = 0.06):
    """
    Ustawia zakres osi tak, zeby mial zadane proporcje.

    Kwartaly strefy rewitalizacji tworza ksztalt wysoki i waski. Przy formacie
    16:9 brakujaca szerokosc nie jest pusta — zakres osi jest rozszerzany
    symetrycznie, a powstale miejsce wypelniaja kafelki mapy.
    """
    minx, miny, maxx, maxy = bounds
    dx, dy = maxx - minx, maxy - miny
    minx, maxx = minx - dx * pad, maxx + dx * pad
    miny, maxy = miny - dy * pad, maxy + dy * pad
    dx, dy = maxx - minx, maxy - miny
    cx_, cy_ = (minx + maxx) / 2, (miny + maxy) / 2

    if aspect is not None:
        target = aspect[0] / aspect[1]
        if dx / dy < target:
            dx = dy * target          # za waskie -> poszerz w poziomie
        else:
            dy = dx / target          # za szerokie -> poszerz w pionie

    ax.set_xlim(cx_ - dx / 2, cx_ + dx / 2)
    ax.set_ylim(cy_ - dy / 2, cy_ + dy / 2)


def _add_tiles(ax, tiles: str | None, tiles_alpha: float):
    """Dokłada kafelki pod warstwami danych. Brak sieci nie przerywa rysowania."""
    if tiles is None or not _CX_AVAILABLE:
        return
    if tiles not in TILE_SOURCES:
        raise ValueError(f"tiles='{tiles}' — dostepne: {sorted(TILE_SOURCES)}")
    try:
        cx.add_basemap(ax, source=TILE_SOURCES[tiles], crs=f"EPSG:{WEB_MERCATOR}",
                       alpha=tiles_alpha, attribution_size=6, zorder=0)
    except Exception as exc:                                  # brak sieci, limit serwera
        warnings.warn(f"Nie udalo sie pobrac kafelkow '{tiles}': {exc}")


def _style_axes(ax, *, title, xlabel, ylabel,
                title_fontsize, label_fontsize, tick_fontsize, show_axes):
    if title:
        ax.set_title(title, fontsize=title_fontsize, loc="left", color="#0b0b0b")
    if show_axes:
        ax.set_xlabel(xlabel or "", fontsize=label_fontsize, color="#52514e")
        ax.set_ylabel(ylabel or "", fontsize=label_fontsize, color="#52514e")
        ax.tick_params(labelsize=tick_fontsize, colors="#52514e")
    else:
        ax.set_axis_off()


def _add_treated_outline(ax, blocks_gdf, *, treated_col, show_indirect,
                         color, width, indirect_color, indirect_width):
    """
    Obwiednia kwartalow poddanych rewitalizacji: ciagla pogrubiona dla
    traktowanych bezposrednio, opcjonalnie przerywana dla strefy oddzialywania.
    Rysowana jest granica sumy geometrii, nie pojedyncze kwartaly.
    """
    if treated_col not in blocks_gdf.columns:
        warnings.warn(f"Brak kolumny '{treated_col}' — pomijam obwiednie.")
        return
    direct = blocks_gdf[blocks_gdf[treated_col] == 1]
    if len(direct):
        direct.dissolve().boundary.plot(ax=ax, color=color, linewidth=width, zorder=5)
    if show_indirect and "treated_1nq" in blocks_gdf.columns:
        indirect = blocks_gdf[(blocks_gdf["treated_1nq"] == 1) & (blocks_gdf[treated_col] != 1)]
        if len(indirect):
            indirect.dissolve().boundary.plot(
                ax=ax, color=indirect_color, linewidth=indirect_width,
                linestyle="--", zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Mapy
# ─────────────────────────────────────────────────────────────────────────────

def plot_block_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    *,
    # --- kolory (hex -> probnik koloru w VS Code) ---
    color_low: str = "#ffffff",
    color_high: str = "#0257a3",
    color_mid: str | None = None,
    diverging: bool = False,
    color_missing: str = "#d2d6d8",
    edge_color: str = "#ffffff",
    edge_width: float = 0.3,
    # --- przezroczystosc ---
    alpha: float = 0.85,
    tiles_alpha: float = 0.55,
    # --- kafelki i format ---
    tiles: str | None = "positron",
    aspect: tuple[int, int] | None = (16, 9),
    fig_width: float = 12.0,
    # --- obwiednia kwartalow zrewitalizowanych ---
    show_treated_outline: bool = True,
    treated_col: str = "treated_d1nq",
    outline_color: str = "#000000",
    outline_width: float = 2.5,
    show_indirect_outline: bool = False,
    indirect_color: str = "#000000",
    indirect_width: float = 1.6,
    # --- opisy i czcionki ---
    title: str | None = None,
    xlabel: str | None = "Easting [m]",
    ylabel: str | None = "Northing [m]",
    legend_label: str | None = None,
    title_fontsize: float = 15,
    label_fontsize: float = 11,
    tick_fontsize: float = 9,
    legend_fontsize: float = 10,
    show_axes: bool = False,
    # --- skala barw ---
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    ax=None,
):
    """
    Choropleth kwartalow na tle kafelkow.

    gdf        : GeoDataFrame z geometria kwartalow i kolumna value_col
    value_col  : kolumna liczbowa do pokolorowania; NaN -> color_missing
    diverging  : True ustawia 0 w srodku skali (dla upliftow dodatnich i ujemnych);
                 wymaga color_mid, np. color_low='#b40000', color_mid='#ffffff',
                 color_high='#0046b4'
    aspect     : docelowe proporcje rysunku, np. (16, 9); None = proporcje danych.
                 Przestrzen powstala przy rozszerzeniu zakresu wypelniaja kafelki.

    Zwraca (fig, ax).
    """
    data = gdf.to_crs(WEB_MERCATOR).copy()
    values = pd.to_numeric(data[value_col], errors="coerce")

    cmap = make_cmap(color_low, color_high, color_mid)
    finite = values.dropna()
    if finite.empty:
        raise ValueError(f"Kolumna '{value_col}' nie zawiera zadnej wartosci liczbowej.")

    lo = float(finite.min()) if vmin is None else vmin
    hi = float(finite.max()) if vmax is None else vmax
    if diverging:
        bound = max(abs(lo), abs(hi)) or 1.0
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    else:
        norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)

    if ax is None:
        height = fig_width / (aspect[0] / aspect[1]) if aspect else fig_width
        fig, ax = plt.subplots(figsize=(fig_width, height))
    else:
        fig = ax.figure

    missing = data[values.isna()]
    if len(missing):
        missing.plot(ax=ax, color=color_missing, alpha=alpha,
                     edgecolor=edge_color, linewidth=edge_width, zorder=2)
    present = data[values.notna()]
    present.plot(ax=ax, column=value_col, cmap=cmap, norm=norm, alpha=alpha,
                 edgecolor=edge_color, linewidth=edge_width, zorder=3)

    if show_treated_outline:
        _add_treated_outline(ax, data, treated_col=treated_col,
                             show_indirect=show_indirect_outline,
                             color=outline_color, width=outline_width,
                             indirect_color=indirect_color, indirect_width=indirect_width)

    ax.set_aspect("equal")
    _fit_extent(ax, data.total_bounds, aspect)
    _add_tiles(ax, tiles, tiles_alpha)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.015)
        cbar.set_label(legend_label or value_col, fontsize=legend_fontsize, color="#52514e")
        cbar.ax.tick_params(labelsize=tick_fontsize, colors="#52514e")
        if len(missing):
            ax.legend(handles=[Patch(facecolor=color_missing, edgecolor=edge_color,
                                     label=f"brak danych ({len(missing)})")],
                      fontsize=legend_fontsize, frameon=False, loc="upper left")

    _style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel,
                title_fontsize=title_fontsize, label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize, show_axes=show_axes)
    fig.tight_layout()
    return fig, ax


def plot_block_category_map(
    gdf: gpd.GeoDataFrame,
    category_col: str,
    category_colors: dict[str, str],
    *,
    category_order: list[str] | None = None,
    alpha: float = 0.85,
    tiles_alpha: float = 0.55,
    edge_color: str = "#ffffff",
    edge_width: float = 0.3,
    tiles: str | None = "positron",
    aspect: tuple[int, int] | None = (16, 9),
    fig_width: float = 12.0,
    show_treated_outline: bool = True,
    treated_col: str = "treated_d1nq",
    outline_color: str = "#000000",
    outline_width: float = 2.5,
    show_indirect_outline: bool = False,
    indirect_color: str = "#000000",
    indirect_width: float = 1.6,
    title: str | None = None,
    xlabel: str | None = "Easting [m]",
    ylabel: str | None = "Northing [m]",
    legend_title: str | None = None,
    title_fontsize: float = 15,
    label_fontsize: float = 11,
    tick_fontsize: float = 9,
    legend_fontsize: float = 10,
    show_axes: bool = False,
    show_legend: bool = True,
    ax=None,
):
    """
    Mapa kwartalow w podziale na kategorie (np. wybrane do rewitalizacji,
    strefa oddzialywania, poza wyborem).

    category_colors : {'nazwa kategorii': '#rrggbb'} — kolor na kategorie
    """
    data = gdf.to_crs(WEB_MERCATOR).copy()
    order = category_order or list(category_colors)

    if ax is None:
        height = fig_width / (aspect[0] / aspect[1]) if aspect else fig_width
        fig, ax = plt.subplots(figsize=(fig_width, height))
    else:
        fig = ax.figure

    handles = []
    for cat in order:
        subset = data[data[category_col] == cat]
        colour = category_colors.get(cat, "#d2d6d8")
        if len(subset):
            subset.plot(ax=ax, color=colour, alpha=alpha,
                        edgecolor=edge_color, linewidth=edge_width, zorder=3)
        handles.append(Patch(facecolor=colour, edgecolor=edge_color,
                             label=f"{cat} ({len(subset)})"))

    if show_treated_outline:
        _add_treated_outline(ax, data, treated_col=treated_col,
                             show_indirect=show_indirect_outline,
                             color=outline_color, width=outline_width,
                             indirect_color=indirect_color, indirect_width=indirect_width)

    ax.set_aspect("equal")
    _fit_extent(ax, data.total_bounds, aspect)
    _add_tiles(ax, tiles, tiles_alpha)

    if show_legend:
        leg = ax.legend(handles=handles, fontsize=legend_fontsize, frameon=False,
                        loc="upper left", title=legend_title)
        if legend_title:
            leg.get_title().set_fontsize(legend_fontsize)

    _style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel,
                title_fontsize=title_fontsize, label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize, show_axes=show_axes)
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Wykresy slupkowe, punktowe i tabele
# ─────────────────────────────────────────────────────────────────────────────

def plot_hbar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    *,
    color: str = "#0257a3",
    color_negative: str | None = None,
    alpha: float = 0.9,
    sort: bool = True,
    sort_desc: bool = True,
    sort_col: str | None = None,
    top_n: int | None = None,
    value_fmt: str = "{:,.3f}",
    show_values: bool = True,
    aspect: tuple[int, int] | None = (16, 9),
    fig_width: float = 10.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title_fontsize: float = 15,
    label_fontsize: float = 11,
    tick_fontsize: float = 10,
    value_fontsize: float = 9,
    grid: bool = True,
    ax=None,
):
    """
    Poziomy wykres slupkowy, domyslnie od najwiekszej wartosci na gorze.

    color_negative : osobny kolor dla slupkow ujemnych (None = jeden kolor)
    sort           : False zachowuje kolejnosc wierszy z ramki — przydatne, gdy
                     slupki pokazuja wartosc ze znakiem, a uszeregowane maja byc
                     wg innej wielkosci (np. waznosci)
    sort_col       : kolumna sortowania; None = value_col
    """
    cols = [label_col, value_col]
    if sort_col and sort_col not in cols:
        cols.append(sort_col)
    data = df[cols].dropna(subset=[value_col]).copy()

    if sort:
        data = data.sort_values(sort_col or value_col, ascending=not sort_desc)
    if top_n:
        data = data.head(top_n)
    data = data.iloc[::-1]   # barh rysuje od dolu -> pierwszy wiersz na gorze

    if ax is None:
        height = fig_width / (aspect[0] / aspect[1]) if aspect else max(3.0, 0.45 * len(data))
        fig, ax = plt.subplots(figsize=(fig_width, height))
    else:
        fig = ax.figure

    colors = [
        color_negative if (color_negative and v < 0) else color
        for v in data[value_col]
    ]
    bars = ax.barh(data[label_col].astype(str), data[value_col],
                   color=colors, alpha=alpha, zorder=3)

    if show_values:
        span = float(data[value_col].abs().max()) or 1.0
        for bar, v in zip(bars, data[value_col]):
            offset = span * 0.01
            ax.text(v + (offset if v >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    value_fmt.format(v),
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=value_fontsize, color="#52514e")
        pad = span * 0.18
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)

    if grid:
        ax.grid(axis="x", color="#e5e5e3", linewidth=0.8)
        ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#52514e")
    ax.tick_params(labelsize=tick_fontsize, colors="#52514e")
    ax.set_xlabel(xlabel or value_col, fontsize=label_fontsize, color="#52514e")
    ax.set_ylabel(ylabel or "", fontsize=label_fontsize, color="#52514e")
    if title:
        ax.set_title(title, fontsize=title_fontsize, loc="left", color="#0b0b0b")
    fig.tight_layout()
    return fig, ax


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    color: str = "#0257a3",
    color_by: str | None = None,
    category_colors: dict[str, str] | None = None,
    alpha: float = 0.7,
    size: float = 38,
    trendline: bool = False,
    trend_color: str = "#e31a1c",
    aspect: tuple[int, int] | None = (16, 9),
    fig_width: float = 10.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title_fontsize: float = 15,
    label_fontsize: float = 11,
    tick_fontsize: float = 10,
    legend_fontsize: float = 10,
    grid: bool = True,
    ax=None,
):
    """Wykres punktowy; color_by dzieli punkty na kategorie wg category_colors."""
    data = df[[c for c in {x_col, y_col, color_by} if c]].dropna().copy()

    if ax is None:
        height = fig_width / (aspect[0] / aspect[1]) if aspect else fig_width * 0.6
        fig, ax = plt.subplots(figsize=(fig_width, height))
    else:
        fig = ax.figure

    if color_by:
        palette = category_colors or {}
        for cat, grp in data.groupby(color_by):
            ax.scatter(grp[x_col], grp[y_col], s=size, alpha=alpha,
                       color=palette.get(cat, color), edgecolor="white",
                       linewidth=0.5, label=f"{cat} ({len(grp)})", zorder=3)
        ax.legend(fontsize=legend_fontsize, frameon=False)
    else:
        ax.scatter(data[x_col], data[y_col], s=size, alpha=alpha, color=color,
                   edgecolor="white", linewidth=0.5, zorder=3)

    if trendline and len(data) > 1:
        b, a = np.polyfit(data[x_col], data[y_col], 1)
        xs = np.linspace(data[x_col].min(), data[x_col].max(), 100)
        ax.plot(xs, a + b * xs, color=trend_color, linewidth=2, zorder=4,
                label=f"OLS: y = {a:.3f} + {b:.3f}x")
        ax.legend(fontsize=legend_fontsize, frameon=False)

    if grid:
        ax.grid(color="#e5e5e3", linewidth=0.8)
        ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#52514e")
    ax.tick_params(labelsize=tick_fontsize, colors="#52514e")
    ax.set_xlabel(xlabel or x_col, fontsize=label_fontsize, color="#52514e")
    ax.set_ylabel(ylabel or y_col, fontsize=label_fontsize, color="#52514e")
    if title:
        ax.set_title(title, fontsize=title_fontsize, loc="left", color="#0b0b0b")
    fig.tight_layout()
    return fig, ax


def distribution_table(series: pd.Series, *, label: str = "value",
                       decimals: int = 2) -> pd.DataFrame:
    """Rozklad zmiennej: n, srednia, odchylenie, min, Q25, mediana, Q75, max."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    out = pd.DataFrame({
        "statystyka": ["n", "mean", "std", "min", "Q25", "median", "Q75", "max"],
        label: [len(s), s.mean(), s.std(ddof=1), s.min(),
                s.quantile(0.25), s.median(), s.quantile(0.75), s.max()],
    })
    out[label] = out[label].round(decimals)
    return out


def causal_effects_table(uplifts_long: pd.DataFrame,
                         treatment_status: pd.DataFrame,
                         models_meta: pd.DataFrame,
                         *, decimals: int = 4) -> pd.DataFrame:
    """
    ATE i ATT z 95% przedzialem ufnosci, osobno dla kazdego modelu i ramienia.

    ATE — srednia po wszystkich kwartalach, na ktorych model byl szacowany.
    ATT — srednia po kwartalach faktycznie poddanych danemu oddzialywaniu.
    Przedzial: srednia +/- 1.96 * sd / sqrt(n), czyli rozrzut efektow miedzy
    kwartalami, nie niepewnosc estymacji samego lasu.
    """
    df = uplifts_long.merge(treatment_status, on="block_id", how="inner")
    targets = models_meta.set_index("model_id")["target_id"].to_dict()
    treated_flag = {"d1nq": "treated_d1nq", "1nq": "treated_1nq"}

    rows = []
    for (model_id, treatment), grp in df.groupby(["model_id", "treatment"]):
        for estimand, subset in (("ATE", grp),
                                 ("ATT", grp[grp[treated_flag[treatment]] == 1])):
            vals = subset["uplift"].dropna()
            n = len(vals)
            if n == 0:
                continue
            mean = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            lo, hi = (mean - 1.96 * se, mean + 1.96 * se) if n > 1 else (np.nan, np.nan)
            rows.append({
                "model_id": model_id,
                "target_id": targets.get(model_id, "?"),
                "treatment": treatment,
                "estimand": estimand,
                "n": n,
                "estimate": mean,
                "se": se,
                "ci_low": lo,
                "ci_high": hi,
                "significant": bool(lo > 0 or hi < 0) if n > 1 else False,
            })

    out = pd.DataFrame(rows).sort_values(["model_id", "treatment", "estimand"])
    num = ["estimate", "se", "ci_low", "ci_high"]
    out[num] = out[num].round(decimals)
    return out.reset_index(drop=True)


def shap_summary(shap_df: pd.DataFrame,
                 model_id: str,
                 *,
                 scale: float = 1e6,
                 labels: dict[str, str] | None = None,
                 drop_var: str = "base_value") -> pd.DataFrame:
    """
    Podsumowanie wartosci SHAP na predyktor, posortowane wg waznosci malejaco.

    mean_shap     — srednia wartosc ZE ZNAKIEM: w ktora strone predyktor
                    przesuwa prognoze (dodatnia podnosi przewidywany koszt)
    mean_abs_shap — srednia wartosc BEZWZGLEDNA: ile predyktor wnosi
                    niezaleznie od kierunku, czyli jego waznosc
    kolumny *_mln — obie wielkosci podzielone przez `scale`

    drop_var usuwa wiersz wyrazu wolnego modelu, ktory nie jest predyktorem.
    """
    df = shap_df[(shap_df["model_id"] == model_id) &
                 (shap_df["var_id"] != drop_var)]
    if df.empty:
        raise ValueError(f"Brak wartosci SHAP dla modelu {model_id}")

    out = (
        df.assign(abs_value=df["value"].abs())
        .groupby("var_id", as_index=False)
        .agg(mean_shap=("value", "mean"),
             mean_abs_shap=("abs_value", "mean"),
             n_blocks=("value", "size"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    out["mean_shap_mln"]     = out["mean_shap"] / scale
    out["mean_abs_shap_mln"] = out["mean_abs_shap"] / scale
    out["label"] = out["var_id"].map(
        lambda v: f"{v}  —  {labels[v][:60]}" if labels and labels.get(v) else v)
    return out
