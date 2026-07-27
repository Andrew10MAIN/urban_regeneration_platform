"""
Folium map builder — interactive (st_folium) + static (matplotlib/geopandas).
Adapted from dashbord_prototyping.ipynb :: plot_heatmap_html.
"""

from __future__ import annotations

import io
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import branca.colormap as bcm
import folium
from shapely.ops import unary_union


def build_folium_map(
    gdf: gpd.GeoDataFrame,
    col_to_vis: str,
    *,
    treatment_col: str | None = None,
    cmap=None,
    show_treatment_outline: bool = True,
    show_legend: bool = True,
    point_gdf1: gpd.GeoDataFrame | None = None,
    point_marker1: str | None = None,       # "circle" | "triangle" | "square"
    point_color1=None,
    point_gdf2: gpd.GeoDataFrame | None = None,
    point_marker2: str | None = None,
    point_color2=None,
    price_gdf: gpd.GeoDataFrame | None = None,
    price_cmap=None,
    zoom_start: int = 13,
) -> folium.Map:
    """Return a folium.Map ready for st_folium()."""
    import matplotlib.pyplot as _plt

    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(4326)

    center = gdf.geometry.union_all().centroid

    m = folium.Map(
        location=[center.y, center.x],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    # ── Colormap ──────────────────────────────────────────────────────────
    if cmap is None:
        cmap = _plt.get_cmap("YlOrRd")
    elif isinstance(cmap, str):
        cmap = _plt.get_cmap(cmap)

    vmin = float(gdf[col_to_vis].min())
    vmax = float(gdf[col_to_vis].max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _hex(value):
        return mcolors.to_hex(cmap(norm(value)))

    cmap_obj = bcm.LinearColormap(
        colors=[_hex(vmin), _hex((vmin + vmax) / 4),
                _hex((vmin + vmax) / 2), _hex((3 * vmin + vmax) / 4), _hex(vmax)],
        vmin=vmin, vmax=vmax,
    )
    cmap_obj.caption = col_to_vis

    folium.GeoJson(
        gdf,
        style_function=lambda f: {
            "fillColor": _hex(f["properties"][col_to_vis]),
            "color": "grey",
            "weight": 0.4,
            "fillOpacity": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["block_id", col_to_vis],
            aliases=["Block:", "Value:"],
        ),
    ).add_to(m)

    # ── Treatment outlines ────────────────────────────────────────────────
    if show_treatment_outline and treatment_col and treatment_col in gdf.columns:
        for val, dash in [(1, "10 8"), (2, None)]:
            subset = gdf[gdf[treatment_col] == val]
            if len(subset):
                union = unary_union(subset.geometry)
                style = {"fillOpacity": 0, "color": "black", "weight": 3}
                if dash:
                    style["dashArray"] = dash
                folium.GeoJson(
                    gpd.GeoSeries([union], crs=gdf.crs).__geo_interface__,
                    style_function=lambda x, s=style: s,
                ).add_to(m)

    # ── Point layers ──────────────────────────────────────────────────────
    def _add_points(pt_gdf, marker, color):
        if pt_gdf is None:
            return
        if isinstance(color, tuple):
            r, g, b = color[:3]
            color = mcolors.to_hex((r, g, b))
        pts = pt_gdf.copy()
        if pts.crs is None:
            pts = pts.set_crs(4326)
        pts = pts.to_crs(4326)
        for _, row in pts.iterrows():
            geom = row.geometry
            if geom.geom_type != "Point":
                continue
            if marker == "circle":
                folium.CircleMarker(
                    [geom.y, geom.x], radius=6,
                    color=color, fill=True, fill_color=color, fill_opacity=0.9,
                ).add_to(m)
            elif marker == "triangle":
                folium.Marker(
                    [geom.y, geom.x],
                    icon=folium.DivIcon(
                        html=f'<div style="color:{color};font-size:24px;">▲</div>'
                    ),
                ).add_to(m)
            elif marker == "square":
                folium.Marker(
                    [geom.y, geom.x],
                    icon=folium.DivIcon(
                        html=f'<div style="width:14px;height:14px;background:{color};"></div>'
                    ),
                ).add_to(m)

    _add_points(point_gdf1, point_marker1, point_color1)
    _add_points(point_gdf2, point_marker2, point_color2)

    # ── Price point layer ─────────────────────────────────────────────────
    if price_gdf is not None and "price_per_m2" in price_gdf.columns:
        import matplotlib.pyplot as _plt2
        pcmap = price_cmap if price_cmap is not None else _plt2.get_cmap("YlOrRd")
        pts = price_gdf.copy()
        if pts.crs is None:
            pts = pts.set_crs(4326)
        pts = pts.to_crs(4326)
        p_norm = mcolors.Normalize(
            vmin=pts["price_per_m2"].min(), vmax=pts["price_per_m2"].max()
        )
        for _, row in pts.iterrows():
            geom = row.geometry
            if geom.geom_type != "Point":
                continue
            color = mcolors.to_hex(pcmap(p_norm(row["price_per_m2"])))
            folium.CircleMarker(
                [geom.y, geom.x], radius=7,
                color=color, fill=True, fill_color=color, fill_opacity=0.5, weight=1,
                tooltip=f"Price per m²: {row['price_per_m2']:,.0f}",
            ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────
    if show_legend:
        cmap_obj.add_to(m)
        if show_treatment_outline and treatment_col:
            m.get_root().html.add_child(folium.Element("""
            <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
                        background:white;padding:10px;border:2px solid grey;font-size:14px;">
              <div><span style="border-bottom:3px dashed black;width:40px;display:inline-block;"></span>
                   &nbsp;Treated indirectly</div>
              <div><span style="border-bottom:3px solid black;width:40px;display:inline-block;"></span>
                   &nbsp;Treated directly</div>
            </div>"""))

    return m


def render_static(
    gdf: gpd.GeoDataFrame,
    col_to_vis: str,
    *,
    treatment_col: str | None = None,
    cmap=None,
    figsize: tuple = (10, 8),
    title: str = "",
) -> plt.Figure:
    """Return a matplotlib Figure for PNG export."""
    import matplotlib.pyplot as _plt
    import matplotlib.patches as mpatches

    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(3857)

    if cmap is None:
        cmap = _plt.get_cmap("YlOrRd")

    fig, ax = _plt.subplots(figsize=figsize)
    gdf.plot(
        column=col_to_vis, cmap=cmap, ax=ax,
        edgecolor="grey", linewidth=0.3, legend=True,
        legend_kwds={"label": col_to_vis, "shrink": 0.6},
    )

    if treatment_col and treatment_col in gdf.columns:
        for val, lw, ls, label in [
            (1, 2, "--", "Indirect"),
            (2, 2, "-",  "Direct"),
        ]:
            subset = gdf[gdf[treatment_col] == val]
            if len(subset):
                subset.dissolve().boundary.plot(
                    ax=ax, color="black", linewidth=lw, linestyle=ls, label=label
                )

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()
