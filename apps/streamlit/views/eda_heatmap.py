"""
EDA › Heatmap dashboard.
Interactive Folium choropleth with optional point overlays.
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

from apps.streamlit import config as C
from apps.streamlit.data_loader import (
    load_mined_variables,
    load_urban_blocks_gdf,
    get_urban_blocks_treatment,
    get_available_years,
    get_vars_for_year,
    get_meta_dict,
    load_osm_poi,
    load_app_prices,
    load_penalties,
    load_build_perm,
)
from apps.streamlit.components.map_utils import build_folium_map, render_static, fig_to_bytes
from apps.streamlit.components.export import download_button_map, download_button_fig


def _get_point_gdf(source: str, year: int) -> gpd.GeoDataFrame | None:
    if source == "(none)":
        return None

    if source in C.OSM_FCLASS_GROUPS:
        fclass_list = C.OSM_FCLASS_GROUPS[source]
        df = load_osm_poi()
        mask = (
            df["fclass"].isin(fclass_list)
            & (df["date"] <= pd.Timestamp(f"{year}-12-31"))
            & (df["date"] >= pd.Timestamp(f"{year}-01-01"))
        )
        return df[mask].reset_index(drop=True)

    if source in C.PENALTIES_TYPES:
        df = load_penalties()
        mask = (
            (df["pen_type"] == source)
            & (df["date"] <= pd.Timestamp(f"{year}-12-31"))
            & (df["date"] >= pd.Timestamp(f"{year}-01-01"))
        )
        return df[mask].reset_index(drop=True)

    if source == "Build_perm":
        df = load_build_perm()
        mask = (
            (df["date"] <= pd.Timestamp(f"{year}-12-31"))
            & (df["date"] >= pd.Timestamp(f"{year}-01-01"))
        )
        return df[mask].reset_index(drop=True)

    return None


def render(interactive: bool = True) -> None:
    st.header("EDA — Heatmap")

    # ── Sidebar controls ─────────────────────────────────────────────────
    meta = get_meta_dict()
    years = get_available_years()
    layer_choices = (
        ["(none)"]
        + list(C.OSM_FCLASS_GROUPS.keys())
        + list(C.PENALTIES_TYPES)
        + ["Build_perm"]
    )
    colour_choices = list(C.HEATMAP_CMAPS.keys())
    marker_choices = ["circle", "triangle", "square"]
    point_colour_choices = list(C.COLOURS.keys())

    with st.sidebar:
        st.subheader("Heatmap settings")

        vis_year = st.selectbox("Year", years, index=len(years) - 1)
        vars_for_year = get_vars_for_year(vis_year)
        var_id = st.selectbox(
            "Variable",
            vars_for_year,
            format_func=lambda v: meta.get(v, v),
        )
        heatmap_colour = st.selectbox("Colormap", colour_choices, index=0)
        post_period = st.selectbox("Treatment reference year", years, index=len(years) - 1)

        st.subheader("Point layer 1")
        first_src = st.selectbox("Source 1", layer_choices, index=0)
        first_marker = st.selectbox("Marker 1", marker_choices, index=1, key="m1")
        first_colour = st.selectbox("Color 1", point_colour_choices, index=2, key="c1")

        st.subheader("Point layer 2")
        second_src = st.selectbox("Source 2", layer_choices, index=0)
        second_marker = st.selectbox("Marker 2", marker_choices, index=0, key="m2")
        second_colour = st.selectbox("Color 2", point_colour_choices, index=3, key="c2")

        st.subheader("App prices overlay")
        show_prices = st.checkbox("Show apartment prices", value=False)
        price_colour = st.selectbox("Price colormap", list(C.SAT_CMAPS.keys()), index=1)

    # ── Data preparation ─────────────────────────────────────────────────
    mined = load_mined_variables()
    ub_gdf = get_urban_blocks_treatment(post_period)

    temp = (
        mined[
            (mined["var_id"] == var_id)
            & (mined["year"] == pd.Timestamp(f"{vis_year}-01-01"))
        ]
        .rename(columns={"value": var_id})
        .drop(columns=["var_id", "year"])
        .reset_index(drop=True)
    )
    heatmap_gdf = ub_gdf.merge(temp, on="block_id", how="right")

    pt1 = _get_point_gdf(first_src, vis_year)
    pt2 = _get_point_gdf(second_src, vis_year)

    price_gdf = None
    if show_prices:
        app_p = load_app_prices()
        app_p = app_p[
            (app_p["date"] <= pd.Timestamp(f"{vis_year}-12-31"))
            & (app_p["date"] >= pd.Timestamp(f"{vis_year}-01-01"))
        ].dropna(subset=["floor_area", "price_gross"]).copy()
        app_p["price_per_m2"] = app_p["price_gross"] / app_p["floor_area"]
        price_gdf = app_p

    # ── Render ────────────────────────────────────────────────────────────
    if interactive:
        m = build_folium_map(
            heatmap_gdf, var_id,
            treatment_col="treated_all",
            cmap=C.HEATMAP_CMAPS[heatmap_colour],
            point_gdf1=pt1, point_marker1=first_marker,
            point_color1=C.COLOURS[first_colour],
            point_gdf2=pt2, point_marker2=second_marker,
            point_color2=C.COLOURS[second_colour],
            price_gdf=price_gdf,
            price_cmap=C.SAT_CMAPS.get(price_colour),
        )
        st_folium(m, width=None, height=620, returned_objects=[])
        download_button_map(m, filename=f"heatmap_{var_id}_{vis_year}.html")
    else:
        fig = render_static(
            heatmap_gdf, var_id,
            treatment_col="treated_all",
            cmap=C.HEATMAP_CMAPS[heatmap_colour],
            title=meta.get(var_id, var_id),
        )
        st.pyplot(fig)
        download_button_fig(fig, filename=f"heatmap_{var_id}_{vis_year}.png")
