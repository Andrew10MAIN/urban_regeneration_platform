"""
Causal Inference › Uplifts heatmap dashboard.
"""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from apps.streamlit import config as C
from apps.streamlit.data_loader import (
    load_models,
    load_uplifts,
    load_uplifts_optimization,
    get_urban_blocks_treatment,
    get_available_years,
    get_meta_dict,
    get_cf_models_dict,
)
from apps.streamlit.components.map_utils import build_folium_map, render_static
from apps.streamlit.components.export import download_button_map, download_button_fig


def render(interactive: bool = True) -> None:
    st.header("CI — Uplifts Heatmap")

    meta = get_meta_dict()
    years = get_available_years()
    cf_dict = get_cf_models_dict()   # {model_id: target_id}

    if not cf_dict:
        st.warning("No CF models found in the database.")
        return

    treatment_shortcut = C.TREATMENT_SHORTCUT

    with st.sidebar:
        st.subheader("Uplift settings")
        target_ids = list(cf_dict.values())
        sel_target = st.selectbox(
            "Target variable", target_ids,
            format_func=lambda v: meta.get(v, v), key="upl_target",
        )
        sel_treatment = st.selectbox(
            "Treatment type", list(treatment_shortcut.keys()), key="upl_treat",
        )
        heatmap_colour = st.selectbox(
            "Colormap", list(C.HEATMAP_CMAPS.keys()), key="upl_cmap",
        )
        post_period = st.selectbox(
            "Treatment reference year", years, index=len(years) - 1, key="upl_post",
        )

    # ── Data ──────────────────────────────────────────────────────────────
    uplifts_df       = load_uplifts()
    uplifts_opt_df   = load_uplifts_optimization()
    ub_gdf           = get_urban_blocks_treatment(post_period)

    # resolve model_id for selected target
    inv = {v: k for k, v in cf_dict.items()}
    sel_model_id = inv.get(sel_target)
    if sel_model_id is None:
        st.warning("No model found for this target.")
        return

    sel_treatment_code = treatment_shortcut[sel_treatment]

    # filter uplifts to CF models that appear in optimization result
    opt_keys = uplifts_opt_df[["model_id", "treatment"]].drop_duplicates()
    uplifts_filtered = uplifts_df[
        uplifts_df["model_id"].isin(list(cf_dict.keys()))
    ].merge(opt_keys, on=["model_id", "treatment"], how="inner").copy()

    uplifts_filtered["target_id"] = uplifts_filtered["model_id"].map(cf_dict)

    sel_uplifts = (
        uplifts_filtered[
            (uplifts_filtered["target_id"] == sel_target)
            & (uplifts_filtered["treatment"] == sel_treatment_code)
        ][["block_id", "uplift"]]
        .rename(columns={"uplift": sel_target})
        .reset_index(drop=True)
    )

    if sel_uplifts.empty:
        st.warning("No uplifts available for the selected target and treatment type.")
        return

    heatmap_gdf = ub_gdf.merge(sel_uplifts, on="block_id", how="right")

    # ── Render ────────────────────────────────────────────────────────────
    if interactive:
        m = build_folium_map(
            heatmap_gdf, sel_target,
            treatment_col="treated_all",
            cmap=C.HEATMAP_CMAPS[heatmap_colour],
        )
        st_folium(m, width=None, height=620, returned_objects=[])
        download_button_map(m, filename=f"uplifts_{sel_target}_{sel_treatment}.html")
    else:
        fig = render_static(
            heatmap_gdf, sel_target,
            treatment_col="treated_all",
            cmap=C.HEATMAP_CMAPS[heatmap_colour],
            title=f"Uplift — {meta.get(sel_target, sel_target)} ({sel_treatment})",
        )
        st.pyplot(fig)
        download_button_fig(fig, filename=f"uplifts_{sel_target}_{sel_treatment}.png")
