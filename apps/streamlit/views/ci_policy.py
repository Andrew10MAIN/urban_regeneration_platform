"""
Causal Inference › Policy Recommendations dashboard.
Left: Folium heatmap of selected model's optimized uplifts.
Right top: bar chart (total uplift by treatment type).
Right bottom: budget pie chart.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from streamlit_folium import st_folium

from apps.streamlit import config as C
from apps.streamlit.data_loader import (
    load_models,
    load_uplifts_optimization,
    load_optimization_summary,
    load_urban_blocks_gdf,
    get_meta_dict,
    get_cf_models_dict,
)
from apps.streamlit.components.map_utils import build_folium_map, render_static
from apps.streamlit.components.export import download_button_map, download_button_fig


def _bar_chart(df, treatment_col: str, uplift_col: str, colour, meta: dict, target_id: str) -> plt.Figure:
    rev = {v: k for k, v in C.TREATMENT_NUMBER.items()}
    bars_df = df.groupby(treatment_col)[uplift_col].sum().reset_index()
    bars_df = bars_df[bars_df[treatment_col] != 0].copy()
    bars_df[uplift_col] = np.floor(bars_df[uplift_col])

    if isinstance(colour, tuple):
        hex_color = mcolors.to_hex(colour[:3])
    else:
        hex_color = colour

    fig, ax = plt.subplots(figsize=(5, 5))
    labels = bars_df[treatment_col].map(rev)
    bars = ax.bar(
        labels, bars_df[uplift_col],
        color=[hex_color] * len(bars_df),
        edgecolor="black", linewidth=1.2, width=0.6,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.0f}",
                ha="center", va="bottom", fontsize=12)
    ax.set_xlabel("Treatment", fontsize=10)
    ax.set_ylabel(meta.get(target_id, target_id), fontsize=7)
    ax.set_title("Estimated uplift by treatment", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _pie_chart(cost_limit: float, cost_used: float) -> plt.Figure:
    remaining = max(cost_limit - cost_used, 0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [cost_used, remaining],
        labels=["Used", "Remaining"],
        colors=[C.PREZ_BLUE, C.PREZ_RED],
        autopct="%.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 0.1},
        textprops={"fontsize": 13},
    )
    ax.set_title("Budget utilization", fontsize=14)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def render(interactive: bool = True) -> None:
    st.header("CI — Policy Recommendations")

    meta    = get_meta_dict()
    cf_dict = get_cf_models_dict()

    if not cf_dict:
        st.warning("No CF models in database.")
        return

    uplifts_opt  = load_uplifts_optimization()
    opt_summary  = load_optimization_summary()
    ub_gdf       = load_urban_blocks_gdf()

    opt_ids = uplifts_opt["optimization_id"].unique().tolist()
    if not opt_ids:
        st.warning("No optimization results found.")
        return

    with st.sidebar:
        st.subheader("Policy settings")
        sel_opt_id = st.selectbox("Optimization run", opt_ids, key="pol_opt")
        model_ids_in_opt = (
            uplifts_opt[uplifts_opt["optimization_id"] == sel_opt_id]["model_id"]
            .unique().tolist()
        )
        colour_cycle = C.MODEL_COLOUR_CYCLE
        sel_model_id = st.selectbox(
            "Model to visualise", model_ids_in_opt,
            format_func=lambda m: f"{m} → {meta.get(cf_dict.get(m,''), cf_dict.get(m,''))}",
            key="pol_model",
        )
        heatmap_colour = st.selectbox(
            "Colormap", list(C.HEATMAP_CMAPS.keys()), key="pol_cmap",
        )

    target_id = cf_dict.get(sel_model_id, sel_model_id)

    # ── Build block-level GDF ─────────────────────────────────────────────
    sel_df = (
        uplifts_opt[
            (uplifts_opt["optimization_id"] == sel_opt_id)
            & (uplifts_opt["model_id"] == sel_model_id)
        ][["block_id", "uplift", "treatment"]]
        .copy()
    )

    ub_merged = (
        ub_gdf[["block_id", "geometry"]]
        .merge(sel_df, on="block_id", how="left")
    )
    ub_merged["uplift"] = ub_merged["uplift"].fillna(0)
    ub_merged["treatment"] = ub_merged["treatment"].fillna("0")

    # numeric treatment for outlines
    ub_merged["treatment_num"] = 0
    ub_merged.loc[ub_merged["treatment"] == "1nq",  "treatment_num"] = 1
    ub_merged.loc[ub_merged["treatment"] == "d1nq", "treatment_num"] = 2

    # ── Layout ────────────────────────────────────────────────────────────
    col_map, col_charts = st.columns([3, 2], gap="medium")

    with col_map:
        if interactive:
            m = build_folium_map(
                ub_merged.copy(), "uplift",
                treatment_col="treatment_num",
                cmap=C.HEATMAP_CMAPS[heatmap_colour],
            )
            st_folium(m, width=None, height=580, returned_objects=[])
            download_button_map(m, filename=f"policy_{sel_model_id}.html")
        else:
            fig_m = render_static(
                ub_merged.copy(), "uplift",
                treatment_col="treatment_num",
                cmap=C.HEATMAP_CMAPS[heatmap_colour],
                title=f"Optimized blocks — {meta.get(target_id, target_id)}",
            )
            st.pyplot(fig_m)
            download_button_fig(fig_m, filename=f"policy_{sel_model_id}.png")

    with col_charts:
        st.subheader("Uplift by treatment type")
        colour = C.COLOURS.get(heatmap_colour, C.PREZ_BLUE)
        fig_bar = _bar_chart(ub_merged, "treatment_num", "uplift", colour, meta, target_id)
        st.pyplot(fig_bar)
        download_button_fig(fig_bar, filename=f"uplift_bar_{sel_model_id}.png")

        st.subheader("Budget utilisation")
        if not opt_summary.empty:
            row = opt_summary[opt_summary["optimization_id"] == sel_opt_id].iloc[0]
            fig_pie = _pie_chart(float(row["cost_limit"]), float(row["cost_used"]))
            st.pyplot(fig_pie)
            download_button_fig(fig_pie, filename=f"budget_pie_{sel_opt_id}.png")

            st.metric("Budget limit (PLN)", f"{row['cost_limit']:,.0f}")
            st.metric("Budget used (PLN)",  f"{row['cost_used']:,.0f}")
            st.metric("Remaining (PLN)",    f"{row['cost_limit'] - row['cost_used']:,.0f}")
