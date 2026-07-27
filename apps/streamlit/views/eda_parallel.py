"""
EDA › Parallel Trends dashboard.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from apps.streamlit import config as C
from apps.streamlit.data_loader import (
    load_mined_variables,
    load_urban_blocks_ng,
    get_multi_year_vars,
    get_available_years,
    get_meta_dict,
)
from apps.streamlit.components.export import download_button_fig


def _parallel_trends(
    df: pd.DataFrame,
    target_col: str,
    post_period: int,
    groups_dict: dict,
    regen_start: bool,
    regen_end: bool,
    figsize: tuple,
) -> plt.Figure:
    if "year_dt" not in df.columns:
        df = df.copy()
        df["year_dt"] = pd.to_datetime(df["year"])
    else:
        df = df.copy()

    years = sorted(df["year_dt"].dt.year.unique())

    avg_rows = []
    for yr in years:
        yrts = pd.Timestamp(f"{yr}-01-01")
        row = {"year": yr}
        sub = df[df["year_dt"] == yrts]
        for g in groups_dict:
            row[g] = sub[sub["treated_all"] == g][target_col].mean()
        avg_rows.append(row)
    avg_df = pd.DataFrame(avg_rows)

    fig, ax = plt.subplots(figsize=figsize)
    legend_lines, legend_labels = [], []

    for g, (color, ls, label) in groups_dict.items():
        (line,) = ax.plot(
            avg_df["year"], avg_df[g],
            color=color, linestyle=ls, linewidth=2, label=label,
        )
        legend_lines.append(line)
        legend_labels.append(label)

    y_max = avg_df[[g for g in groups_dict]].max().max()
    ax.set_ylim(bottom=0, top=y_max * 1.1)
    ax.set_xticks(years)
    ax.set_xlabel("Year", fontsize=10)
    ax.grid(False)

    if regen_start:
        vl = ax.axvline(C.REGEN_START, color="tab:blue", linestyle="--", linewidth=2)
        legend_lines.append(vl)
        legend_labels.append("Start of regeneration programme")

    if regen_end:
        vl = ax.axvline(C.REGEN_END, color="tab:blue", linestyle="--", linewidth=2)
        legend_lines.append(vl)
        legend_labels.append("End of regeneration programme")

    ax.legend(
        legend_lines, legend_labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    return fig


def render(interactive: bool = True) -> None:
    st.header("EDA — Parallel Trends")

    meta = get_meta_dict()
    multi_vars = get_multi_year_vars()
    years = get_available_years()

    with st.sidebar:
        st.subheader("Parallel Trends settings")
        target_col = st.selectbox(
            "Target variable", multi_vars,
            format_func=lambda v: meta.get(v, v), key="pt_target",
        )
        post_period = st.selectbox(
            "Treatment reference year", years, index=len(years) - 1, key="pt_post",
        )
        regen_start = st.checkbox("Show regeneration start line", value=True)
        regen_end   = st.checkbox("Show regeneration end line",   value=True)
        figsize_w   = st.slider("Figure width",  8, 24, 18)
        figsize_h   = st.slider("Figure height", 4, 12, 7)

    mined = load_mined_variables()
    ub_ng = load_urban_blocks_ng()

    panel = (
        mined[mined["var_id"] == target_col]
        .rename(columns={"value": target_col})
        .drop(columns="var_id")
        .copy()
    )
    treated_ref = ub_ng[
        ub_ng["year"] == pd.Timestamp(f"{post_period}-01-01")
    ][["block_id", "treated_all"]].reset_index(drop=True)
    panel = panel.merge(treated_ref, on="block_id", how="left")

    groups = {
        g: (color, ls, label)
        for g, (color, ls, label) in C.PT_GROUPS.items()
    }

    fig = _parallel_trends(
        panel, target_col, post_period,
        groups_dict=groups,
        regen_start=regen_start,
        regen_end=regen_end,
        figsize=(figsize_w, figsize_h),
    )
    st.pyplot(fig)
    download_button_fig(fig, filename=f"parallel_trends_{target_col}.png", dpi=150)
