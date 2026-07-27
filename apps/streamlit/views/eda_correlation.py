"""
EDA › Correlation (scatter) dashboard.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

from apps.streamlit import config as C
from apps.streamlit.data_loader import (
    load_mined_variables,
    get_available_years,
    get_vars_for_year,
    get_meta_dict,
)
from apps.streamlit.components.export import download_button_fig


def _scatter(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    xlabel: str,
    ylabel: str,
    point_color,
    add_ols: bool,
    line_color,
    point_size: int,
    alpha: float,
) -> plt.Figure:
    data = df[[col_x, col_y]].dropna()

    def _conv(c):
        if isinstance(c, tuple) and len(c) >= 3:
            return (*c[:3], 1) if len(c) == 3 else c
        return c

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        data[col_x], data[col_y],
        alpha=alpha, s=point_size, color=_conv(point_color),
    )

    if add_ols:
        X = data[[col_x]].values
        y = data[col_y].values
        reg = LinearRegression().fit(X, y)
        x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        ax.plot(x_range, reg.predict(x_range), color=_conv(line_color), linewidth=2, label="OLS")
        ax.legend(fontsize=9)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def render(interactive: bool = True) -> None:
    st.header("EDA — Correlation")

    meta = get_meta_dict()
    years = get_available_years()
    colour_choices = list(C.COLOURS.keys())

    with st.sidebar:
        st.subheader("Correlation settings")
        vis_year = st.selectbox("Year", years, index=len(years) - 1, key="corr_year")
        vars_for_year = get_vars_for_year(vis_year)

        var_x = st.selectbox(
            "X variable", vars_for_year,
            format_func=lambda v: meta.get(v, v), key="corr_x",
        )
        var_y = st.selectbox(
            "Y variable", vars_for_year,
            format_func=lambda v: meta.get(v, v),
            index=min(1, len(vars_for_year) - 1), key="corr_y",
        )
        point_colour = st.selectbox("Point color", colour_choices, index=4, key="corr_pc")
        line_colour  = st.selectbox("OLS line color", colour_choices, index=7, key="corr_lc")
        add_ols      = st.checkbox("Show OLS regression line", value=True)
        point_size   = st.slider("Point size", 10, 200, 50)
        alpha        = st.slider("Opacity", 0.1, 1.0, 0.7)

    mined = load_mined_variables()
    df_pre = (
        mined[
            (mined["year"] == pd.Timestamp(f"{vis_year}-01-01"))
            & (mined["var_id"].isin([var_x, var_y]))
        ]
        .drop(columns=["year"])
        .pivot(index="block_id", columns="var_id", values="value")
        .reset_index()
    )

    if var_x not in df_pre.columns or var_y not in df_pre.columns:
        st.warning("Selected variables not available for this year.")
        return

    fig = _scatter(
        df_pre, var_x, var_y,
        xlabel=meta.get(var_x, var_x),
        ylabel=meta.get(var_y, var_y),
        point_color=C.COLOURS[point_colour],
        add_ols=add_ols,
        line_color=C.COLOURS[line_colour],
        point_size=point_size,
        alpha=alpha,
    )
    st.pyplot(fig)
    download_button_fig(fig, filename=f"correlation_{var_x}_{var_y}_{vis_year}.png")
