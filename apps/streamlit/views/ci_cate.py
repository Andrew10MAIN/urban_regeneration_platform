"""
Causal Inference › CATE (uplift vs confounder scatter) dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

from apps.streamlit import config as C
from apps.streamlit.data_loader import (
    load_models,
    load_uplifts,
    load_uplifts_optimization,
    load_features,
    load_mined_variables,
    get_urban_blocks_treatment,
    get_available_years,
    get_meta_dict,
    get_cf_models_dict,
)
from apps.streamlit.components.export import download_button_fig


def render(interactive: bool = True) -> None:
    st.header("CI — CATE (uplift vs confounder)")

    meta          = get_meta_dict()
    years         = get_available_years()
    cf_dict       = get_cf_models_dict()   # {model_id: target_id}
    features_df   = load_features()

    if not cf_dict:
        st.warning("No CF models in database.")
        return

    with st.sidebar:
        st.subheader("CATE settings")
        target_ids = list(cf_dict.values())
        sel_target = st.selectbox(
            "Target variable", target_ids,
            format_func=lambda v: meta.get(v, v), key="cate_target",
        )
        sel_treatment = st.selectbox(
            "Treatment type", list(C.TREATMENT_SHORTCUT.keys()), key="cate_treat",
        )
        post_period = st.selectbox(
            "Treatment reference year", years, index=len(years) - 1, key="cate_post",
        )

        inv = {v: k for k, v in cf_dict.items()}
        sel_model_id = inv.get(sel_target)

        confounder_dict: dict[str, int] = {}
        if sel_model_id:
            sub = features_df[features_df["model_id"] == sel_model_id]
            confounder_dict = sub.set_index("var_id")["year"].to_dict()

        confounder_list = list(confounder_dict.keys())
        if confounder_list:
            chosen_confounder = st.selectbox(
                "Confounder", confounder_list,
                format_func=lambda v: meta.get(v, v), key="cate_conf",
            )
        else:
            st.info("No confounders found for this model.")
            return

        point_colour = st.selectbox("Point color", list(C.COLOURS.keys()), index=4, key="cate_pc")
        add_ols      = st.checkbox("OLS line", value=True, key="cate_ols")
        point_size   = st.slider("Point size", 10, 200, 50, key="cate_ps")
        alpha        = st.slider("Opacity", 0.1, 1.0, 0.7, key="cate_a")

    if sel_model_id is None:
        st.warning("No model found for this target.")
        return

    treatment_code = C.TREATMENT_SHORTCUT[sel_treatment]
    treatment_num  = C.TREATMENT_NUMBER[sel_treatment]
    conf_year      = confounder_dict[chosen_confounder]

    # ── Data ──────────────────────────────────────────────────────────────
    uplifts_df    = load_uplifts()
    uplifts_opt   = load_uplifts_optimization()
    ub_gdf        = get_urban_blocks_treatment(post_period)
    mined         = load_mined_variables()

    opt_keys = uplifts_opt[["model_id", "treatment"]].drop_duplicates()
    uplifts_filtered = (
        uplifts_df[uplifts_df["model_id"].isin(list(cf_dict.keys()))]
        .merge(opt_keys, on=["model_id", "treatment"], how="inner")
        .copy()
    )
    uplifts_filtered["target_id"] = uplifts_filtered["model_id"].map(cf_dict)

    sel_uplifts = (
        uplifts_filtered[
            (uplifts_filtered["target_id"] == sel_target)
            & (uplifts_filtered["treatment"] == treatment_code)
        ][["block_id", "uplift"]]
        .rename(columns={"uplift": sel_target})
        .reset_index(drop=True)
    )

    # keep only treated blocks of the correct type
    treated_blocks = ub_gdf[ub_gdf["treated_all"] == treatment_num]["block_id"]
    sel_uplifts = sel_uplifts[sel_uplifts["block_id"].isin(treated_blocks)]

    conf_df = (
        mined[
            (mined["var_id"] == chosen_confounder)
            & (mined["year"] == pd.Timestamp(f"{conf_year}-01-01"))
        ]
        .rename(columns={"value": chosen_confounder})
        .drop(columns=["var_id", "year"])
        .reset_index(drop=True)
    )

    cate_df = sel_uplifts.merge(conf_df, on="block_id", how="left")

    if cate_df.empty or chosen_confounder not in cate_df.columns:
        st.warning("No data for the selected combination.")
        return

    # ── Plot ──────────────────────────────────────────────────────────────
    data = cate_df[[chosen_confounder, sel_target]].dropna()
    color = C.COLOURS[point_colour]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[chosen_confounder], data[sel_target],
               alpha=alpha, s=point_size, color=color)

    if add_ols and len(data) >= 2:
        X = data[[chosen_confounder]].values
        y = data[sel_target].values
        reg = LinearRegression().fit(X, y)
        x_rng = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        ax.plot(x_rng, reg.predict(x_rng), color="black", linewidth=1.5, label="OLS")
        ax.legend(fontsize=9)

    ax.set_xlabel(meta.get(chosen_confounder, chosen_confounder), fontsize=7)
    ax.set_ylabel(f"Uplift — {meta.get(sel_target, sel_target)}", fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)
    download_button_fig(
        fig,
        filename=f"cate_{sel_target}_{chosen_confounder}_{sel_treatment}.png",
    )
