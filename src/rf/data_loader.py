"""
Data loading and preparation for the RF Regeneration Cost pipeline.

Replicates the "Loading data" and "Data preparation" sections of
rf_experimenting.ipynb.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.rf.config import RFConfig

log = logging.getLogger(__name__)


@dataclass
class RFDataBundle:
    """All arrays and metadata needed by the RF model."""
    X_train: pd.DataFrame           # features for training (index = block_id)
    X_pred: pd.DataFrame            # features for prediction (index = block_id)
    y_train: pd.Series              # actual costs for trained blocks
    block_ids_predict: list[int]    # block_ids of untreated blocks
    feature_names: list[str]        # ordered list of feature column names
    feature_years: dict[str, int]   # {var_id: year used}
    n_train: int
    n_predict: int


def load_data(config: RFConfig, engine: Engine) -> RFDataBundle:
    """
    Load urban_blocks, mined_variables and regeneration.actions,
    then build train/predict splits exactly as in rf_experimenting.ipynb.
    """
    log.info("Loading urban_blocks_geom from DB...")
    urban_blocks = pd.read_sql(
        "SELECT block_id, area FROM core.urban_blocks_geom", engine
    )
    log.info("  %d blocks loaded", len(urban_blocks))

    log.info("Loading mined.variables from DB...")
    mined_variables = pd.read_sql("SELECT * FROM mined.variables", engine)
    log.info("  %d variable-rows loaded", len(mined_variables))

    log.info("Loading regeneration.actions from DB...")
    regen_actions = pd.read_sql("SELECT block_id, costs FROM regeneration.actions", engine)
    regen_costs_agg = (
        regen_actions
        .groupby("block_id")["costs"]
        .sum()
        .reset_index()
    )
    regeneration_real_costs = (
        urban_blocks
        .merge(regen_costs_agg, on="block_id", how="left")
        .fillna({"costs": 0.0})
        .copy()
    )
    log.info(
        "  %d blocks with real costs, %d without",
        (regeneration_real_costs["costs"] != 0).sum(),
        (regeneration_real_costs["costs"] == 0).sum(),
    )

    # ── Build main feature matrix from mined.variables at pre_period ─────────
    pre_ts = pd.Timestamp(f"{config.pre_period}-01-01")

    main_feat_pre = (
        mined_variables[mined_variables["year"] == pre_ts]
        .reset_index(drop=True)
        .drop(columns=["year"])
    )
    # exclude unreliable / unavailable variables
    main_feat_pre = main_feat_pre[
        ~main_feat_pre["var_id"].isin(config.excluded_vars)
    ].copy()

    main_feat_df = (
        main_feat_pre
        .pivot_table(index="block_id", columns="var_id", values="value", aggfunc="first")
        .reset_index()
    )
    main_feat_df.columns.name = None

    # ── Special-year variables (e.g. socVrPopt from 2021) ────────────────────
    for var_id, spec_year in config.special_year_vars.items():
        spec_ts = pd.Timestamp(f"{spec_year}-01-01")
        spec_df = (
            mined_variables[
                (mined_variables["year"] == spec_ts) &
                (mined_variables["var_id"] == var_id)
            ]
            .reset_index(drop=True)
            .rename(columns={"value": var_id})
            [["block_id", var_id]]
        )
        if spec_df.empty:
            log.warning("Special-year var %s at %d not found — skipped", var_id, spec_year)
            continue
        # drop the pre_period version if it was already included
        if var_id in main_feat_df.columns:
            main_feat_df = main_feat_df.drop(columns=[var_id])
        main_feat_df = main_feat_df.merge(spec_df, on="block_id", how="left")
        log.info("Special-year var %s loaded from %d", var_id, spec_year)

    # ── Combine: area + special-year + main features ─────────────────────────
    model_df = (
        urban_blocks[["block_id", "area"]]
        .merge(main_feat_df, on="block_id", how="left")
        .sort_values("block_id")
        .reset_index(drop=True)
    )
    feature_names = [c for c in model_df.columns if c != "block_id"]

    # Build feature_years dict: default = pre_period, override with special_year_vars
    feature_years: dict[str, int] = {}
    for fn in feature_names:
        feature_years[fn] = config.special_year_vars.get(fn, config.pre_period)

    log.info("Feature matrix shape: %s  (%d features)", model_df.shape, len(feature_names))

    # ── Train / predict split ─────────────────────────────────────────────────
    train_block_ids = (
        regeneration_real_costs[regeneration_real_costs["costs"] != 0]["block_id"].unique()
    )
    predict_block_ids = (
        regeneration_real_costs[regeneration_real_costs["costs"] == 0]["block_id"].unique()
    )

    model_df_train = (
        model_df[model_df["block_id"].isin(train_block_ids)]
        .reset_index(drop=True)
    )
    model_df_predict = (
        model_df[model_df["block_id"].isin(predict_block_ids)]
        .reset_index(drop=True)
    )

    costs_train = (
        regeneration_real_costs[regeneration_real_costs["costs"] != 0]
        .sort_values("block_id")
        .reset_index(drop=True)
    )

    X_train = model_df_train.set_index("block_id")
    X_pred  = model_df_predict.set_index("block_id")
    y_train = costs_train.set_index("block_id")["costs"].reindex(X_train.index)

    log.info(
        "Train set: %d blocks  |  Predict set: %d blocks",
        len(X_train), len(X_pred),
    )

    return RFDataBundle(
        X_train=X_train,
        X_pred=X_pred,
        y_train=y_train,
        block_ids_predict=list(predict_block_ids),
        feature_names=feature_names,
        feature_years=feature_years,
        n_train=len(X_train),
        n_predict=len(X_pred),
    )
