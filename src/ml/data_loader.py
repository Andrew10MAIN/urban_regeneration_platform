"""
Data loading and preparation for the CausalForest ML pipeline.

Replicates the "Data preparation" section of ml_experimenting.ipynb.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine

from src.ml.config import ModelConfig

log = logging.getLogger(__name__)


@dataclass
class DataBundle:
    """All arrays and dataframes needed by the model."""
    Y: np.ndarray             # (N, 1) – target: post - pre difference
    X: np.ndarray             # (N, k) – confounders at pre_period
    T: np.ndarray             # (N, 2) – treatment indicators [d1nq, 1nq]
    uplift_base: pd.DataFrame # block_id + treatment columns, sorted by block_id
    actual_confounders: list[str]        # confounders that had data (some may be skipped)
    confounder_years: dict[str, int]     # {var_id: year actually used}
    n_blocks: int


def load_data(config: ModelConfig, engine: Engine) -> DataBundle:
    """
    Load urban_blocks and mined.variables, then build Y / X / T arrays
    exactly as in the notebook.
    """
    log.info("Loading urban_blocks and mined.variables from DB...")

    urban_blocks_ng = pd.read_sql("SELECT * FROM core.urban_blocks", engine)
    mined_variables = pd.read_sql("SELECT * FROM mined.variables", engine)

    log.info(
        "Loaded %d block-year rows from urban_blocks, %d variable rows from mined.variables",
        len(urban_blocks_ng), len(mined_variables),
    )

    # ── Y: target = post_period_value - pre_period_value ─────────────────────
    target = config.target_variable
    pre_ts  = pd.Timestamp(f"{config.pre_period}-01-01")
    post_ts = pd.Timestamp(f"{config.post_period}-01-01")

    target_post = (
        mined_variables[(mined_variables["var_id"] == target) &
                        (mined_variables["year"] == post_ts)]
        [["block_id", "value"]]
        .rename(columns={"value": f"{target}_{config.post_period}"})
        .reset_index(drop=True)
    )
    target_pre = (
        mined_variables[(mined_variables["var_id"] == target) &
                        (mined_variables["year"] == pre_ts)]
        [["block_id", "value"]]
        .rename(columns={"value": f"{target}_{config.pre_period}"})
        .reset_index(drop=True)
    )
    target_merged = target_post.merge(target_pre, on="block_id", how="left")
    target_merged[target] = (
        target_merged[f"{target}_{config.post_period}"] -
        target_merged[f"{target}_{config.pre_period}"]
    )
    target_df = target_merged[["block_id", target]].copy()
    Y = target_df.sort_values("block_id")[[target]].values
    log.info("Y (target) shape: %s", Y.shape)

    # ── X: confounders at pre_period (with ±deviation fallback) ──────────────
    confounder_dfs: list[pd.DataFrame] = []
    actual_confounders: list[str] = []
    confounder_years: dict[str, int] = {}

    for var in config.confounders:
        df_c = mined_variables[
            (mined_variables["var_id"] == var) &
            (mined_variables["year"] == pre_ts)
        ][["block_id", "value"]].rename(columns={"value": var}).copy()
        year_used = config.pre_period

        if df_c.empty and config.pre_period_deviation > 0:
            found = False
            for d in range(1, config.pre_period_deviation + 1):
                for fallback_year in [config.pre_period - d, config.pre_period + d]:
                    df_c = mined_variables[
                        (mined_variables["var_id"] == var) &
                        (mined_variables["year"] == pd.Timestamp(f"{fallback_year}-01-01"))
                    ][["block_id", "value"]].rename(columns={"value": var}).copy()
                    if not df_c.empty:
                        year_used = fallback_year
                        log.info(
                            "Confounder %s: used data from year %d (pre_period fallback)",
                            var, fallback_year,
                        )
                        found = True
                        break
                if found:
                    break

        if df_c.empty:
            log.warning(
                "Confounder %s: no data within ±%d years of %d — skipped",
                var, config.pre_period_deviation, config.pre_period,
            )
            continue

        confounder_dfs.append(df_c)
        actual_confounders.append(var)
        confounder_years[var] = year_used

    if confounder_dfs:
        merged_confounders_pre = reduce(
            lambda l, r: l.merge(r, on="block_id", how="left"),
            confounder_dfs,
        )
    else:
        raise ValueError("No confounder data available — cannot train model.")

    # ── Align X to blocks present in target_df ───────────────────────────────
    target_block_ids = target_df["block_id"].unique()
    merged_confounders = merged_confounders_pre[
        merged_confounders_pre["block_id"].isin(target_block_ids)
    ].copy()

    X = (
        merged_confounders
        .sort_values("block_id")
        .drop(columns=["block_id"])
        .values
    )
    log.info("X (confounders) shape: %s  features: %s", X.shape, actual_confounders)

    # ── T: treatment indicators at post_period ────────────────────────────────
    uplift_base_pre = (
        urban_blocks_ng[urban_blocks_ng["year"] == post_ts]
        .sort_values("block_id")
        .reset_index(drop=True)
        .drop(columns=["year", "treated_all"])
        .copy()
    )

    # ── Align T and uplift_base to blocks present in target_df ───────────────
    uplift_base = uplift_base_pre[
        uplift_base_pre["block_id"].isin(target_block_ids)
    ].copy()

    T = uplift_base[["treated_d1nq", "treated_1nq"]].reset_index(drop=True).to_numpy()
    log.info(
        "T (treatment) shape: %s  |  d1nq treated: %d  |  1nq treated: %d",
        T.shape,
        (T[:, 0] == 1).sum(),
        (T[:, 1] == 1).sum(),
    )

    # ── Sanity check: Y / X / T must have identical row counts ───────────────
    if not (Y.shape[0] == X.shape[0] == T.shape[0]):
        raise ValueError(
            f"Shape mismatch after alignment: Y={Y.shape[0]}, X={X.shape[0]}, T={T.shape[0]}. "
            "Check that target_variable, confounders and urban_blocks share the same block_ids."
        )
    log.info("Alignment OK — Y/X/T all have %d rows", Y.shape[0])

    return DataBundle(
        Y=Y,
        X=X,
        T=T,
        uplift_base=uplift_base,
        actual_confounders=actual_confounders,
        confounder_years=confounder_years,
        n_blocks=len(uplift_base),
    )
