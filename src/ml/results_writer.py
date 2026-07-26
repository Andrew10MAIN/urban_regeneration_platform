"""
Persist ML results to the results.* tables in urban_db.

Tables written by save_results() (per CF model):
  results.models           — df_model     (ON CONFLICT DO UPDATE)
  results.hyperparameters  — hyper_df     (ON CONFLICT DO UPDATE)
  results.features         — features_df  (ON CONFLICT DO UPDATE)
  results.uplifts          — uplift_long  (ON CONFLICT DO UPDATE)

Tables written by save_optimization():
  results.uplifts_optimization  — uplifts_final  (ON CONFLICT DO UPDATE)
  results.optimization_summary  — summary        (ON CONFLICT DO UPDATE)
"""

from __future__ import annotations

import logging
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

log = logging.getLogger(__name__)

BATCH = 500


# ── CF model results ────────────────────────────────────────────────────────────

def _upsert_models(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.models
            (model_id, model_type, random_seed, target_id, pre_period, post_period, run_at)
        VALUES
            (:model_id, :model_type, :random_seed, :target_id, :pre_period, :post_period, :run_at)
        ON CONFLICT (model_id) DO UPDATE SET
            model_type  = EXCLUDED.model_type,
            random_seed = EXCLUDED.random_seed,
            target_id   = EXCLUDED.target_id,
            pre_period  = EXCLUDED.pre_period,
            post_period = EXCLUDED.post_period,
            run_at      = EXCLUDED.run_at
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    log.info("results.models: upserted %d row(s)", len(rows))
    return len(rows)


def _upsert_hyperparameters(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.hyperparameters (model_id, hyper_parameter, value)
        VALUES (:model_id, :hyper_parameter, :value)
        ON CONFLICT (model_id, hyper_parameter) DO UPDATE SET
            value = EXCLUDED.value
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    log.info("results.hyperparameters: upserted %d row(s)", len(rows))
    return len(rows)


def _upsert_features(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.features (model_id, feature_no, var_id, year)
        VALUES (:model_id, :feature_no, :var_id, :year)
        ON CONFLICT (model_id, feature_no) DO UPDATE SET
            var_id = EXCLUDED.var_id,
            year   = EXCLUDED.year
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    log.info("results.features: upserted %d row(s)", len(rows))
    return len(rows)


def _upsert_uplifts(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.uplifts (block_id, model_id, treatment, uplift)
        VALUES (:block_id, :model_id, :treatment, :uplift)
        ON CONFLICT (block_id, model_id, treatment) DO UPDATE SET
            uplift = EXCLUDED.uplift
    """)
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            conn.execute(sql, rows[i : i + BATCH])
            total += len(rows[i : i + BATCH])
    log.info("results.uplifts: upserted %d row(s)", total)
    return total


def save_results(result, engine: Engine) -> dict[str, int]:
    """Persist a single CF model run to the database."""
    log.info("Saving CF model results [%s]...", result.df_model["model_id"].iloc[0])
    return {
        "results.models":          _upsert_models(result.df_model, engine),
        "results.hyperparameters": _upsert_hyperparameters(result.hyper_df, engine),
        "results.features":        _upsert_features(result.features_df, engine),
        "results.uplifts":         _upsert_uplifts(result.uplift_long, engine),
    }


# ── Optimization results ────────────────────────────────────────────────────────

def _upsert_uplifts_optimization(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.uplifts_optimization
            (optimization_id, model_id, block_id, treatment, uplift)
        VALUES
            (:optimization_id, :model_id, :block_id, :treatment, :uplift)
        ON CONFLICT (optimization_id, model_id, block_id, treatment) DO UPDATE SET
            uplift = EXCLUDED.uplift
    """)
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            conn.execute(sql, rows[i : i + BATCH])
            total += len(rows[i : i + BATCH])
    log.info("results.uplifts_optimization: upserted %d row(s)", total)
    return total


def _upsert_optimization_summary(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.optimization_summary
            (optimization_id, cost_used, cost_limit, treatment_spillovers,
             clusters_number, time_limit, run_at)
        VALUES
            (:optimization_id, :cost_used, :cost_limit, :treatment_spillovers,
             :clusters_number, :time_limit, :run_at)
        ON CONFLICT (optimization_id) DO UPDATE SET
            cost_used            = EXCLUDED.cost_used,
            cost_limit           = EXCLUDED.cost_limit,
            treatment_spillovers = EXCLUDED.treatment_spillovers,
            clusters_number      = EXCLUDED.clusters_number,
            time_limit           = EXCLUDED.time_limit,
            run_at               = EXCLUDED.run_at
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    log.info("results.optimization_summary: upserted %d row(s)", len(rows))
    return len(rows)


def _upsert_optimization_weights(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.optimization_weights_setup (optimization_id, model_id, weight)
        VALUES (:optimization_id, :model_id, :weight)
        ON CONFLICT (optimization_id, model_id) DO UPDATE SET
            weight = EXCLUDED.weight
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    log.info("results.optimization_weights_setup: upserted %d row(s)", len(rows))
    return len(rows)


def save_optimization(result, engine: Engine) -> dict[str, int]:
    """Persist optimization results to the database."""
    opt_id = result.summary["optimization_id"].iloc[0]
    log.info("Saving optimization results [%s]...", opt_id)
    return {
        "results.optimization_summary":      _upsert_optimization_summary(result.summary, engine),
        "results.optimization_weights_setup": _upsert_optimization_weights(result.weights_df, engine),
        "results.uplifts_optimization":       _upsert_uplifts_optimization(result.uplifts_final, engine),
    }
