"""
Persist RF results to the results.* tables in urban_db.

Tables written:
  results.models               — df_model          (ON CONFLICT DO UPDATE)
  results.hyperparameters      — hyper_df           (ON CONFLICT DO UPDATE)
  results.features             — features_df        (ON CONFLICT DO UPDATE)
  results.shap_rf_reg_price    — shap_long_df       (ON CONFLICT DO UPDATE)
  results.predicted_reg_prices — predicted_prices   (ON CONFLICT DO UPDATE)
"""

from __future__ import annotations

import logging
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

log = logging.getLogger(__name__)

BATCH_SIZE = 500


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


def _upsert_shap(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.shap_rf_reg_price (model_id, block_id, var_id, value)
        VALUES (:model_id, :block_id, :var_id, :value)
        ON CONFLICT (model_id, block_id, var_id) DO UPDATE SET
            value = EXCLUDED.value
    """)
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH_SIZE):
            conn.execute(sql, rows[i : i + BATCH_SIZE])
            total += len(rows[i : i + BATCH_SIZE])
    log.info("results.shap_rf_reg_price: upserted %d row(s)", total)
    return total


def _upsert_predicted_prices(df: pd.DataFrame, engine: Engine) -> int:
    rows = df.to_dict(orient="records")
    sql = text("""
        INSERT INTO results.predicted_reg_prices (model_id, block_id, costs)
        VALUES (:model_id, :block_id, :costs)
        ON CONFLICT (model_id, block_id) DO UPDATE SET
            costs = EXCLUDED.costs
    """)
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH_SIZE):
            conn.execute(sql, rows[i : i + BATCH_SIZE])
            total += len(rows[i : i + BATCH_SIZE])
    log.info("results.predicted_reg_prices: upserted %d row(s)", total)
    return total


def save_results(result, engine: Engine) -> dict[str, int]:
    """Persist all RF result dataframes to the database."""
    log.info("Saving RF results to database...")
    counts = {
        "results.models":               _upsert_models(result.df_model, engine),
        "results.hyperparameters":      _upsert_hyperparameters(result.hyper_df, engine),
        "results.features":             _upsert_features(result.features_df, engine),
        "results.shap_rf_reg_price":    _upsert_shap(result.shap_long_df, engine),
        "results.predicted_reg_prices": _upsert_predicted_prices(result.predicted_prices, engine),
    }
    log.info("DB write complete: %s", counts)
    return counts
