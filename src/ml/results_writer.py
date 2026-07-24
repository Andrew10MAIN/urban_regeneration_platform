"""
Persist ML results to the results.* tables in urban_db.

Tables written:
  results.models           — df_model     (ON CONFLICT DO UPDATE)
  results.hyperparameters  — hyper_df     (ON CONFLICT DO UPDATE)
  results.features         — features_df  (ON CONFLICT DO UPDATE)
  results.uplifts          — uplift_long  (ON CONFLICT DO UPDATE)
"""

from __future__ import annotations

import logging
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

log = logging.getLogger(__name__)

# ── Upsert helpers ─────────────────────────────────────────────────────────────

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
        INSERT INTO results.features (model_id, feature_no, var_id)
        VALUES (:model_id, :feature_no, :var_id)
        ON CONFLICT (model_id, feature_no) DO UPDATE SET
            var_id = EXCLUDED.var_id
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
    # batch inserts for large tables
    batch_size = 500
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), batch_size):
            conn.execute(sql, rows[i : i + batch_size])
            total += len(rows[i : i + batch_size])
    log.info("results.uplifts: upserted %d row(s)", total)
    return total


# ── Public entry ───────────────────────────────────────────────────────────────

def save_results(result, engine: Engine) -> dict[str, int]:
    """
    Persist all result dataframes to the database.
    Returns dict of row counts per table.
    """
    log.info("Saving ML results to database...")

    counts = {
        "results.models":          _upsert_models(result.df_model, engine),
        "results.hyperparameters": _upsert_hyperparameters(result.hyper_df, engine),
        "results.features":        _upsert_features(result.features_df, engine),
        "results.uplifts":         _upsert_uplifts(result.uplift_long, engine),
    }

    log.info("DB write complete: %s", counts)
    return counts
