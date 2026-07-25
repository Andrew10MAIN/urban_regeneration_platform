"""
CLI entry point for the RF Regeneration Cost pipeline.

Usage (inside Docker):
    python src/rf/run_pipeline.py --config configs/rf/default_run.yaml
    python src/rf/run_pipeline.py --config configs/rf/default_run.yaml --model-id RF00000000002
    python src/rf/run_pipeline.py --config configs/rf/default_run.yaml --no-db --no-mlflow

Environment variables:
    DATABASE_URL          — SQLAlchemy connection string (required unless --no-db)
    MLFLOW_TRACKING_URI   — MLflow server URI (optional)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rf_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF Regeneration Cost Pipeline")
    p.add_argument("--config",      required=True, help="Path to YAML config file")
    p.add_argument("--model-id",    default=None,  help="Override model_id from config")
    p.add_argument("--pre-period",  type=int, default=None, help="Override pre_period")
    p.add_argument("--post-period", type=int, default=None, help="Override post_period")
    p.add_argument("--no-db",       action="store_true", help="Skip DB writes")
    p.add_argument("--no-mlflow",   action="store_true", help="Skip MLflow logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── Config ────────────────────────────────────────────────────────────────
    from src.rf.config import RFConfig
    config = RFConfig.from_yaml(args.config)
    config = config.override(
        model_id=args.model_id,
        pre_period=args.pre_period,
        post_period=args.post_period,
    )
    log.info("=== RF Regeneration Cost Pipeline ===")
    log.info(config.summary())

    # ── Database connection ───────────────────────────────────────────────────
    db_url = os.environ.get("DATABASE_URL")
    if not db_url and not args.no_db:
        log.error("DATABASE_URL not set. Use --no-db to skip DB writes.")
        sys.exit(1)

    engine = None
    if not args.no_db:
        from sqlalchemy import create_engine
        engine = create_engine(db_url, pool_pre_ping=True)
        log.info("Connected to database: %s", db_url.split("@")[-1])

    # ── MLflow URI ────────────────────────────────────────────────────────────
    mlflow_uri = None if args.no_mlflow else os.environ.get("MLFLOW_TRACKING_URI")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    log.info("--- Step 1: Loading data ---")
    from src.rf.data_loader import load_data
    data = load_data(config, engine)
    log.info(
        "Data ready — train: %d blocks  predict: %d blocks  features: %d",
        data.n_train, data.n_predict, len(data.feature_names),
    )

    # ── Step 2: Train model ───────────────────────────────────────────────────
    log.info("--- Step 2: Training model ---")
    from src.rf.model import train
    result = train(config, data, mlflow_tracking_uri=mlflow_uri)
    log.info("Training complete — R²=%.4f  MLflow run_id=%s", result.train_r2, result.mlflow_run_id)

    # ── Step 3: Save results ──────────────────────────────────────────────────
    if args.no_db:
        log.info("--- Step 3: DB write skipped (--no-db) ---")
    else:
        log.info("--- Step 3: Saving results to database ---")
        from src.rf.results_writer import save_results
        counts = save_results(result, engine)
        log.info("Rows written: %s", counts)

    elapsed = time.time() - t0
    log.info("=== Pipeline complete in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
