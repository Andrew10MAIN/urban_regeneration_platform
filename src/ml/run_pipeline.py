#!/usr/bin/env python3
"""
Urban Regeneration Platform — ML Pipeline Entry Point
======================================================
Trains a CausalForestDML model, logs to MLflow and writes results to DB.

Usage (from project root):
  python src/ml/run_pipeline.py
  python src/ml/run_pipeline.py --config configs/ml/default_run.yaml
  python src/ml/run_pipeline.py --config configs/ml/default_run.yaml --model-id CF00000000002

Environment variables:
  DATABASE_URL          postgresql+psycopg2://...   (overrides config default)
  MLFLOW_TRACKING_URI   http://localhost:5000        (set to enable MLflow)

Exit codes:
  0   success
  1   error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import datetime
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine

from src.ml.config import ModelConfig
from src.ml.data_loader import load_data
from src.ml.model import train
from src.ml.results_writer import save_results


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ml_pipeline")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Urban Regeneration Platform — CausalForest ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "ml" / "default_run.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--model-id",    default=None, help="Override model_id")
    parser.add_argument("--target",      default=None, help="Override target_variable")
    parser.add_argument("--pre-period",  default=None, type=int, help="Override pre_period")
    parser.add_argument("--post-period", default=None, type=int, help="Override post_period")
    parser.add_argument("--no-db",       action="store_true", help="Skip DB write (dry run)")
    parser.add_argument("--no-mlflow",   action="store_true", help="Skip MLflow logging")
    return parser.parse_args()


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    line = "=" * 64
    print(f"\n{line}")
    print(f"  Urban Regeneration Platform — CausalForest ML Pipeline")
    print(f"  Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(line)


def print_summary(result, counts: dict, elapsed: float) -> None:
    line = "=" * 64
    print(f"\n{line}")
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"  model_id : {result.df_model['model_id'].iloc[0]}")
    print(f"  run_at   : {result.df_model['run_at'].iloc[0]}")
    print(f"\n  ATT Results:")
    for arm, s in result.att_stats.items():
        print(
            f"    [{arm}]  ATT={s['att']:.4f}  "
            f"95%CI [{s['ci_low']:.4f}, {s['ci_high']:.4f}]  "
            f"n_treated={s['n_treated']}"
        )
    if result.mlflow_run_id:
        print(f"\n  MLflow run_id: {result.mlflow_run_id}")
    if counts:
        print(f"\n  DB writes:")
        for tbl, n in counts.items():
            print(f"    {tbl}: {n} rows")
    print(f"{line}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    print_banner()

    # ── Load config ──────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        return 1

    config = ModelConfig.from_yaml(config_path)

    # Apply CLI overrides
    overrides: dict = {}
    if args.model_id:    overrides["model_id"]         = args.model_id
    if args.target:      overrides["target_variable"]   = args.target
    if args.pre_period:  overrides["pre_period"]        = args.pre_period
    if args.post_period: overrides["post_period"]       = args.post_period
    if overrides:
        config = config.override(**overrides)

    log.info("Configuration loaded:\n%s", config.summary())

    # ── DB engine ─────────────────────────────────────────────────────────────
    db_url = os.environ.get("DATABASE_URL") or \
             "postgresql+psycopg2://urban_user:urban_password@localhost:5433/urban_db"
    engine = create_engine(db_url, pool_pre_ping=True)

    # ── MLflow URI ────────────────────────────────────────────────────────────
    mlflow_uri = None if args.no_mlflow else os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        log.info("MLflow tracking URI: %s", mlflow_uri)
    else:
        log.info("MLflow disabled (set MLFLOW_TRACKING_URI to enable)")

    # ── Load data ─────────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("Step 1/3: Loading data...")
    try:
        data = load_data(config, engine)
    except Exception as exc:
        log.exception("Data loading failed: %s", exc)
        return 1

    # ── Train model ───────────────────────────────────────────────────────────
    log.info("Step 2/3: Training CausalForestDML...")
    try:
        result = train(config, data, mlflow_tracking_uri=mlflow_uri)
    except Exception as exc:
        log.exception("Model training failed: %s", exc)
        return 1

    # ── Write results ─────────────────────────────────────────────────────────
    counts: dict = {}
    if not args.no_db:
        log.info("Step 3/3: Writing results to DB...")
        try:
            counts = save_results(result, engine)
        except Exception as exc:
            log.exception("DB write failed: %s", exc)
            return 1
    else:
        log.info("Step 3/3: --no-db flag set, skipping DB write (dry run)")

    elapsed = time.time() - t0
    print_summary(result, counts, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
