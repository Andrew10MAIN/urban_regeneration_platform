"""
CLI entry point — multi-model CausalForest + ILP Optimization pipeline.

Usage (inside Docker):
    python src/ml/run_pipeline.py --config configs/ml/default_run.yaml
    python src/ml/run_pipeline.py --config configs/ml/default_run.yaml --no-optimization
    python src/ml/run_pipeline.py --config configs/ml/default_run.yaml --no-db --no-mlflow

Environment variables:
    DATABASE_URL         — SQLAlchemy connection string (required unless --no-db)
    MLFLOW_TRACKING_URI  — MLflow server URI (optional)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path (needed when running as script)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ml_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML + Optimization Pipeline")
    p.add_argument(
        "--config",
        default=str(ROOT / "configs" / "ml" / "default_run.yaml"),
        help="Path to YAML config",
    )
    p.add_argument("--no-db",           action="store_true", help="Skip DB writes")
    p.add_argument("--no-mlflow",       action="store_true", help="Skip MLflow logging")
    p.add_argument("--no-optimization", action="store_true", help="Skip ILP optimization step")
    return p.parse_args()


def _banner(n_models: int) -> None:
    line = "=" * 64
    print(f"\n{line}")
    print(f"  Urban Regeneration — ML + Optimization Pipeline")
    print(f"  Models: {n_models}  |  Run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(line)


def main() -> int:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    from src.ml.config import PipelineConfig
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        return 1

    config = PipelineConfig.from_yaml(config_path)
    _banner(len(config.model_configs))
    log.info(config.summary())

    # ── DB connection ─────────────────────────────────────────────────────────
    engine = None
    if not args.no_db:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            log.error("DATABASE_URL not set. Use --no-db for a dry run.")
            return 1
        from sqlalchemy import create_engine
        engine = create_engine(db_url, pool_pre_ping=True)
        log.info("DB connected: %s", db_url.split("@")[-1])

    mlflow_uri = None if args.no_mlflow else os.environ.get("MLFLOW_TRACKING_URI")

    # ── Steps 1–3: Train each CF model ───────────────────────────────────────
    from src.ml.data_loader import load_data
    from src.ml.model import train
    from src.ml.results_writer import save_results

    t0 = time.time()
    all_ok = True

    for i, model_cfg in enumerate(config.model_configs, 1):
        log.info(
            "=== Model %d/%d: %s  target=%s ===",
            i, len(config.model_configs),
            model_cfg.model_id, model_cfg.target_variable,
        )
        t_model = time.time()

        try:
            data = load_data(model_cfg, engine)
            log.info(
                "  Data: %d blocks | confounders: %s",
                data.n_blocks, data.actual_confounders,
            )

            result = train(model_cfg, data, mlflow_tracking_uri=mlflow_uri)
            log.info("  Training: %.1fs | MLflow=%s", time.time() - t_model, result.mlflow_run_id)

            for arm, s in result.att_stats.items():
                log.info(
                    "  ATT [%s]: %.4f  95%%CI [%.4f, %.4f]  n_treated=%d",
                    arm, s["att"], s["ci_low"], s["ci_high"], s["n_treated"],
                )

            if not args.no_db:
                counts = save_results(result, engine)
                log.info("  DB: %s", counts)

        except Exception as exc:
            log.exception("Model %s failed: %s", model_cfg.model_id, exc)
            all_ok = False

    if not all_ok:
        log.error("One or more models failed — aborting optimization.")
        return 1

    # ── Step 4: ILP Optimization ──────────────────────────────────────────────
    if args.no_optimization:
        log.info("=== Optimization skipped (--no-optimization) ===")
    elif args.no_db:
        log.info("=== Optimization skipped (--no-db flag set) ===")
    else:
        log.info("=== Step 4: ILP Optimization [%s] ===", config.optimization.opt_id)
        t_opt = time.time()
        try:
            from src.ml.optimizer import run as run_optimizer
            from src.ml.results_writer import save_optimization

            opt_result = run_optimizer(config.optimization, engine)
            log.info(
                "  Optimization: %.1fs | direct=%d | spillover=%d",
                time.time() - t_opt,
                (opt_result.uplifts_final["treatment"] == "d1nq").sum(),
                (opt_result.uplifts_final["treatment"] == "1nq").sum(),
            )

            opt_counts = save_optimization(opt_result, engine)
            log.info("  DB: %s", opt_counts)

        except Exception as exc:
            log.exception("Optimization failed: %s", exc)
            return 1

    log.info("=== Pipeline complete in %.1fs ===", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
