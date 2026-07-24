"""
CausalForestDML training + MLflow experiment tracking.

Replicates the "Modelling" section of ml_experimenting.ipynb.
"""

from __future__ import annotations

import logging
import warnings
import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

log = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Output of a single training run."""
    uplift_base: pd.DataFrame       # block_id, treated_d1nq, treated_1nq, *_uplift cols
    uplift_long: pd.DataFrame       # block_id, model_id, treatment, uplift
    df_model: pd.DataFrame          # model metadata (1 row)
    hyper_df: pd.DataFrame          # hyperparameters (one row per param)
    features_df: pd.DataFrame       # feature spec (one row per confounder)
    att_stats: dict[str, dict]      # ATT + CI per treatment arm
    mlflow_run_id: str | None = None


def _compute_att(uplift_base: pd.DataFrame) -> dict[str, dict]:
    """
    Compute ATT and 95% CI for each treatment arm.
    Replicates calculate_effect_signifficance() from the notebook.
    """
    stats: dict[str, dict] = {}
    for col in ["treated_d1nq", "treated_1nq"]:
        arm = col.replace("treated_", "")
        uplift_col = f"{col}_uplift"
        series = uplift_base[uplift_base[col] == 1][uplift_col]
        att = series.mean()
        se  = series.std(ddof=1) / np.sqrt(len(series))
        stats[arm] = {
            "att":     float(att),
            "ci_low":  float(att - 1.96 * se),
            "ci_high": float(att + 1.96 * se),
            "n_treated": int((uplift_base[col] == 1).sum()),
        }
    return stats


def train(
    config,                 # ModelConfig
    data,                   # DataBundle
    mlflow_tracking_uri: str | None = None,
) -> ModelResult:
    """
    Train CausalForestDML and return all result dataframes.
    If mlflow_tracking_uri is provided (and reachable), logs the run to MLflow.
    """
    from econml.dml import CausalForestDML

    hp = config.hyperparameters
    n_est        = hp["n_estimators"]
    max_depth    = hp.get("max_depth", None)
    min_leaf     = hp["min_samples_leaf"]
    n_jobs       = hp.get("n_jobs", 1)
    criterion    = hp.get("criterion", "mse")
    min_imp_dec  = hp.get("min_impurity_decrease", 0.0)
    rng          = hp["random_state"]
    global_seed  = config.random_seed

    log.info("Training CausalForestDML  model_id=%s", config.model_id)
    log.info("  n_estimators=%d  max_depth=%s  min_samples_leaf=%d  random_state=%d",
             n_est, max_depth, min_leaf, rng)

    np.random.seed(global_seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        inner_rf_kwargs = dict(
            n_estimators=n_est,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            n_jobs=n_jobs,
        )

        mCF = CausalForestDML(
            model_y=RandomForestRegressor(**inner_rf_kwargs),
            model_t=MultiOutputRegressor(RandomForestRegressor(**inner_rf_kwargs)),
            cv=None,
            criterion=criterion,
            n_estimators=n_est,
            min_samples_leaf=min_leaf,
            min_impurity_decrease=min_imp_dec,
            random_state=rng,
        )
        mCF.tune(data.Y, data.T, X=data.X)
        mCF.fit(data.Y, data.T, X=data.X)

    log.info("Training complete. Computing CATE predictions...")
    pred = mCF.const_marginal_effect(data.X)

    # ── Build uplift_base (wide format) ──────────────────────────────────────
    uplift_base = data.uplift_base.copy()
    pred_df = pd.DataFrame(
        pred.squeeze(), columns=["treated_d1nq_uplift", "treated_1nq_uplift"]
    )
    uplift_base["treated_d1nq_uplift"] = pred_df["treated_d1nq_uplift"].values
    uplift_base["treated_1nq_uplift"]  = pred_df["treated_1nq_uplift"].values

    # ── ATT statistics ────────────────────────────────────────────────────────
    att_stats = _compute_att(uplift_base)
    for arm, s in att_stats.items():
        log.info(
            "ATT [%s]: %.4f  95%%CI [%.4f, %.4f]  n_treated=%d",
            arm, s["att"], s["ci_low"], s["ci_high"], s["n_treated"],
        )

    # ── uplift_long (long format → results.uplifts) ───────────────────────────
    uplift_df = uplift_base.drop(columns=["treated_d1nq", "treated_1nq"]).copy()
    uplift_long = uplift_df.melt(
        id_vars="block_id",
        value_vars=["treated_d1nq_uplift", "treated_1nq_uplift"],
        var_name="treatment",
        value_name="uplift",
    )
    uplift_long["treatment"] = (
        uplift_long["treatment"]
        .str.replace("treated_", "", regex=False)
        .str.replace("_uplift", "", regex=False)
    )
    uplift_long["model_id"] = config.model_id
    uplift_long = uplift_long[["block_id", "model_id", "treatment", "uplift"]].reset_index(drop=True)

    # ── df_model (→ results.models) ──────────────────────────────────────────
    run_at = datetime.datetime.now(tz=datetime.timezone.utc)
    df_model = pd.DataFrame([{
        "model_id":    config.model_id,
        "model_type":  config.model_type,
        "random_seed": config.random_seed,
        "target_id":   config.target_variable,
        "pre_period":  config.pre_period,
        "post_period": config.post_period,
        "run_at":      run_at,
    }])

    # ── hyper_df (→ results.hyperparameters) ─────────────────────────────────
    hyper_df = pd.DataFrame([
        {"model_id": config.model_id, "hyper_parameter": k, "value": str(v)}
        for k, v in hp.items()
    ])

    # ── features_df (→ results.features) ─────────────────────────────────────
    sorted_confounders = sorted(data.actual_confounders)
    features_df = pd.DataFrame([
        {"model_id": config.model_id, "feature_no": i, "var_id": var}
        for i, var in enumerate(sorted_confounders)
    ])

    # ── MLflow logging (optional, best-effort) ────────────────────────────────
    mlflow_run_id = _log_to_mlflow(
        config, data, mCF, att_stats, mlflow_tracking_uri
    )

    return ModelResult(
        uplift_base=uplift_base,
        uplift_long=uplift_long,
        df_model=df_model,
        hyper_df=hyper_df,
        features_df=features_df,
        att_stats=att_stats,
        mlflow_run_id=mlflow_run_id,
    )


def _log_to_mlflow(
    config,
    data,
    trained_model,
    att_stats: dict,
    tracking_uri: str | None,
) -> str | None:
    """Log run to MLflow. Returns run_id on success, None if unavailable."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        log.warning("mlflow not installed — skipping MLflow logging")
        return None

    if not tracking_uri:
        log.info("MLFLOW_TRACKING_URI not set — skipping MLflow logging")
        return None

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("urban_regeneration_causal_forest")

        with mlflow.start_run(run_name=config.model_id) as run:

            # Tags
            mlflow.set_tags({
                "model_id":    config.model_id,
                "target":      config.target_variable,
                "pre_period":  str(config.pre_period),
                "post_period": str(config.post_period),
            })

            # Parameters
            mlflow.log_param("model_id",        config.model_id)
            mlflow.log_param("target",           config.target_variable)
            mlflow.log_param("pre_period",       config.pre_period)
            mlflow.log_param("post_period",      config.post_period)
            mlflow.log_param("confounders_used", ",".join(data.actual_confounders))
            mlflow.log_param("n_blocks",         data.n_blocks)
            mlflow.log_param("random_seed",      config.random_seed)
            for k, v in config.hyperparameters.items():
                mlflow.log_param(f"hp_{k}", v)

            # Metrics
            for arm, s in att_stats.items():
                mlflow.log_metric(f"att_{arm}",      s["att"])
                mlflow.log_metric(f"ci_low_{arm}",   s["ci_low"])
                mlflow.log_metric(f"ci_high_{arm}",  s["ci_high"])
                mlflow.log_metric(f"n_treated_{arm}", s["n_treated"])

            # Model artifact (scikit-learn compatible wrapper)
            try:
                mlflow.sklearn.log_model(trained_model, artifact_path="model")
            except Exception as exc:
                log.warning("Could not log model artifact to MLflow: %s", exc)

        run_id = run.info.run_id
        log.info("MLflow run logged: %s  (run_id=%s)", config.model_id, run_id)
        return run_id

    except Exception as exc:
        log.warning("MLflow logging failed (non-critical): %s", exc)
        return None
