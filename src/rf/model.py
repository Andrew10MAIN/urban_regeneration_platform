"""
RandomForestRegressor training + SHAP computation + MLflow tracking.

Replicates the "Model train" and "Model predict" sections of
rf_experimenting.ipynb.
"""

from __future__ import annotations

import datetime
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RFModelResult:
    """Output of a single RF training run."""
    shap_long_df: pd.DataFrame          # model_id, block_id, var_id, value
    predicted_prices: pd.DataFrame      # model_id, block_id, costs
    df_model: pd.DataFrame              # results.models (1 row)
    hyper_df: pd.DataFrame              # results.hyperparameters
    features_df: pd.DataFrame           # results.features  (model_id, feature_no, var_id, year)
    train_r2: float | None
    mlflow_run_id: str | None = None


def train(
    config,                             # RFConfig
    data,                               # RFDataBundle
    mlflow_tracking_uri: str | None = None,
) -> RFModelResult:
    """
    Train RandomForestRegressor, compute SHAP values, build all result
    dataframes and (optionally) log the run to MLflow.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    hp = config.hyperparameters

    log.info("Training RandomForestRegressor  model_id=%s", config.model_id)
    log.info(
        "  n_estimators=%d  random_state=%d  n_jobs=%d",
        hp["n_est"], hp["random_state"], hp.get("n_jobs", 1),
    )

    np.random.seed(config.random_seed)

    rf = RandomForestRegressor(
        n_estimators=hp["n_est"],
        random_state=hp["random_state"],
        n_jobs=hp.get("n_jobs", 1),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(data.X_train, data.y_train)

    y_pred_train = rf.predict(data.X_train)
    train_r2 = float(r2_score(data.y_train, y_pred_train))
    log.info("Train R²: %.4f", train_r2)

    # ── SHAP values (on training set) ─────────────────────────────────────────
    log.info("Computing SHAP values (TreeExplainer on %d training blocks)...", data.n_train)
    import shap
    explainer = shap.TreeExplainer(rf)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(data.X_train)

    shap_df = pd.DataFrame(
        shap_values,
        columns=data.X_train.columns,
        index=data.X_train.index,
    ).reset_index()

    shap_df["model_id"] = config.model_id
    id_cols = ["model_id", "block_id"]
    value_cols = [c for c in shap_df.columns if c not in id_cols]

    shap_long = shap_df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="var_id",
        value_name="value",
    )

    # append base_value row per block
    base_value = float(
        explainer.expected_value[0]
        if hasattr(explainer.expected_value, "__len__")
        else explainer.expected_value
    )
    base_rows = (
        shap_long[["model_id", "block_id"]]
        .drop_duplicates(["model_id", "block_id"])
        .copy()
    )
    base_rows["var_id"] = "base_value"
    base_rows["value"] = base_value

    shap_long_df = (
        pd.concat([shap_long, base_rows], ignore_index=True)
        .sort_values(["block_id", "var_id"])
        .reset_index(drop=True)
    )
    log.info("SHAP long df: %d rows", len(shap_long_df))

    # ── Predictions for untreated blocks ─────────────────────────────────────
    log.info("Predicting costs for %d untreated blocks...", data.n_predict)
    y_pred = rf.predict(data.X_pred)

    predicted_prices = pd.DataFrame({
        "model_id": config.model_id,
        "block_id": data.X_pred.index,
        "costs":    y_pred,
    }).reset_index(drop=True)
    log.info(
        "Predicted costs — min: %.0f  max: %.0f  mean: %.0f",
        y_pred.min(), y_pred.max(), y_pred.mean(),
    )

    # ── df_model (→ results.models) ──────────────────────────────────────────
    run_at = datetime.datetime.now(tz=datetime.timezone.utc)
    df_model = pd.DataFrame([{
        "model_id":    config.model_id,
        "model_type":  config.model_type,
        "random_seed": config.random_seed,
        "target_id":   "costs",
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
    features_df = pd.DataFrame([
        {
            "model_id":   config.model_id,
            "feature_no": i,
            "var_id":     fn,
            "year":       data.feature_years.get(fn, config.pre_period),
        }
        for i, fn in enumerate(data.feature_names)
    ])

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow_run_id = _log_to_mlflow(
        config, data, rf, train_r2, mlflow_tracking_uri
    )

    return RFModelResult(
        shap_long_df=shap_long_df,
        predicted_prices=predicted_prices,
        df_model=df_model,
        hyper_df=hyper_df,
        features_df=features_df,
        train_r2=train_r2,
        mlflow_run_id=mlflow_run_id,
    )


def _log_to_mlflow(
    config,
    data,
    trained_model,
    train_r2: float,
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
        mlflow.set_experiment("urban_regeneration_rf_cost")

        with mlflow.start_run(run_name=config.model_id) as run:

            mlflow.set_tags({
                "model_id":    config.model_id,
                "model_type":  config.model_type,
                "pre_period":  str(config.pre_period),
                "post_period": str(config.post_period),
            })

            mlflow.log_param("model_id",    config.model_id)
            mlflow.log_param("pre_period",  config.pre_period)
            mlflow.log_param("post_period", config.post_period)
            mlflow.log_param("random_seed", config.random_seed)
            mlflow.log_param("n_train",     data.n_train)
            mlflow.log_param("n_predict",   data.n_predict)
            for k, v in config.hyperparameters.items():
                mlflow.log_param(f"hp_{k}", v)

            mlflow.log_metric("train_r2", train_r2)

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
