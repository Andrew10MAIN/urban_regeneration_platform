-- =============================================================================
-- 006_rf_tables.sql
-- RF Regeneration Cost Model — new result tables
-- Shared tables (results.models / .hyperparameters / .features) already
-- defined in 005_ml_tables.sql and require no changes here.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- results.shap_rf_reg_price
-- SHAP values per block per feature for the RF cost model.
-- var_id = "base_value" stores the model's expected_value intercept.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.shap_rf_reg_price (
    model_id TEXT          NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    block_id BIGINT        NOT NULL,
    var_id   TEXT          NOT NULL,
    value    DOUBLE PRECISION,
    PRIMARY KEY (model_id, block_id, var_id)
);

CREATE INDEX IF NOT EXISTS idx_shap_rf_model   ON results.shap_rf_reg_price(model_id);
CREATE INDEX IF NOT EXISTS idx_shap_rf_block   ON results.shap_rf_reg_price(block_id);


-- ---------------------------------------------------------------------------
-- results.predicted_reg_prices
-- RF-predicted regeneration cost for blocks with costs == 0 (untreated).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.predicted_reg_prices (
    model_id TEXT          NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    block_id BIGINT        NOT NULL,
    costs    DOUBLE PRECISION,
    PRIMARY KEY (model_id, block_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_prices_model ON results.predicted_reg_prices(model_id);
