-- =============================================================================
-- 005_ml_tables.sql
-- ML Results Tables — replaces old results.* tables with ML-pipeline-aligned schema
-- Run manually on an existing DB:
--   docker exec -i urban_platform_db psql -U urban_user -d urban_db < docker/postgres/init/005_ml_tables.sql
-- =============================================================================

-- Drop old normalized results tables (replaced by simpler ML-output tables)
DROP TABLE IF EXISTS results.model_setups  CASCADE;
DROP TABLE IF EXISTS results.hypp_spec     CASCADE;
DROP TABLE IF EXISTS results.feature_spec  CASCADE;
DROP TABLE IF EXISTS results.uplifts       CASCADE;


-- ---------------------------------------------------------------------------
-- results.models
-- One row per model run.  run_at allows tracing back to the data snapshot.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.models (
    model_id    TEXT PRIMARY KEY,
    model_type  TEXT          NOT NULL,
    random_seed INTEGER,
    target_id   TEXT          NOT NULL,
    pre_period  INTEGER,
    post_period INTEGER,
    run_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);


-- ---------------------------------------------------------------------------
-- results.hyperparameters
-- Hyperparameter values stored as TEXT to handle mixed types (int/float/str).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.hyperparameters (
    model_id        TEXT    NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    hyper_parameter TEXT    NOT NULL,
    value           TEXT,
    PRIMARY KEY (model_id, hyper_parameter)
);


-- ---------------------------------------------------------------------------
-- results.features
-- Which confounders (X variables) were used, in order.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.features (
    model_id   TEXT    NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    feature_no INTEGER NOT NULL,
    var_id     TEXT    NOT NULL,
    year       INTEGER,
    PRIMARY KEY (model_id, feature_no)
);


-- ---------------------------------------------------------------------------
-- results.uplifts
-- CATE (Conditional Average Treatment Effect) per block per treatment arm.
-- treatment: "d1nq" | "1nq"  (maps to treated_d1nq / treated_1nq columns)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.uplifts (
    block_id  BIGINT  NOT NULL,
    model_id  TEXT    NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    treatment TEXT    NOT NULL,
    uplift    DOUBLE PRECISION,
    PRIMARY KEY (block_id, model_id, treatment)
);

CREATE INDEX IF NOT EXISTS idx_uplifts_model   ON results.uplifts(model_id);
CREATE INDEX IF NOT EXISTS idx_uplifts_block   ON results.uplifts(block_id);
