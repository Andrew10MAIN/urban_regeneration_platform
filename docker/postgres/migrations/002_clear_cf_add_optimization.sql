-- =============================================================================
-- Migration 002: clear CF model results + add optimization tables
-- Removes all CF model records (cascade: hyperparameters, features, uplifts).
-- RF cost model results are preserved.
-- Run in PowerShell:
--   Get-Content docker\postgres\migrations\002_clear_cf_add_optimization.sql | docker exec -i urban_platform_db psql -U urban_user -d urban_db
-- =============================================================================

-- Remove CF models only (cascade deletes hyperparameters, features, uplifts)
DELETE FROM results.models WHERE model_id LIKE 'CF%';

-- Optimization tables
CREATE TABLE IF NOT EXISTS results.uplifts_optimization (
    optimization_id TEXT   NOT NULL,
    model_id        TEXT   NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    block_id        BIGINT NOT NULL,
    treatment       TEXT   NOT NULL,
    uplift          DOUBLE PRECISION,
    PRIMARY KEY (optimization_id, model_id, block_id, treatment)
);

CREATE INDEX IF NOT EXISTS idx_upl_opt_opt   ON results.uplifts_optimization(optimization_id);
CREATE INDEX IF NOT EXISTS idx_upl_opt_block ON results.uplifts_optimization(block_id);

CREATE TABLE IF NOT EXISTS results.optimization_summary (
    optimization_id      TEXT          PRIMARY KEY,
    cost_used            DOUBLE PRECISION,
    cost_limit           DOUBLE PRECISION,
    treatment_spillovers BOOLEAN,
    clusters_number      INTEGER,
    time_limit           INTEGER,
    run_at               TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
