-- =============================================================================
-- 007_optimization_tables.sql
-- ILP Optimization result tables
-- =============================================================================

-- ---------------------------------------------------------------------------
-- results.uplifts_optimization
-- Selected uplifts from the ILP optimizer (direct + spillover blocks).
-- ---------------------------------------------------------------------------
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


-- ---------------------------------------------------------------------------
-- results.optimization_summary
-- One row per optimization run — budget, parameters, run metadata.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.optimization_summary (
    optimization_id      TEXT          PRIMARY KEY,
    cost_used            DOUBLE PRECISION,
    cost_limit           DOUBLE PRECISION,
    treatment_spillovers BOOLEAN,
    clusters_number      INTEGER,
    time_limit           INTEGER,
    run_at               TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);


-- ---------------------------------------------------------------------------
-- results.optimization_weights_setup
-- Per-model weights used in the ILP objective function.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.optimization_weights_setup (
    optimization_id TEXT    NOT NULL,
    model_id        TEXT    NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    weight          NUMERIC NOT NULL,
    PRIMARY KEY (optimization_id, model_id)
);
