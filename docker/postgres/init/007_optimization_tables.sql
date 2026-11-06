-- =============================================================================
-- 007_optimization_tables.sql
-- ILP Optimization result tables
--
-- One optimization run = one budget scenario, identified by optimization_id.
-- A sweep over a budget range therefore writes many rows here (one per
-- scenario), all sharing the same model set and objective-function weights.
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
-- One row per optimization run — budget, solver outcome, run metadata.
--
-- objective_function_value is the real value of the ILP objective function
-- reported by the solver for this scenario. With a finite time_limit the
-- solver may return a feasible-but-not-proven-optimal solution, so
-- solver_status / solution_status must be read alongside it (a plateau or a
-- dip in the objective as the budget grows means the time limit bit, not that
-- a bigger budget is worse).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.optimization_summary (
    optimization_id          TEXT          PRIMARY KEY,
    cost_used                DOUBLE PRECISION,
    cost_limit               DOUBLE PRECISION,
    objective_function_value DOUBLE PRECISION,
    solver_status            TEXT,
    solution_status          TEXT,
    treatment_spillovers     BOOLEAN,
    clusters_number          INTEGER,
    time_limit               INTEGER,
    run_at                   TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);


-- ---------------------------------------------------------------------------
-- results.optimization_weights_setup
-- Objective-function weights actually used, in long format: one row per
-- (optimization_id, model_id, uplift).
--
-- uplift is the uplift column name, "<target_variable>_<treatment>", e.g.
-- urVibSCBx_coun_00000000_d1nq (direct) or urVibSCBx_coun_00000000_1nq
-- (spatial spillover). uplift_weight is the weight after normalization, so
-- the weights of one optimization_id sum to 1 when normalize_weights is on.
-- Only uplifts that passed the significance test are stored.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results.optimization_weights_setup (
    optimization_id TEXT    NOT NULL,
    model_id        TEXT    NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    uplift          TEXT    NOT NULL,
    uplift_weight   NUMERIC NOT NULL,
    PRIMARY KEY (optimization_id, model_id, uplift)
);

CREATE INDEX IF NOT EXISTS idx_opt_weights_opt ON results.optimization_weights_setup(optimization_id);
