-- =============================================================================
-- 008_optimization_sweep.sql
-- Migration: parallel budget sweep + per-uplift objective weights.
--
-- Files in this directory run automatically only when the PostGIS volume is
-- created from scratch. On an existing database, apply this one by hand:
--
--   docker exec -i urban_platform_db psql -U urban_user -d urban_db \
--       < docker/postgres/init/008_optimization_sweep.sql
--
-- Idempotent — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- results.optimization_summary — record the real objective value and the
-- solver outcome for each budget scenario.
-- ---------------------------------------------------------------------------
ALTER TABLE results.optimization_summary
    ADD COLUMN IF NOT EXISTS objective_function_value DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS solver_status            TEXT,
    ADD COLUMN IF NOT EXISTS solution_status          TEXT;


-- ---------------------------------------------------------------------------
-- results.optimization_weights_setup — weights move from per-model to
-- per-uplift, i.e. per (target_variable, treatment) pair. The old three-column
-- table (optimization_id, model_id, weight) cannot represent that, and its
-- rows refer to a superseded model set, so it is replaced rather than altered.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS results.optimization_weights_setup CASCADE;

CREATE TABLE results.optimization_weights_setup (
    optimization_id TEXT    NOT NULL,
    model_id        TEXT    NOT NULL REFERENCES results.models(model_id) ON DELETE CASCADE,
    uplift          TEXT    NOT NULL,
    uplift_weight   NUMERIC NOT NULL,
    PRIMARY KEY (optimization_id, model_id, uplift)
);

CREATE INDEX IF NOT EXISTS idx_opt_weights_opt ON results.optimization_weights_setup(optimization_id);
