-- =============================================================================
-- 009_proximity_criterion.sql
-- Migration: record which adjacency rule the connectivity constraint used.
--
-- Files in this directory run automatically only when the PostGIS volume is
-- created from scratch. On an existing database, apply this one by hand:
--
--   docker exec -i urban_platform_db psql -U urban_user -d urban_db \
--       < docker/postgres/init/009_proximity_criterion.sql
--
-- Idempotent — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- results.optimization_summary.proximity_criterion
--
-- 'queen' — blocks count as neighbours when their boundaries touch at all,
--           a single shared corner point included
-- 'rook'  — blocks count as neighbours only when they share a boundary segment
--           of positive length, i.e. a real common edge
--
-- The choice changes the meaning of the single-cluster constraint: under queen
-- a recommended area may be joined only at a corner, which is not contiguous
-- on the ground.
-- ---------------------------------------------------------------------------
ALTER TABLE results.optimization_summary
    ADD COLUMN IF NOT EXISTS proximity_criterion TEXT;

-- Every optimisation written before this column existed was solved with queen
-- adjacency (geopandas sjoin, predicate='touches').
UPDATE results.optimization_summary
   SET proximity_criterion = 'queen'
 WHERE proximity_criterion IS NULL;
