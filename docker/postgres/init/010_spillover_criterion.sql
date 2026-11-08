-- =============================================================================
-- 010_spillover_criterion.sql
-- Migration: record the adjacency rule used for the spillover reach, which is
-- not the same as the one used for contiguity.
--
-- Files in this directory run automatically only when the PostGIS volume is
-- created from scratch. On an existing database, apply this one by hand:
--
--   docker exec -i urban_platform_db psql -U urban_user -d urban_db \
--       < docker/postgres/init/010_spillover_criterion.sql
--
-- Idempotent — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- results.optimization_summary.spillover_criterion
--
-- proximity_criterion governs which blocks may be joined into the directly
-- treated cluster: with 'rook' they must share an edge, because two blocks
-- meeting at a corner are not one contiguous area on the ground.
--
-- spillover_criterion governs how far the indirect effect reaches around that
-- cluster: with 'queen' a block touching a treated block at a corner still
-- picks up the effect, since an effect is not bound by the contiguity rule.
-- ---------------------------------------------------------------------------
ALTER TABLE results.optimization_summary
    ADD COLUMN IF NOT EXISTS spillover_criterion TEXT;

-- Every optimisation written before this column existed used one neighbourhood
-- for both roles, and that neighbourhood was queen.
UPDATE results.optimization_summary
   SET spillover_criterion = 'queen'
 WHERE spillover_criterion IS NULL;
