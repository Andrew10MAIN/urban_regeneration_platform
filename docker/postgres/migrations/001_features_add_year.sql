-- =============================================================================
-- Migration 001: add year column to results.features
-- Clears all results data and adds year column tracking which year's data
-- was used for each confounder (may differ from pre_period due to fallback).
--
-- Run in PowerShell:
--   Get-Content docker/postgres/migrations/001_features_add_year.sql | docker exec -i urban_platform_db psql -U urban_user -d urban_db
-- =============================================================================

-- Clear all results (CASCADE handles FK-dependent tables: hyperparameters, features, uplifts)
TRUNCATE results.models CASCADE;

-- Add year column (idempotent)
ALTER TABLE results.features
    ADD COLUMN IF NOT EXISTS year INTEGER;
