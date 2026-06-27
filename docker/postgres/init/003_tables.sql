-- =========================
-- CORE URBAN BLOCKS
-- =========================

CREATE TABLE core.urban_blocks (
    block_id BIGINT,
    year TIMESTAMP,

    treated_all BIGINT,
    treated_d1nq BIGINT,
    treated_1nq BIGINT,

    PRIMARY KEY (block_id, year)
);


-- =========================
-- GEOMETRY BLOCKS
-- =========================

CREATE TABLE core.urban_blocks_geom (
    block_id BIGINT PRIMARY KEY,
    area DOUBLE PRECISION,
    geometry geometry(Geometry, 2177)
);


-- =========================
-- REGENERATION ACTIONS
-- =========================

CREATE TABLE regeneration.actions (
    regen_id BIGINT PRIMARY KEY,
    regen_type TEXT,
    regen_star TIMESTAMP,
    regen_end TIMESTAMP,
    price_pln TEXT,
    entity TEXT,
    block_id BIGINT,
    geometry geometry(Geometry, 2177)
);


-- =========================
-- LEGACY VARIABLES
-- =========================

CREATE TABLE legacy.variables (
    block_id BIGINT,
    year TIMESTAMP,
    var_id TEXT,
    value DOUBLE PRECISION,

    PRIMARY KEY (block_id, year, var_id)
);


-- =========================
-- META VARIABLES
-- =========================

CREATE TABLE meta.var_description (
    var_id TEXT PRIMARY KEY,
    origin TEXT,
    description TEXT,
    unit TEXT
);