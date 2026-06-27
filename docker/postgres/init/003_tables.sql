-- =========================
-- CORE
-- =========================

CREATE TABLE IF NOT EXISTS core.urban_blocks (
    block_id     BIGINT,
    year         TIMESTAMP,
    treated_all  BOOLEAN,
    treated_d1nq BOOLEAN,
    treated_1nq  BOOLEAN,
    PRIMARY KEY (block_id, year)
);

CREATE TABLE IF NOT EXISTS core.urban_blocks_geom (
    block_id BIGINT PRIMARY KEY,
    area     DOUBLE PRECISION,
    centroid GEOMETRY(Point, 2177),
    geometry GEOMETRY(MultiPolygon, 2177)
);


-- =========================
-- LEGACY
-- =========================

CREATE TABLE IF NOT EXISTS legacy.variables (
    var_id   TEXT,
    year     TIMESTAMP,
    block_id BIGINT,
    value    DOUBLE PRECISION,
    PRIMARY KEY (var_id, year, block_id)
);


-- =========================
-- META
-- =========================

CREATE TABLE IF NOT EXISTS meta.var_description (
    var_id      TEXT PRIMARY KEY,
    unit        TEXT,
    origin      TEXT,
    description TEXT
);


-- =========================
-- REGENERATION
-- =========================

CREATE TABLE IF NOT EXISTS regeneration.actions (
    reg_id      BIGINT PRIMARY KEY,
    block_id    BIGINT,
    type        TEXT,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    geometry    GEOMETRY(Point, 2177),
    costs       DOUBLE PRECISION
);


-- =========================
-- MINED
-- =========================

CREATE TABLE IF NOT EXISTS mined.variables (
    var_id   TEXT,
    year     TIMESTAMP,
    block_id BIGINT,
    value    DOUBLE PRECISION,
    PRIMARY KEY (var_id, year, block_id)
);

CREATE TABLE IF NOT EXISTS mined."Build_perm" (
    build_perm_id TEXT,
    build_plot_no TEXT,
    block_id      BIGINT,
    date          TIMESTAMP,
    description   TEXT,
    geometry      GEOMETRY(Point, 2177),
    PRIMARY KEY (build_perm_id, build_plot_no)
);

CREATE TABLE IF NOT EXISTS mined.adresses (
    adress   TEXT PRIMARY KEY,
    block_id BIGINT
);

CREATE TABLE IF NOT EXISTS mined.penalties (
    pen_id BIGINT PRIMARY KEY,
    adress TEXT,
    date   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mined.app_prices (
    build_id TEXT,
    geometry GEOMETRY(Point, 2177),
    price    DOUBLE PRECISION,
    area     DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS mined.buildings (
    build_id    TEXT PRIMARY KEY,
    valid_from  TIMESTAMP,
    valid_until TIMESTAMP,
    area        DOUBLE PRECISION,
    height      INTEGER,
    geometry    GEOMETRY(MultiPolygon, 2177),
    centroid    GEOMETRY(Point, 2177)
);


-- =========================
-- RESULTS
-- =========================

CREATE TABLE IF NOT EXISTS results.uplifts (
    block_id BIGINT,
    var_id   TEXT,
    uplift   DOUBLE PRECISION,
    PRIMARY KEY (var_id, block_id)
);

CREATE TABLE IF NOT EXISTS results.optimization (
    block_id          BIGINT PRIMARY KEY,
    designated_to_reg BOOLEAN
);

CREATE TABLE IF NOT EXISTS results.predicted_reg_price (
    block_id   BIGINT PRIMARY KEY,
    pred_price DOUBLE PRECISION
);


-- =========================
-- OSM
-- =========================

CREATE TABLE IF NOT EXISTS osm.raw_poi (
    osm_id      BIGINT,
    type        TEXT,
    valid_from  TIMESTAMP,
    valid_until TIMESTAMP,
    geometry    GEOMETRY(Point, 4326)
);

CREATE TABLE IF NOT EXISTS osm.poi (
    osm_id   BIGINT,
    type     TEXT,
    year     TIMESTAMP,
    geometry GEOMETRY(Point, 4326)
);

CREATE TABLE IF NOT EXISTS osm.raw_poly (
    osm_id      BIGINT,
    type        TEXT,
    valid_from  TIMESTAMP,
    valid_until TIMESTAMP,
    geometry    GEOMETRY(Polygon, 4326)
);

CREATE TABLE IF NOT EXISTS osm.poly (
    osm_id   BIGINT,
    type     TEXT,
    year     TIMESTAMP,
    geometry GEOMETRY(Polygon, 4326),
    area     DOUBLE PRECISION,
    centroid GEOMETRY(Point, 4326)
);


-- =========================
-- AUDIT / STAGING
-- =========================

CREATE TABLE IF NOT EXISTS audit.etl_log (
    run_id         SERIAL PRIMARY KEY,
    dag_id         TEXT,
    source         TEXT,
    started_at     TIMESTAMP DEFAULT NOW(),
    finished_at    TIMESTAMP,
    rows_extracted INTEGER,
    rows_staged    INTEGER,
    rows_new       INTEGER,
    rows_loaded    INTEGER,
    status         TEXT DEFAULT 'running',
    error_msg      TEXT
);

CREATE TABLE IF NOT EXISTS audit.stg_build_perm (
    id            SERIAL,
    run_id        INTEGER REFERENCES audit.etl_log(run_id),
    build_perm_id TEXT,
    build_plot_no TEXT,
    issue_date    DATE,
    description   TEXT,
    geometry      GEOMETRY(Point, 2177),
    loaded_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stg_build_perm_run ON audit.stg_build_perm(run_id);
