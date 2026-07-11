-- =========================
-- CORE
-- =========================

CREATE TABLE IF NOT EXISTS core.urban_blocks (
    block_id     BIGINT,
    year         TIMESTAMP,
    treated_all  INTEGER,
    treated_d1nq INTEGER,
    treated_1nq  INTEGER,
    PRIMARY KEY (block_id, year)
);

CREATE TABLE IF NOT EXISTS core.urban_blocks_geom (
    block_id BIGINT PRIMARY KEY,
    area     DOUBLE PRECISION,

    geometry GEOMETRY(MultiPolygon, 2177)
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
    gml_id      TEXT PRIMARY KEY,
    guid        TEXT,
    full_adress TEXT,
    street      TEXT,
    building_no TEXT,
    zip_code    TEXT,
    status      TEXT,
    geometry    GEOMETRY(Point, 2177)
);

CREATE TABLE IF NOT EXISTS mined.penalties (
    pen_id           BIGINT PRIMARY KEY,
    date             TIMESTAMP,
    place_of_penalty TEXT,
    pen_type         TEXT,
    geometry         GEOMETRY(Point, 2177)
);

CREATE TABLE IF NOT EXISTS mined.buildings (
    building_id          TEXT PRIMARY KEY,
    floors_above_ground  INTEGER,
    floors_below_ground  INTEGER,
    geometry             GEOMETRY(Geometry, 2177)
);

CREATE TABLE IF NOT EXISTS mined.app_prices (
    gml_id      TEXT PRIMARY KEY,
    res_unit_id TEXT,
    building_id TEXT,
    date        TIMESTAMP,
    floor_no    DOUBLE PRECISION,
    floor_area  DOUBLE PRECISION,
    price_gross DOUBLE PRECISION,
    block_id    BIGINT,
    geometry    GEOMETRY(Point, 2177)
);


-- =========================
-- RESULTS
-- =========================

CREATE TABLE IF NOT EXISTS results.uplifts (
    block_id BIGINT,
    target_id   TEXT,
    model_id TEXT,
    uplift   DOUBLE PRECISION,
    PRIMARY KEY (target_id, block_id, model_id)
);

CREATE TABLE IF NOT EXISTS results.optimization (
    block_id          BIGINT,
    optimization_id   BIGINT,
    designated_to_reg BOOLEAN,
    PRIMARY KEY (block_id, optimization_id)
);

CREATE TABLE IF NOT EXISTS results.predicted_reg_price (
    block_id   BIGINT,
    model_id TEXT,    
    pred_price DOUBLE PRECISION,
    PRIMARY KEY (block_id, model_id)
);

CREATE TABLE IF NOT EXISTS results.feature_spec (
    features_id   TEXT,
    features_no INTEGER,
    feature_id TEXT,
    PRIMARY KEY (features_id, features_no)
);

CREATE TABLE IF NOT EXISTS results.hypp_spec (
    hyppar_id   TEXT,
    model_id TEXT,    
    hyppar TEXT,
    value DOUBLE PRECISION,
    PRIMARY KEY (hyppar_id, model_id)
);


CREATE TABLE IF NOT EXISTS results.model_setups (
    model_id TEXT,  
    model_type TEXT,    
    target_id TEXT,
    hyppar_id   TEXT,
    features_id   TEXT,
    PRIMARY KEY (hyppar_id, model_type)
);
-- =========================
-- OSM
-- =========================



CREATE TABLE IF NOT EXISTS osm.poi (
    osm_id   BIGINT,
    fclass   TEXT,
    date     TIMESTAMP,
    geometry GEOMETRY(Point, 2177),
    UNIQUE (osm_id, date)
);



CREATE TABLE IF NOT EXISTS osm.poly (
    osm_id   BIGINT,
    fclass   TEXT,
    date     TIMESTAMP,
    geometry GEOMETRY(Geometry, 2177),
    UNIQUE (osm_id, date)
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
    block_id      BIGINT,
    geometry      GEOMETRY(Point, 2177),
    loaded_at     TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stg_build_perm_run ON audit.stg_build_perm(run_id);

CREATE TABLE IF NOT EXISTS audit.stg_addresses (
    id          SERIAL,
    run_id      INTEGER REFERENCES audit.etl_log(run_id),
    gml_id      TEXT,
    guid        TEXT,
    full_adress TEXT,
    street      TEXT,
    building_no TEXT,
    zip_code    TEXT,
    status      TEXT,
    geometry    GEOMETRY(Point, 2177),
    loaded_at   TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stg_addresses_run ON audit.stg_addresses(run_id);

CREATE TABLE IF NOT EXISTS audit.stg_penalties (
    id               SERIAL,
    run_id           INTEGER REFERENCES audit.etl_log(run_id),
    source_file      TEXT,
    pen_id           BIGINT,
    date             TIMESTAMP,
    place_of_penalty TEXT,
    pen_type         TEXT,
    geometry         GEOMETRY(Point, 2177),
    loaded_at        TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stg_penalties_run ON audit.stg_penalties(run_id);

CREATE TABLE IF NOT EXISTS audit.processed_files (
    id           SERIAL PRIMARY KEY,
    dag_id       TEXT NOT NULL,
    file_name    TEXT NOT NULL,
    file_hash    TEXT,
    processed_at TIMESTAMP DEFAULT NOW(),
    run_id       INTEGER,
    UNIQUE (dag_id, file_name)
);

CREATE TABLE IF NOT EXISTS audit.stg_buildings (
    id                   SERIAL,
    run_id               INTEGER REFERENCES audit.etl_log(run_id),
    building_id          TEXT,
    floors_above_ground  INTEGER,
    floors_below_ground  INTEGER,
    geometry             GEOMETRY(Geometry, 2177),
    loaded_at            TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stg_buildings_run ON audit.stg_buildings(run_id);

CREATE TABLE IF NOT EXISTS audit.stg_building_vars (
    id          SERIAL PRIMARY KEY,
    run_id      INTEGER REFERENCES audit.etl_log(run_id),
    var_id      TEXT,
    year        TIMESTAMP,
    block_id    BIGINT,
    value       DOUBLE PRECISION,
    computed_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stg_building_vars_run  ON audit.stg_building_vars(run_id);
CREATE INDEX IF NOT EXISTS idx_stg_building_vars_year ON audit.stg_building_vars(var_id, year);

CREATE TABLE IF NOT EXISTS audit.stg_app_prices (
    id          SERIAL,
    run_id      INTEGER REFERENCES audit.etl_log(run_id),
    gml_id      TEXT,
    res_unit_id TEXT,
    building_id TEXT,
    date        TIMESTAMP,
    floor_no    DOUBLE PRECISION,
    floor_area  DOUBLE PRECISION,
    price_gross DOUBLE PRECISION,
    block_id    BIGINT,
    geometry    GEOMETRY(Point, 2177),
    loaded_at   TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stg_app_prices_run ON audit.stg_app_prices(run_id);
