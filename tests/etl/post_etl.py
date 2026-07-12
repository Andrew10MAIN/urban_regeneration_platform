#!/usr/bin/env python3
"""
Post-ETL Data Quality Pipeline
================================
Verifies data integrity, schema correctness and variable coverage
across all Urban Regeneration Platform database tables.

Checks:
  1. Table loading          – all expected tables readable from DB
  2. Data types             – column dtypes match expected schema
  3. mined.variables        – per-variable coverage, date format, NaN check
  4. Row counts             – basic sanity (non-empty tables)

Usage (from project root, PowerShell):
  python tests\\etl\\post_etl.py

Exit codes:
  0  all checks passed
  1  one or more checks failed
"""

import os
import sys
import datetime
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from geopandas.array import GeometryDtype

# ── Configuration ──────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://urban_user:urban_password@localhost:5433/urban_db",
)

# ── Result collector ───────────────────────────────────────────────────────────

results: list[dict] = []


def record(category: str, label: str, passed: bool, detail: str = "") -> None:
    results.append(
        {"category": category, "label": label, "passed": passed, "detail": detail}
    )
    status = "PASS" if passed else "FAIL"
    suffix = f"  -> {detail}" if detail else ""
    print(f"  [{status}]  {label}{suffix}")


# ── Core test functions (from notebooks) ───────────────────────────────────────

def qualify(table: str) -> str:
    """Quote the table name to handle mixed-case identifiers (e.g. mined.Build_perm)."""
    schema, name = table.split(".", 1)
    return f'{schema}."{name}"'


def compare_dtypes(
    df_dict: dict,
    expected_dtypes_dict: dict,
) -> tuple[bool, str]:
    """
    Compare actual vs expected column dtypes for each table in df_dict.
    Returns (all_passed: bool, message: str).
    """
    messages: list[str] = []

    keys_actual   = set(df_dict.keys())
    keys_expected = set(expected_dtypes_dict.keys())

    for name in keys_expected - keys_actual:
        messages.append(f"Table '{name}' missing from loaded data.")
    for name in keys_actual - keys_expected:
        messages.append(f"Table '{name}' loaded but not in expected schema.")

    for name in keys_actual & keys_expected:
        actual   = df_dict[name].dtypes.to_dict()
        expected = expected_dtypes_dict[name]

        for col in set(expected) - set(actual):
            messages.append(f"[{name}] missing column '{col}' (expected {expected[col]})")
        for col in set(actual) - set(expected):
            messages.append(f"[{name}] unexpected column '{col}' (type {actual[col]})")
        for col in set(actual) & set(expected):
            if actual[col] != expected[col]:
                messages.append(
                    f"[{name}] '{col}': got {actual[col]}, expected {expected[col]}"
                )

    if not messages:
        return True, "All dtype checks passed"
    return False, "; ".join(messages)


def test_mined_variables(
    name_arg: str,
    mined_variables_arg: pd.DataFrame,
    urban_blocks_arg: gpd.GeoDataFrame,
    check_all_blocks: bool   = True,
    check_non_zero_blocks: bool = True,
    check_date_format: bool  = True,
    check_post_period: bool  = True,
    check_pre_period: bool   = True,
    check_nans: bool         = True,
) -> tuple[bool, dict]:
    """
    Runs a set of diagnostic checks on a single variable in mined.variables.
    Returns (passed: bool, issues: dict).
    """
    df_var = (
        mined_variables_arg[mined_variables_arg["var_id"] == name_arg]
        .copy()
        .rename(columns={"value": name_arg})
        .drop(columns="var_id")
        .sort_values(["block_id", "year"])
        .reset_index(drop=True)
    )

    n_blocks = urban_blocks_arg["block_id"].nunique()

    all_blocks_check      = df_var["year"].value_counts().eq(n_blocks).all()
    all_non0_blocks_check = (
        df_var[df_var[name_arg] != 0]["year"].value_counts().eq(n_blocks).all()
    )
    check_0101 = (
        set(str(c)[4:10] for c in df_var["year"].unique()) == {"-01-01"}
    )
    years      = sorted(set(int(str(c)[:4]) for c in df_var["year"].unique()))
    post_check = any(x >= 2025 for x in years)
    pre_check  = any(x <= 2019 for x in years)
    nas_check  = df_var[name_arg].isna().sum() == 0

    issues: dict = {}

    # Collect detail on failures
    blocks_years = urban_blocks_arg[["block_id"]].merge(
        pd.DataFrame({"year": df_var["year"].unique()}), how="cross"
    )
    df_full = blocks_years.merge(
        df_var[["block_id", "year", name_arg]], on=["block_id", "year"], how="left"
    )
    df_missing = df_full[df_full[name_arg].isna()].reset_index(drop=True)
    df_zeros   = df_var[df_var[name_arg] == 0].copy()

    if check_all_blocks and not all_blocks_check:
        issues["missing_blocks"] = (
            f"{len(df_missing)} block-year combinations missing"
        )
    if check_non_zero_blocks and not all_non0_blocks_check:
        issues["zero_value_blocks"] = (
            f"{len(df_zeros)} block-years with value=0"
        )
    if check_date_format and not check_0101:
        bad = sorted(set(str(c)[:10] for c in df_var["year"].unique())
                     - {f"{y}-01-01" for y in years})
        issues["bad_date_format"] = f"non-YYYY-01-01 dates: {bad}"
    if check_pre_period and not pre_check:
        issues["pre_period_missing"] = f"no years <= 2019, found: {years}"
    if check_post_period and not post_check:
        issues["post_period_missing"] = f"no years >= 2025, found: {years}"
    if check_nans and not nas_check:
        issues["nan_count"] = f"{df_var[name_arg].isna().sum()} NaN values"

    tests = []
    if check_all_blocks:      tests.append(all_blocks_check)
    if check_non_zero_blocks: tests.append(all_non0_blocks_check)
    if check_date_format:     tests.append(check_0101)
    if check_pre_period:      tests.append(pre_check)
    if check_post_period:     tests.append(post_check)
    if check_nans:            tests.append(nas_check)

    return all(tests), issues


# ── Expected schemas ───────────────────────────────────────────────────────────

EXPECTED_DTYPES: dict[str, dict] = {

    "core.urban_blocks": {
        "block_id":    np.dtype("int64"),
        "year":        np.dtype("<M8[ns]"),
        "treated_all": np.dtype("int64"),
        "treated_d1nq":np.dtype("int64"),
        "treated_1nq": np.dtype("int64"),
    },
    "core.urban_blocks_geom": {
        "block_id": np.dtype("int64"),
        "area":     np.dtype("float64"),
        "geometry": GeometryDtype(),
    },
    "mined.variables": {
        "var_id":   np.dtype("O"),
        "year":     np.dtype("<M8[ns]"),
        "block_id": np.dtype("int64"),
        "value":    np.dtype("float64"),
    },
    "meta.var_description": {
        "var_id":      np.dtype("O"),
        "unit":        np.dtype("O"),
        "origin":      np.dtype("O"),
        "description": np.dtype("O"),
    },
    "mined.Build_perm": {
        "build_perm_id": np.dtype("O"),
        "build_plot_no": np.dtype("O"),
        "block_id":      np.dtype("int64"),
        "date":          np.dtype("<M8[ns]"),
        "description":   np.dtype("O"),
        "geometry":      GeometryDtype(),
    },
    "mined.adresses": {
        "gml_id":      np.dtype("O"),
        "guid":        np.dtype("O"),
        "full_adress": np.dtype("O"),
        "street":      np.dtype("O"),
        "building_no": np.dtype("O"),
        "zip_code":    np.dtype("O"),
        "status":      np.dtype("O"),
        "geometry":    GeometryDtype(),
    },
    "mined.app_prices": {
        "gml_id":      np.dtype("O"),
        "res_unit_id": np.dtype("O"),
        "building_id": np.dtype("O"),
        "date":        np.dtype("<M8[ns]"),
        "floor_no":    np.dtype("float64"),
        "floor_area":  np.dtype("float64"),
        "price_gross": np.dtype("float64"),
        "block_id":    np.dtype("int64"),
        "geometry":    GeometryDtype(),
    },
    "mined.buildings": {
        "building_id":          np.dtype("O"),
        "floors_above_ground":  np.dtype("float64"),
        "floors_below_ground":  np.dtype("float64"),
        "geometry":             GeometryDtype(),
    },
    "mined.penalties": {
        "pen_id":            np.dtype("int64"),
        "date":              np.dtype("<M8[ns]"),
        "place_of_penalty":  np.dtype("O"),
        "pen_type":          np.dtype("O"),
        "geometry":          GeometryDtype(),
    },
    "regeneration.actions": {
        "reg_id":      np.dtype("int64"),
        "block_id":    np.dtype("int64"),
        "type":        np.dtype("O"),
        "started_at":  np.dtype("<M8[ns]"),
        "finished_at": np.dtype("<M8[ns]"),
        "geometry":    GeometryDtype(),
        "costs":       np.dtype("float64"),
    },
    "osm.poi": {
        "osm_id":  np.dtype("int64"),
        "fclass":  np.dtype("O"),
        "date":    np.dtype("<M8[ns]"),
        "geometry":GeometryDtype(),
    },
    "osm.poly": {
        "osm_id":  np.dtype("int64"),
        "fclass":  np.dtype("O"),
        "date":    np.dtype("<M8[ns]"),
        "geometry":GeometryDtype(),
    },
    "results.uplifts": {
        "block_id":  np.dtype("int64"),
        "target_id": np.dtype("O"),
        "model_id":  np.dtype("O"),
        "uplift":    np.dtype("float64"),
    },
    "results.optimization": {
        "block_id":           np.dtype("int64"),
        "optimization_id":    np.dtype("int64"),
        "designated_to_reg":  np.dtype("int64"),
    },
    "results.predicted_reg_price": {
        "block_id":   np.dtype("int64"),
        "model_id":   np.dtype("O"),
        "pred_price": np.dtype("float64"),
    },
    "results.feature_spec": {
        "features_id": np.dtype("O"),
        "features_no": np.dtype("int64"),
        "feature_id":  np.dtype("O"),
    },
    "results.hypp_spec": {
        "hyppar_id": np.dtype("O"),
        "model_id":  np.dtype("O"),
        "hyppar":    np.dtype("O"),
        "value":     np.dtype("float64"),
    },
    "results.model_setups": {
        "model_id":    np.dtype("O"),
        "model_type":  np.dtype("O"),
        "target_id":   np.dtype("O"),
        "hyppar_id":   np.dtype("O"),
        "features_id": np.dtype("O"),
    },
}

# Geometry tables (loaded with gpd.read_postgis)
GEO_TABLES = {
    "core.urban_blocks_geom",
    "mined.Build_perm",
    "mined.adresses",
    "mined.app_prices",
    "mined.buildings",
    "mined.penalties",
    "regeneration.actions",
    "osm.poi",
    "osm.poly",
}

# Tables loaded as plain pandas (no geometry)
PLAIN_TABLES = set(EXPECTED_DTYPES.keys()) - GEO_TABLES

# ── Per-variable test configuration ───────────────────────────────────────────
# Flags: [check_all_blocks, check_non_zero_blocks, check_date_format,
#          check_post_period, check_pre_period, check_nans]

VAR_CHECK_CONFIG: dict[str, list[bool]] = {
    "bdEnvKnrg_coun_00000000": [True,  False, True, True,  True,  True],
    "urVibBlPr_coun_00000000": [True,  False, True, True,  True,  True],
    "bdEnvScrb_arrt_00000000": [True,  False, True, True,  True,  True],
    "bdEnvCaPr_arrt_00000000": [True,  False, True, True,  True,  True],
    "bdEnvFRxx_arrt_00000000": [True,  False, True, True,  True,  True],
    "urVibEnAl_coun_00000000": [False, False, False,True,  False, True],
    "urVibEnBl_coun_00000000": [False, False, True, True,  True,  True],
    "socVrEduc_avrg_00000000": [False, False, True, True,  True,  True],
    "socVrUnmp_avrg_00000000": [False, False, True, True,  True,  True],
    "socVrAgex_avrg_00000000": [False, False, True, False, True,  True],
    "socVrWage_avrg_00000000": [False, False, True, True,  True,  True],
    "urVibPrUs_arrt_00000000": [True,  False, True, False, True,  True],
    "bdEnvFRab_arrt_00000000": [True,  False, True, True,  True,  True],
    "urVibAlPn_coun_00000000": [True,  False, True, False, True,  True],
    "urVibOfPn_coun_00000000": [True,  False, True, False, True,  True],
    "bdEnvFtxx_arrt_00000000": [True,  False, True, True,  True,  True],
    "socVrAS00_avrg_00000000": [False, False, True, True,  True,  True],
    "urVibSCBx_coun_00000000": [True,  False, True, True,  True,  True],
    "urVibAOut_coun_00000000": [True,  False, True, True,  True,  True],
    "bdEnvPlgr_coun_00000000": [True,  False, True, True,  True,  True],
    "bdEnvSchl_coun_00000000": [True,  False, True, True,  True,  True],
    "socVrPopt_coun_00000000": [True,  False, True, False, False, True],
}

# ── Load tables ────────────────────────────────────────────────────────────────

def load_tables(engine) -> tuple[dict, list[str]]:
    """
    Loads all expected tables. Returns (data_dict, load_errors).
    Tables that fail to load are recorded in load_errors and absent from data_dict.
    """
    data: dict = {}
    errors: list[str] = []

    print("\n[TABLES] Loading data from database...")

    for table_key in EXPECTED_DTYPES:
        label = table_key
        try:
            if table_key in GEO_TABLES:
                q_name = qualify(table_key) if "." in table_key else table_key
                df = gpd.read_postgis(
                    f"SELECT * FROM {q_name}", engine, geom_col="geometry"
                )
            else:
                schema, name = table_key.split(".", 1)
                df = pd.read_sql(f'SELECT * FROM {schema}."{name}"', engine)

            data[table_key] = df
            record("LOAD", label, True, f"{len(df):,} rows")
        except Exception as exc:
            short_err = str(exc).split("\n")[0][:100]
            record("LOAD", label, False, short_err)
            errors.append(table_key)

    return data, errors


# ── Check functions ────────────────────────────────────────────────────────────

def check_dtypes(data: dict) -> None:
    print("\n[DTYPES] Checking column data types...")

    # Split into groups for cleaner reporting
    groups = {
        "core":         ["core.urban_blocks", "core.urban_blocks_geom"],
        "mined":        [k for k in EXPECTED_DTYPES if k.startswith("mined.")],
        "meta":         ["meta.var_description"],
        "osm":          [k for k in EXPECTED_DTYPES if k.startswith("osm.")],
        "regeneration": ["regeneration.actions"],
        "results":      [k for k in EXPECTED_DTYPES if k.startswith("results.")],
    }

    for group, tables in groups.items():
        subset_data     = {t: data[t] for t in tables if t in data}
        subset_expected = {t: EXPECTED_DTYPES[t] for t in tables}
        passed, msg = compare_dtypes(subset_data, subset_expected)
        # Report per table for clarity
        for t in tables:
            if t not in data:
                record("DTYPES", t, False, "table not loaded")
                continue
            single_data     = {t: data[t]}
            single_expected = {t: EXPECTED_DTYPES[t]}
            ok, detail = compare_dtypes(single_data, single_expected)
            record("DTYPES", t, ok, "" if ok else detail[:120])


def check_row_counts(data: dict) -> None:
    print("\n[ROW COUNTS] Basic non-empty sanity checks...")

    # Tables expected to always have rows
    must_have_rows = [
        "core.urban_blocks",
        "core.urban_blocks_geom",
        "mined.variables",
        "meta.var_description",
        "osm.poi",
        "osm.poly",
        "mined.Build_perm",
        "mined.adresses",
        "mined.app_prices",
        "mined.buildings",
        "mined.penalties",
        "regeneration.actions",
        "audit.etl_log",
    ]
    # Audit staging tables are expected non-empty only after ETL runs
    may_be_empty = {
        "audit.stg_osm_poi", "audit.stg_osm_poly",
        "results.optimization", "results.predicted_reg_price",
        "results.uplifts", "results.feature_spec",
        "results.hypp_spec", "results.model_setups",
    }

    for table in must_have_rows:
        if table not in data:
            record("ROW COUNT", table, False, "table not loaded")
            continue
        n = len(data[table])
        record("ROW COUNT", table, n > 0, f"{n:,} rows" if n > 0 else "EMPTY")


def check_mined_variables(data: dict) -> None:
    print("\n[MINED VARS] Checking mined.variables per variable...")

    if "mined.variables" not in data or "core.urban_blocks_geom" not in data:
        record("MINED VARS", "mined.variables", False,
               "Cannot run – mined.variables or urban_blocks_geom not loaded")
        return

    mv   = data["mined.variables"]
    blks = data["core.urban_blocks_geom"]

    db_var_ids  = set(mv["var_id"].unique())
    cfg_var_ids = set(VAR_CHECK_CONFIG.keys())

    # Vars in DB but not in config
    unknown = db_var_ids - cfg_var_ids
    if unknown:
        record("MINED VARS", "Unknown var_ids in DB", False,
               f"{len(unknown)} vars not in test config: {sorted(unknown)[:5]}...")

    # Vars in config but not in DB
    absent = cfg_var_ids - db_var_ids
    if absent:
        for v in sorted(absent):
            record("MINED VARS", v, False, "var_id not found in mined.variables")

    # Run checks for vars present in both
    for var_id in sorted(cfg_var_ids & db_var_ids):
        flags = VAR_CHECK_CONFIG[var_id]
        try:
            passed, issues = test_mined_variables(
                var_id, mv, blks,
                check_all_blocks    = flags[0],
                check_non_zero_blocks = flags[1],
                check_date_format   = flags[2],
                check_post_period   = flags[3],
                check_pre_period    = flags[4],
                check_nans          = flags[5],
            )
            detail = "; ".join(f"{k}={v}" for k, v in issues.items()) if issues else ""
            record("MINED VARS", var_id, passed, detail)
        except Exception as exc:
            record("MINED VARS", var_id, False, str(exc)[:120])


# ── Banner / summary ───────────────────────────────────────────────────────────

def print_banner(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}")
    print(f" {title}")
    print(f" Urban Regeneration Platform")
    print(f" Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(line)


def print_summary() -> None:
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed
    line    = "=" * 64

    # Group failures by category
    failures_by_cat: dict[str, list[str]] = {}
    for r in results:
        if not r["passed"]:
            failures_by_cat.setdefault(r["category"], []).append(
                f"{r['label']}" + (f": {r['detail']}" if r["detail"] else "")
            )

    print(f"\n{line}")
    print(f" SUMMARY")
    print(f" Total checks : {total}")
    print(f" Passed       : {passed}")
    print(f" Failed       : {failed}")
    print(line)

    if failures_by_cat:
        print("\nFailed checks by category:")
        for cat, items in failures_by_cat.items():
            print(f"\n  [{cat}]")
            for item in items:
                print(f"    - {item}")
    else:
        print("\nAll checks passed.")

    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    print_banner("POST-ETL DATA QUALITY PIPELINE")

    # DB connectivity
    print("\n[DATABASE]")
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()
        record("DATABASE", "PostgreSQL connectivity", True, version.split(",")[0])
    except Exception as exc:
        record("DATABASE", "PostgreSQL connectivity", False, str(exc))
        print_summary()
        return 1

    # Load + check
    data, load_errors = load_tables(engine)
    check_dtypes(data)
    check_row_counts(data)
    check_mined_variables(data)
    print_summary()

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
