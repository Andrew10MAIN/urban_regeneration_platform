"""
ILP Optimization — spatial block selection under budget constraint.

Replicates the "Optimization" section of optimization_experimenting.ipynb.

Steps:
  1. Read uplifts, models, costs, urban_blocks from DB
  2. Find latest model per target_variable; determine minimum common block set
  3. Check statistical significance of uplifts per (model, treatment)
  4. Filter untreated blocks; build weighted uplift matrix
  5. Run PuLP CBC ILP: maximize weighted uplift subject to budget + connectivity
  6. Return uplifts_df_final_results and optimization_df_sum
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy.engine import Engine

from src.ml.config import OptimizationConfig

log = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    uplifts_final: pd.DataFrame       # → results.uplifts_optimization
    summary: pd.DataFrame             # → results.optimization_summary
    weights_df: pd.DataFrame          # → results.optimization_weights_setup


def _calculate_effect_significance(df: pd.DataFrame, columns_list: list[str]) -> pd.DataFrame:
    """ATT + 95% CI significance test per column. Matches notebook function."""
    output = []
    for col in columns_list:
        series = df[col].dropna()
        att = series.mean()
        se  = series.std(ddof=1) / np.sqrt(len(series))
        ci_low  = att - 1.96 * se
        ci_high = att + 1.96 * se
        output.append({
            "model":          col,
            "ci_low":         ci_low,
            "att":            att,
            "ci_high":        ci_high,
            "is_significant": (ci_low > 0) or (ci_high < 0),
        })
    return pd.DataFrame(output)


def _build_neighborhood(gdf: gpd.GeoDataFrame) -> dict[int, list[int]]:
    """
    Build adjacency dict {row_index: [neighbor_row_indices]} using spatial join.
    Falls back from libpysal to geopandas sjoin (predicate='touches').
    """
    n = len(gdf)
    try:
        from libpysal.weights import Queen
        w = Queen.from_dataframe(gdf, use_index=False)
        neigh = {i: sorted(set(w.neighbors[i])) for i in range(n)}
        log.info("Neighborhood built via libpysal Queen (%d arcs)", sum(len(v) for v in neigh.values()))
    except Exception:
        sj = gpd.sjoin(
            gdf[["geometry"]],
            gdf[["geometry"]],
            predicate="touches",
            how="inner",
        )
        neigh: dict[int, list[int]] = {i: [] for i in range(n)}
        for i, j in zip(sj.index, sj["index_right"]):
            if i != j:
                neigh[i].append(int(j))
        neigh = {i: sorted(set(v)) for i, v in neigh.items()}
        log.info("Neighborhood built via gpd.sjoin (%d arcs)", sum(len(v) for v in neigh.values()))
    isolated = sum(1 for i in range(n) if not neigh[i])
    log.info("Blocks: %d | isolated: %d", n, isolated)
    return neigh


def run(config: OptimizationConfig, engine: Engine) -> OptimizationResult:
    """
    Full optimization pipeline matching optimization_experimenting.ipynb.
    Reads from DB (CF uplifts + RF costs must already be saved).
    """
    import pulp

    log.info("=== Optimization: %s ===", config.opt_id)

    # ── 1. Load data from DB ──────────────────────────────────────────────────
    log.info("Loading data from DB...")
    uplifts_df       = pd.read_sql("SELECT * FROM results.uplifts",           engine)
    models_df        = pd.read_sql("SELECT * FROM results.models",             engine)
    pred_prices_df   = pd.read_sql(
        f"SELECT block_id, costs FROM results.predicted_reg_prices "
        f"WHERE model_id = '{config.rf_model_id}'",
        engine,
    )
    urban_blocks_ng  = pd.read_sql("SELECT * FROM core.urban_blocks",          engine)
    urban_blocks_gdf = gpd.read_postgis(
        "SELECT block_id, geometry FROM core.urban_blocks_geom",
        engine, geom_col="geometry",
    )

    if pred_prices_df.empty:
        raise RuntimeError(
            f"No cost predictions found for rf_model_id='{config.rf_model_id}'. "
            "Run the RF cost pipeline first."
        )

    # ── 2. Determine target variables from all CF model_ids in DB ────────────
    # Use only CF models (model_id starts with 'CF')
    cf_models = models_df[models_df["model_id"].str.startswith("CF")].copy()
    list_of_target_vars = cf_models["target_id"].unique().tolist()
    log.info("Target variables found: %s", list_of_target_vars)

    # Latest model per target_id
    models_dict: dict[str, str] = (
        cf_models
        .sort_values("run_at")
        .drop_duplicates("target_id", keep="last")
        .set_index("model_id")["target_id"]
        .to_dict()
    )
    list_of_models = list(models_dict.keys())
    log.info("Using models: %s", models_dict)

    # ── 3. Find minimum common block set ─────────────────────────────────────
    block_counts = (
        uplifts_df[uplifts_df["model_id"].isin(list_of_models)]
        ["model_id"].value_counts().to_dict()
    )
    min_model_id = min(block_counts, key=block_counts.get)
    min_block_ids = (
        uplifts_df[uplifts_df["model_id"] == min_model_id]["block_id"].unique().tolist()
    )
    log.info("Min block set from model %s: %d blocks", min_model_id, len(min_block_ids))

    # ── 4. Treated blocks at post_period ─────────────────────────────────────
    post_ts = pd.Timestamp(f"{list(models_dict.values()) and cf_models['post_period'].max()}-01-01")
    # Use post_period from the models table
    post_period = int(cf_models["post_period"].max())
    post_ts = pd.Timestamp(f"{post_period}-01-01")

    ub_post = urban_blocks_ng[
        (urban_blocks_ng["year"] == post_ts) &
        (urban_blocks_ng["block_id"].isin(min_block_ids))
    ].reset_index(drop=True)

    urban_blocks_d1nq = ub_post[ub_post["treated_d1nq"] == 1]["block_id"].unique().tolist()
    urban_blocks_1nq  = ub_post[ub_post["treated_1nq"]  == 1]["block_id"].unique().tolist()
    untreated_ids     = ub_post[ub_post["treated_all"]   == 0]["block_id"].unique().tolist()
    log.info(
        "Treated d1nq: %d | 1nq: %d | untreated: %d",
        len(urban_blocks_d1nq), len(urban_blocks_1nq), len(untreated_ids),
    )

    # ── 5. Significance check ─────────────────────────────────────────────────
    sel_uplifts = uplifts_df[uplifts_df["model_id"].isin(list_of_models)]

    def _pivot_for_sig(block_ids, treatment):
        return (
            sel_uplifts[
                (sel_uplifts["block_id"].isin(block_ids)) &
                (sel_uplifts["treatment"] == treatment)
            ]
            .pivot(index="block_id", columns="model_id", values="uplift")
            .reset_index()
        )

    piv_d1nq = _pivot_for_sig(urban_blocks_d1nq, "d1nq")
    piv_1nq  = _pivot_for_sig(urban_blocks_1nq,  "1nq")

    sig_d1nq = _calculate_effect_significance(piv_d1nq, list_of_models)
    sig_d1nq["treatment"] = "d1nq"
    sig_1nq  = _calculate_effect_significance(piv_1nq,  list_of_models)
    sig_1nq["treatment"]  = "1nq"

    sign_upl = pd.concat([sig_d1nq, sig_1nq])[
        ["model", "treatment", "ci_low", "att", "ci_high", "is_significant"]
    ].reset_index(drop=True)
    sign_upl = sign_upl[sign_upl["is_significant"]].copy()
    log.info("Significant (model, treatment) pairs:\n%s", sign_upl.to_string(index=False))

    if sign_upl.empty:
        raise RuntimeError("No statistically significant uplift combinations found.")

    # ── 6. Build optimization input matrix ───────────────────────────────────
    untreated_uplifts = (
        uplifts_df[uplifts_df["block_id"].isin(untreated_ids)]
        .merge(sign_upl[["model", "treatment"]], left_on=["model_id", "treatment"],
               right_on=["model", "treatment"], how="inner")
        .drop(columns="model")
    )
    # Replace model_id with target_id for readable column names
    untreated_uplifts["model_id"] = untreated_uplifts["model_id"].replace(models_dict)

    uplifts_wide = (
        untreated_uplifts
        .assign(model_treatment=lambda df: df["model_id"] + "_" + df["treatment"])
        .pivot(index="block_id", columns="model_treatment", values="uplift")
        .reset_index()
    )
    uplifts_wide.columns.name = None

    uplifts_for_opt = uplifts_wide.merge(
        pred_prices_df[["block_id", "costs"]], on="block_id", how="left"
    )
    uplifts_for_opt_gdf = urban_blocks_gdf[["block_id", "geometry"]].merge(
        uplifts_for_opt, on="block_id", how="right"
    ).copy()

    log.info("Optimization matrix: %d blocks × %d columns", *uplifts_for_opt_gdf.shape)

    # ── 7. Build neighborhood ─────────────────────────────────────────────────
    gdf = uplifts_for_opt_gdf.reset_index(drop=True)
    n   = len(gdf)

    direct_cols   = [c for c in gdf.columns if c.endswith("_d1nq")]
    indirect_cols = [c for c in gdf.columns if c.endswith("_1nq") and not c.endswith("_d1nq")]
    log.info("Direct cols (%d): %s", len(direct_cols), direct_cols)
    log.info("Indirect cols (%d): %s", len(indirect_cols), indirect_cols)

    U = gdf[direct_cols + indirect_cols].astype(float).fillna(0.0)
    if config.normalize:
        rng = U.max() - U.min()
        U = (U - U.min()) / rng.replace(0, 1)

    # Build target_id → weight lookup (model_weights keyed by model_id)
    # models_dict: {model_id: target_id}  →  reverse to {target_id: weight}
    target_weights: dict[str, float] = {
        target_id: config.model_weights.get(model_id, 1.0)
        for model_id, target_id in models_dict.items()
    }
    if config.model_weights:
        log.info("Model weights: %s", config.model_weights)
        log.info("Target weights: %s", target_weights)
    else:
        log.info("No model_weights configured — all columns weighted equally (1.0)")

    # Weighted sum: a[i] = Σ w_col * U[col][i]  for direct cols
    a = np.zeros(n)
    for col in direct_cols:
        target_id = col[:-5]          # strip "_d1nq"
        a += target_weights.get(target_id, 1.0) * U[col].values

    # Weighted sum: b[i] = Σ w_col * U[col][i]  for indirect cols
    b = np.zeros(n)
    for col in indirect_cols:
        target_id = col[:-4]          # strip "_1nq"
        b += target_weights.get(target_id, 1.0) * U[col].values

    cost = gdf["costs"].astype(float).values

    neigh = _build_neighborhood(gdf)
    arcs  = [(i, j) for i in range(n) for j in neigh[i]]

    # ── 8. ILP ────────────────────────────────────────────────────────────────
    log.info("Building ILP (n=%d, arcs=%d)...", n, len(arcs))
    prob = pulp.LpProblem("revitalization", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")   # directly treated
    y = pulp.LpVariable.dicts("y", range(n), cat="Binary")   # has ≥1 treated neighbour
    r = pulp.LpVariable.dicts("r", range(n), cat="Binary")   # cluster root
    f = pulp.LpVariable.dicts("f", arcs, lowBound=0)         # flow
    s = pulp.LpVariable.dicts("s", range(n), lowBound=0)     # supply at root

    # Objective
    prob += (
        pulp.lpSum(a[i] * x[i] for i in range(n)) +
        pulp.lpSum(b[i] * y[i] for i in range(n))
    )

    # Budget (direct treatment cost only)
    prob += pulp.lpSum(cost[i] * x[i] for i in range(n)) <= config.cost_limit, "budget"

    # Spillover: y_i = 1 only if at least one neighbour is treated
    BIG = n
    for i in range(n):
        if neigh[i]:
            prob += y[i] <= pulp.lpSum(x[j] for j in neigh[i]), f"spill_{i}"
        else:
            prob += y[i] == 0, f"spill_{i}"
        if not config.spill_on_treated:
            prob += y[i] + x[i] <= 1, f"nodouble_{i}"

    # Connectivity: single-commodity flow
    for i in range(n):
        inflow  = pulp.lpSum(f[(j, i)] for j in neigh[i])
        outflow = pulp.lpSum(f[(i, j)] for j in neigh[i])
        prob += inflow + s[i] - outflow == x[i], f"flow_{i}"
        prob += s[i] <= BIG * r[i], f"supply_{i}"
        prob += r[i] <= x[i], f"root_sel_{i}"

    for (i, j) in arcs:
        prob += f[(i, j)] <= BIG * x[i], f"cap_i_{i}_{j}"
        prob += f[(i, j)] <= BIG * x[j], f"cap_j_{i}_{j}"

    prob += pulp.lpSum(r[i] for i in range(n)) == config.n_clusters, "n_roots"

    log.info("Solving ILP (time_limit=%ds)...", config.time_limit)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=config.time_limit))
    log.info(
        "Solver status: %s | objective: %.4f",
        pulp.LpStatus[prob.status], pulp.value(prob.objective),
    )

    # ── 9. Extract solution ───────────────────────────────────────────────────
    sel  = np.array([int(round(x[i].value() or 0)) for i in range(n)], dtype=bool)
    spil = np.array([int(round(y[i].value() or 0)) for i in range(n)], dtype=bool)

    res = gdf.copy()
    res["treated_direct"]    = sel
    res["treated_spillover"] = spil & ~sel

    selected_direct_ids   = res[res["treated_direct"]]["block_id"].unique().tolist()
    selected_spillover_ids = res[res["treated_spillover"]]["block_id"].unique().tolist()
    cost_used = float(cost[sel].sum())
    log.info(
        "Selected: %d direct | %d spillover | cost used: %.0f / %.0f",
        len(selected_direct_ids), len(selected_spillover_ids),
        cost_used, config.cost_limit,
    )

    # ── 10. Build output dataframes matching notebook ─────────────────────────
    # Rebuild from original uplifts_df (with original model_ids, not target_ids)
    uplifts_tagged = uplifts_df.copy()
    uplifts_tagged["target_id"] = uplifts_tagged["model_id"].replace(models_dict)

    d1nq_results = uplifts_tagged[
        (uplifts_tagged["treatment"] == "d1nq") &
        (uplifts_tagged["target_id"].isin([c[:-5] for c in direct_cols])) &
        (uplifts_tagged["block_id"].isin(selected_direct_ids))
    ]
    onq_results = uplifts_tagged[
        (uplifts_tagged["treatment"] == "1nq") &
        (uplifts_tagged["target_id"].isin([c[:-4] for c in indirect_cols])) &
        (uplifts_tagged["block_id"].isin(selected_spillover_ids))
    ]

    uplifts_final = (
        pd.concat([d1nq_results, onq_results])
        .drop(columns=["target_id"])
        .reset_index(drop=True)
    )
    uplifts_final["optimization_id"] = config.opt_id
    uplifts_final = uplifts_final[
        ["optimization_id", "model_id", "block_id", "treatment", "uplift"]
    ]
    log.info("uplifts_final: %d rows", len(uplifts_final))

    run_at = datetime.datetime.now(tz=datetime.timezone.utc)
    opt_summary = pd.DataFrame([{
        "optimization_id":      config.opt_id,
        "cost_used":            cost_used,
        "cost_limit":           config.cost_limit,
        "treatment_spillovers": config.spill_on_treated,
        "clusters_number":      config.n_clusters,
        "time_limit":           config.time_limit,
        "run_at":               run_at,
    }])

    # Weights table — one row per model_id that appears in models_dict
    effective_weights = {
        model_id: config.model_weights.get(model_id, 1.0)
        for model_id in models_dict
    }
    weights_df = pd.DataFrame([
        {"optimization_id": config.opt_id, "model_id": mid, "weight": w}
        for mid, w in effective_weights.items()
    ])
    log.info("weights_df: %d rows", len(weights_df))

    return OptimizationResult(
        uplifts_final=uplifts_final,
        summary=opt_summary,
        weights_df=weights_df,
    )
