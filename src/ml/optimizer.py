"""
ILP Optimization — spatial block selection under budget constraints.

Replicates the "Optimization – budget scenarios (parallel)" section of
optimization_experimenting.ipynb.

Steps:
  1. Read uplifts, models, costs, urban_blocks from DB
  2. Restrict to the target variables configured for this run; pick the latest
     model per target; determine the minimum common block set
  3. Check statistical significance of uplifts per (model, treatment)
  4. Filter untreated blocks; build the uplift matrix for significant pairs
  5. Weight each uplift column (per target_variable + treatment); optionally
     rescale the weights of the significant columns so that they sum to 1
  6. Solve one PuLP CBC ILP per budget scenario, n_jobs at a time
  7. Return uplifts_final (all scenarios), summary (incl. the real objective
     function value) and weights_df in long format

The ILP itself is unchanged: maximize the weighted uplift of directly treated
blocks plus the weighted spillover uplift of their neighbours, subject to a
budget constraint and a single-commodity-flow connectivity constraint.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
from sqlalchemy import text
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


# Two blocks count as rook neighbours when the line they share is at least this
# long (metres). Guards against slivers where digitised boundaries overlap by a
# numerically irrelevant amount instead of meeting along a real edge.
ROOK_MIN_SHARED_EDGE_M = 0.01


def _build_neighborhood(
    gdf: gpd.GeoDataFrame,
    criterion: str = "rook",
) -> dict[int, list[int]]:
    """
    Build adjacency dict {row_index: [neighbor_row_indices]}.

    criterion:
      "queen" — blocks are neighbours when their boundaries touch at all,
                a single shared corner point included
      "rook"  — blocks are neighbours only when they share a boundary segment
                of positive length, i.e. a common edge

    The distinction matters for the connectivity constraint: under queen a
    selected area may be joined only at a corner, which is not a contiguous
    area on the ground.
    """
    criterion = criterion.lower()
    if criterion not in {"rook", "queen"}:
        raise ValueError(f"proximity_criterion='{criterion}' — expected 'rook' or 'queen'")

    n = len(gdf)
    geoms = gdf.geometry.to_list()

    # Candidate pairs: anything whose boundaries meet. Rook is a subset of queen,
    # so both criteria start here.
    sj = gpd.sjoin(
        gdf[["geometry"]],
        gdf[["geometry"]],
        predicate="touches",
        how="inner",
    )

    neigh: dict[int, list[int]] = {i: [] for i in range(n)}
    rejected = 0
    for i, j in zip(sj.index, sj["index_right"]):
        i, j = int(i), int(j)
        if i == j:
            continue
        if criterion == "rook":
            shared = geoms[i].intersection(geoms[j])
            if shared.is_empty or shared.length < ROOK_MIN_SHARED_EDGE_M:
                rejected += 1          # contact at a corner only
                continue
        neigh[i].append(j)
    neigh = {i: sorted(set(v)) for i, v in neigh.items()}

    arcs = sum(len(v) for v in neigh.values())
    isolated = sum(1 for i in range(n) if not neigh[i])
    log.info(
        "Neighborhood (%s): %d blocks | %d arcs | %d isolated%s",
        criterion, n, arcs, isolated,
        f" | {rejected} corner-only contacts dropped" if criterion == "rook" else "",
    )
    if isolated:
        log.warning(
            "%d block(s) have no %s neighbour — they can only be selected as a "
            "single-block cluster.", isolated, criterion,
        )
    return neigh


def _resolve_weights(
    uplift_cols: list[str],
    configured: dict[str, float],
    normalize: bool,
) -> dict[str, float]:
    """
    Map each uplift column that survived the significance test to its weight.

    Columns absent from the config default to 1.0. When normalize is True the
    weights are divided by their sum, so the weights of the columns actually
    entering the objective add up to 1.
    """
    raw: dict[str, float] = {}
    for col in uplift_cols:
        if col in configured:
            raw[col] = float(configured[col])
        else:
            log.warning("No weight configured for uplift column %s — using 1.0", col)
            raw[col] = 1.0

    if not normalize:
        return raw

    total = sum(raw.values())
    if total <= 0:
        log.warning("Sum of weights is %s — skipping normalization", total)
        return raw

    normalized = {col: w / total for col, w in raw.items()}
    log.info(
        "Weights normalized by %.4f: %s",
        total,
        {c: round(w, 6) for c, w in normalized.items()},
    )
    return normalized


def _solve_scenario(
    cost_limit: float,
    opt_id: str,
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    neigh: dict[int, list[int]],
    arcs: list[tuple[int, int]],
    n: int,
    n_clusters: int,
    spill_on_treated: bool,
    time_limit: int,
    neigh_spill: dict[int, list[int]] | None = None,
    warm_start: dict[str, float] | None = None,
) -> dict:
    """
    Solve the MILP for a single budget limit.

    Two neighbourhoods are used, because contiguity and spillover are not the
    same relation:

      neigh       — drives the connectivity constraint, i.e. which blocks may
                    be joined into the directly treated cluster. Normally rook:
                    a shared edge is required, a shared corner is not enough to
                    call two blocks one contiguous area.
      neigh_spill — drives the spillover constraint, i.e. which blocks pick up
                    an indirect effect from a treated neighbour. Normally
                    queen: an effect crosses a corner just as it crosses an
                    edge. Defaults to neigh when not given.

    warm_start is a {variable_name: value} map from an earlier solve. Because
    the budget only ever grows along the ladder and no other constraint
    changes, a solution feasible at a smaller budget stays feasible here, so it
    can be handed to CBC as a MIP start: the solver begins from it rather than
    from nothing and can only improve on it.

    Module-level and self-contained so joblib's loky backend can ship it to a
    worker process. CBC is an external process, so threads=... is deliberately
    not passed (the bundled CBC fails on it under Windows).
    """
    import pulp
    import numpy as np  # re-imported: runs in a separate process

    if neigh_spill is None:
        neigh_spill = neigh

    prob = pulp.LpProblem(f"revitalization_{opt_id}", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")   # directly treated
    y = pulp.LpVariable.dicts("y", range(n), cat="Binary")   # has >=1 treated neighbour
    r = pulp.LpVariable.dicts("r", range(n), cat="Binary")   # cluster root
    f = pulp.LpVariable.dicts("f", arcs, lowBound=0)         # flow
    s = pulp.LpVariable.dicts("s", range(n), lowBound=0)     # supply at root

    # Objective
    prob += (
        pulp.lpSum(a[i] * x[i] for i in range(n)) +
        pulp.lpSum(b[i] * y[i] for i in range(n))
    )

    # Budget (direct treatment cost only)
    prob += pulp.lpSum(cost[i] * x[i] for i in range(n)) <= cost_limit, "budget"

    # Spillover: y_i = 1 only if at least one neighbour is treated.
    # Uses neigh_spill (queen), so a block touching a treated block at a corner
    # still receives the indirect effect.
    BIG = n
    for i in range(n):
        if neigh_spill[i]:
            prob += y[i] <= pulp.lpSum(x[j] for j in neigh_spill[i]), f"spill_{i}"
        else:
            prob += y[i] == 0, f"spill_{i}"
        if not spill_on_treated:
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

    prob += pulp.lpSum(r[i] for i in range(n)) == n_clusters, "n_roots"

    used_warm_start = False
    if warm_start:
        # The whole variable vector is replayed, flow and supply included — a
        # partial start would leave CBC to repair the flow itself.
        for var in prob.variables():
            value = warm_start.get(var.name)
            if value is not None:
                var.setInitialValue(float(value))
        used_warm_start = True

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit,
                                 warmStart=used_warm_start))

    solver_status   = pulp.LpStatus[prob.status]
    solution_status = pulp.LpSolution[prob.sol_status]

    # A run that hits the time limit before finding any integer solution still
    # leaves fractional values on the variables. Rounding those would produce a
    # selection that violates the budget, so such a run is reported as a
    # failure rather than as a result.
    has_integer_solution = prob.sol_status in (
        pulp.LpSolutionOptimal,
        pulp.LpSolutionIntegerFeasible,
    )

    empty = np.zeros(n, dtype=bool)
    if not has_integer_solution:
        return {
            "optimization_id": opt_id,
            "cost_limit":      float(cost_limit),
            "solver_status":   solver_status,
            "solution_status": solution_status,
            "objective":       None,
            "sel":             empty,
            "spil":            empty.copy(),
            "warm_started":    used_warm_start,
            "vector":          None,
        }

    sel  = np.array([int(round(x[i].value() or 0)) for i in range(n)], dtype=bool)
    spil = np.array([int(round(y[i].value() or 0)) for i in range(n)], dtype=bool)

    # Safety net: the budget must hold for the rounded selection as well.
    cost_used = float(np.asarray(cost)[sel].sum())
    if cost_used > float(cost_limit) + 1e-6:
        return {
            "optimization_id": opt_id,
            "cost_limit":      float(cost_limit),
            "solver_status":   solver_status,
            "solution_status": f"Rejected: cost {cost_used:.0f} over budget",
            "objective":       None,
            "sel":             empty,
            "spil":            empty.copy(),
            "warm_started":    used_warm_start,
            "vector":          None,
        }

    objective = pulp.value(prob.objective)

    return {
        "optimization_id": opt_id,
        "cost_limit":      float(cost_limit),
        "solver_status":   solver_status,
        "solution_status": solution_status,
        "objective":       None if objective is None else float(objective),
        "sel":             sel,
        "spil":            spil & ~sel,
        "warm_started":    used_warm_start,
        # full variable vector, so this solution can seed the next budget step
        "vector":          {v.name: v.varValue for v in prob.variables()
                            if v.varValue is not None},
    }


def _objective_of(result: dict | None) -> float:
    """Objective of a solve, with a failed or missing solve ranked last."""
    if result is None or result.get("objective") is None:
        return float("-inf")
    return float(result["objective"])


def _run_ladder(config: OptimizationConfig, solve_args: dict,
                budget_limits: list[float], opt_ids: list[str]) -> list[dict]:
    """
    Sequential ladder over the budget range.

    Step 1 solves the smallest budget from scratch; its solution becomes the
    incumbent. Every later step runs two solves in parallel at the next budget
    — one seeded with the incumbent, one from scratch — and the incumbent
    itself is carried forward as a third candidate, since it remains feasible
    once the budget grows. The best of the three becomes the new incumbent.

    This is what makes the objective non-decreasing in the budget. In the plain
    parallel sweep every scenario started cold, so a harder instance could end
    the time limit on a worse feasible solution than a smaller budget had
    found, and the curve dipped. Here a step can never report less than the
    previous one, because the previous solution is always among the candidates.
    """
    from joblib import Parallel, delayed

    n_workers = max(1, min(2, config.n_jobs))
    results: list[dict] = []
    incumbent: dict | None = None

    for step, (limit, opt_id) in enumerate(zip(budget_limits, opt_ids), start=1):
        branches: list[tuple[str, dict | None]] = []
        if incumbent is not None:
            branches.append(("warm", incumbent["vector"]))
        branches.append(("cold", None))

        log.info(
            "--- step %d/%d  %s  budget %s  (%s)",
            step, len(budget_limits), opt_id, f"{limit:,.0f}",
            " + ".join(tag for tag, _ in branches),
        )

        solved = Parallel(n_jobs=min(len(branches), n_workers), backend="loky")(
            delayed(_solve_scenario)(
                limit, f"{opt_id}_{tag}", **solve_args,
                time_limit=config.time_limit, warm_start=vector,
            )
            for tag, vector in branches
        )
        candidates = list(zip([tag for tag, _ in branches], solved))

        if incumbent is not None:
            carried = dict(incumbent)
            carried.update(
                cost_limit=float(limit),
                solver_status="CarriedForward",
                solution_status="Previous step reused",
                warm_started=False,
            )
            candidates.append(("carried", carried))

        for tag, res in candidates:
            log.info(
                "      %-8s objective=%s  direct=%d  spillover=%d  %s",
                tag,
                "n/a" if res["objective"] is None else f"{res['objective']:.4f}",
                int(res["sel"].sum()), int(res["spil"].sum()),
                res["solver_status"],
            )

        best_tag, best = max(candidates, key=lambda c: _objective_of(c[1]))
        best = dict(best)
        best["optimization_id"] = opt_id
        best["cost_limit"] = float(limit)

        if _objective_of(best) == float("-inf"):
            # No branch produced an integer solution within the time limit.
            best["branch"] = "failed"
            log.warning(
                "      -> no branch found an integer solution at %s; "
                "raise time_limit if this repeats", f"{limit:,.0f}",
            )
        else:
            best["branch"] = best_tag
            log.info("      -> kept '%s'", best_tag)

        results.append(best)
        # A failed step must not discard a working incumbent — the next step
        # can still be seeded from the last solution that did solve.
        if best.get("vector") and best.get("objective") is not None:
            incumbent = best

    return results


def run(config: OptimizationConfig, engine: Engine) -> OptimizationResult:
    """
    Full optimization sweep matching optimization_experimenting.ipynb.
    Reads from DB (CF uplifts + RF costs must already be saved).
    """
    log.info("=== Optimization sweep: %s ===", config.opt_id_start)

    # ── 1. Load data from DB ──────────────────────────────────────────────────
    log.info("Loading data from DB...")
    uplifts_df       = pd.read_sql("SELECT * FROM results.uplifts",    engine)
    models_df        = pd.read_sql("SELECT * FROM results.models",     engine)
    pred_prices_df   = pd.read_sql(
        text("SELECT block_id, costs FROM results.predicted_reg_prices "
             "WHERE model_id = :rf_model_id"),
        engine, params={"rf_model_id": config.rf_model_id},
    )
    urban_blocks_ng  = pd.read_sql("SELECT * FROM core.urban_blocks", engine)
    urban_blocks_gdf = gpd.read_postgis(
        "SELECT block_id, geometry FROM core.urban_blocks_geom",
        engine, geom_col="geometry",
    )

    if pred_prices_df.empty:
        raise RuntimeError(
            f"No cost predictions found for rf_model_id='{config.rf_model_id}'. "
            "Run the RF cost pipeline first."
        )

    # ── 2. Latest CF model per configured target variable ────────────────────
    cf_models = models_df[models_df["model_id"].str.startswith("CF")].copy()
    if config.target_variables:
        cf_models = cf_models[cf_models["target_id"].isin(config.target_variables)].copy()
        missing = set(config.target_variables) - set(cf_models["target_id"])
        if missing:
            raise RuntimeError(
                f"No CF model in results.models for target variable(s): {sorted(missing)}. "
                "Train the CF models first."
            )
    log.info("Target variables used: %s", cf_models["target_id"].unique().tolist())

    models_dict: dict[str, str] = (
        cf_models
        .sort_values("run_at")
        .drop_duplicates("target_id", keep="last")
        .set_index("model_id")["target_id"]
        .to_dict()
    )
    list_of_models = list(models_dict.keys())
    target_to_model = {target: model for model, target in models_dict.items()}
    log.info("Using models: %s", models_dict)

    # ── 3. Common block set across all models ────────────────────────────────
    # Models may be estimated on different block sets (a target or confounder
    # with partial coverage narrows the sample — see data_loader). The
    # optimization has to run on blocks that every selected model scored,
    # otherwise the pivot below would turn a missing uplift into a zero one.
    sel_for_blocks = uplifts_df[uplifts_df["model_id"].isin(list_of_models)]
    if sel_for_blocks.empty:
        raise RuntimeError(
            "results.uplifts contains no rows for the selected models — "
            "train the CF models first."
        )

    per_model_blocks = {
        model_id: set(group["block_id"])
        for model_id, group in sel_for_blocks.groupby("model_id")
    }
    common_blocks = set.intersection(*per_model_blocks.values())

    log.info("Block coverage per model:")
    for model_id, ids in sorted(per_model_blocks.items()):
        log.info(
            "  %-16s %-28s %4d blocks  (%d outside the common set)",
            model_id, models_dict.get(model_id, "?"), len(ids), len(ids - common_blocks),
        )
    log.info("  %-45s %4d blocks  <- optimization runs on this set",
             "INTERSECTION", len(common_blocks))

    if not common_blocks:
        raise RuntimeError(
            "The selected models share no block_id: "
            + ", ".join(f"{m}: {len(b)}" for m, b in sorted(per_model_blocks.items()))
        )

    min_block_ids = sorted(common_blocks)

    # ── 4. Treated / untreated blocks at post_period ─────────────────────────
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

    # Every block in the matrix must have an uplift for every significant pair.
    # A gap here would be read as a zero uplift by the fillna below and would
    # quietly make that block look unattractive rather than unknown.
    value_cols = [c for c in uplifts_wide.columns if c != "block_id"]
    gaps = uplifts_wide[value_cols].isna()
    if gaps.any().any():
        missing = gaps.sum().loc[lambda s: s > 0].to_dict()
        raise RuntimeError(
            "Missing uplifts in the optimization matrix after restricting to the "
            f"common block set: {missing}. This should not happen — check "
            "results.uplifts for partially written models."
        )

    uplifts_for_opt = uplifts_wide.merge(
        pred_prices_df[["block_id", "costs"]], on="block_id", how="left"
    )
    uplifts_for_opt_gdf = urban_blocks_gdf[["block_id", "geometry"]].merge(
        uplifts_for_opt, on="block_id", how="right"
    ).copy()

    log.info("Optimization matrix: %d blocks × %d columns", *uplifts_for_opt_gdf.shape)

    gdf = uplifts_for_opt_gdf.reset_index(drop=True)
    n   = len(gdf)

    direct_cols   = [c for c in gdf.columns if c.endswith("_d1nq")]
    indirect_cols = [c for c in gdf.columns if c.endswith("_1nq") and not c.endswith("_d1nq")]
    log.info("Direct cols (%d): %s", len(direct_cols), direct_cols)
    log.info("Indirect cols (%d): %s", len(indirect_cols), indirect_cols)

    U = gdf[direct_cols + indirect_cols].astype(float).fillna(0.0)

    # Drop uplift columns with no variation across the candidate blocks. A model
    # that predicts the same effect everywhere carries no information about
    # which block to pick, and min-max normalization maps such a column to all
    # zeros — so keeping it would reserve weight for a column contributing
    # exactly nothing, diluting the models that do vary.
    spread = (U.max() - U.min()).abs()
    flat_cols = spread[spread <= 1e-12].index.tolist()
    if flat_cols:
        log.warning(
            "Uplift column(s) constant across all %d candidate blocks — excluded "
            "from the objective and from weight normalization: %s",
            len(U), flat_cols,
        )
        log.warning(
            "A constant CATE means the causal forest found no heterogeneity. Such "
            "a column says nothing about which block to treat — it would only add "
            "a flat per-block bonus and tilt the solution towards picking more "
            "blocks, which is an artefact of the degenerate fit, not evidence."
        )
        direct_cols   = [c for c in direct_cols   if c not in flat_cols]
        indirect_cols = [c for c in indirect_cols if c not in flat_cols]
        U = U.drop(columns=flat_cols)
        if not direct_cols and not indirect_cols:
            raise RuntimeError(
                "Every uplift column is constant across blocks — the objective "
                "function would be identically zero and the selection arbitrary."
            )

    if config.normalize:
        # Sign-preserving scaling: divide each column by its largest absolute
        # value, mapping it into [-1, 1] with zero fixed at zero. Uplift columns
        # become comparable in magnitude while a negative uplift stays negative
        # and therefore lowers the objective function.
        #
        # NOT min-max ((U - min) / (max - min)): that maps every column onto
        # [0, 1] regardless of sign, so an outcome the programme damages would
        # enter the objective as a gain.
        scale = U.abs().max()
        U = U / scale.replace(0, 1)
        log.info(
            "Uplift columns scaled by max|value| (sign preserved): %s",
            {c: round(float(v), 4) for c, v in scale.items()},
        )
        negatives = {c: int((U[c] < 0).sum()) for c in U.columns if (U[c] < 0).any()}
        if negatives:
            log.info(
                "Blocks with a negative (objective-lowering) uplift per column: %s",
                negatives,
            )

    # ── 7. Weights per uplift column, normalized over significant columns ────
    weights = _resolve_weights(
        direct_cols + indirect_cols,
        config.uplift_weights,
        config.normalize_weights,
    )

    a = np.zeros(n)
    for col in direct_cols:
        a += weights[col] * U[col].values

    b = np.zeros(n)
    for col in indirect_cols:
        b += weights[col] * U[col].values

    cost = gdf["costs"].astype(float).values
    if np.isnan(cost).any():
        n_nan = int(np.isnan(cost).sum())
        raise RuntimeError(
            f"{n_nan} block(s) in the optimization matrix have no RF cost prediction "
            f"for rf_model_id='{config.rf_model_id}'."
        )

    # Two neighbourhoods, because contiguity and spillover are different
    # relations: the directly treated blocks have to form one area joined along
    # edges, while the indirect effect also crosses a corner.
    log.info("Contiguity of the treated cluster:")
    neigh = _build_neighborhood(gdf, config.proximity_criterion)
    arcs  = [(i, j) for i in range(n) for j in neigh[i]]

    if config.spillover_criterion == config.proximity_criterion:
        neigh_spill = neigh
    else:
        log.info("Spillover reach:")
        neigh_spill = _build_neighborhood(gdf, config.spillover_criterion)

    extra_spill = sum(
        len(set(neigh_spill[i]) - set(neigh[i])) for i in range(n)
    )
    log.info(
        "Spillover reaches %d block pair(s) that are not contiguous "
        "(corner-only contacts)", extra_spill,
    )

    # ── 8. Budget sweep ───────────────────────────────────────────────────────
    budget_limits = config.budget_limits
    opt_ids       = config.opt_ids
    solve_args = dict(a=a, b=b, cost=cost, neigh=neigh, neigh_spill=neigh_spill,
                      arcs=arcs, n=n,
                      n_clusters=config.n_clusters,
                      spill_on_treated=config.spill_on_treated)

    log.info(
        "Solving %d ILP scenarios (n=%d, arcs=%d, contiguity=%s, spillover=%s) — "
        "%s, time_limit=%ds per solve",
        len(budget_limits), n, len(arcs),
        config.proximity_criterion, config.spillover_criterion,
        "sequential ladder" if config.sequential else f"parallel, n_jobs={config.n_jobs}",
        config.time_limit,
    )
    log.info(
        "Budget range: %s .. %s step %s",
        f"{budget_limits[0]:,.0f}",
        f"{budget_limits[-1]:,.0f}",
        f"{config.budget_step:,.0f}",
    )

    if config.sequential:
        scenario_results = _run_ladder(config, solve_args, budget_limits, opt_ids)
    else:
        from joblib import Parallel, delayed
        scenario_results = Parallel(n_jobs=config.n_jobs, backend="loky", verbose=5)(
            delayed(_solve_scenario)(
                cost_limit, opt_id, **solve_args, time_limit=config.time_limit,
            )
            for cost_limit, opt_id in zip(budget_limits, opt_ids)
        )
        for res_k in scenario_results:
            res_k["branch"] = "cold"

    for res_k in scenario_results:
        log.info(
            "  %s  budget=%12.0f  %-14s %-24s objective=%s  direct=%d  spillover=%d  [%s]",
            res_k["optimization_id"], res_k["cost_limit"],
            res_k["solver_status"], res_k["solution_status"],
            "n/a" if res_k["objective"] is None else f"{res_k['objective']:.4f}",
            int(res_k["sel"].sum()), int(res_k["spil"].sum()),
            res_k.get("branch", "?"),
        )

    objectives = [r["objective"] for r in scenario_results]
    drops = [
        (scenario_results[i]["optimization_id"], objectives[i], objectives[i + 1])
        for i in range(len(objectives) - 1)
        if objectives[i] is not None and objectives[i + 1] is not None
        and objectives[i + 1] < objectives[i] - 1e-9
    ]
    if drops:
        log.warning("Objective decreased on %d step(s) despite the ladder: %s",
                    len(drops), drops)
    else:
        log.info("Objective is non-decreasing across the whole budget range.")

    # ── 9. Assemble per-scenario uplift rows ─────────────────────────────────
    uplifts_tagged = uplifts_df.copy()
    uplifts_tagged["target_id"] = uplifts_tagged["model_id"].replace(models_dict)

    direct_targets   = [c[:-5] for c in direct_cols]    # strip "_d1nq"
    indirect_targets = [c[:-4] for c in indirect_cols]  # strip "_1nq"

    uplift_parts = []
    for res_k in scenario_results:
        direct_blocks    = gdf.loc[res_k["sel"],  "block_id"].unique().tolist()
        spillover_blocks = gdf.loc[res_k["spil"], "block_id"].unique().tolist()

        part_d1nq = uplifts_tagged[
            (uplifts_tagged["treatment"] == "d1nq") &
            (uplifts_tagged["target_id"].isin(direct_targets)) &
            (uplifts_tagged["block_id"].isin(direct_blocks))
        ]
        part_1nq = uplifts_tagged[
            (uplifts_tagged["treatment"] == "1nq") &
            (uplifts_tagged["target_id"].isin(indirect_targets)) &
            (uplifts_tagged["block_id"].isin(spillover_blocks))
        ]
        part = pd.concat([part_d1nq, part_1nq]).copy()
        part["optimization_id"] = res_k["optimization_id"]
        uplift_parts.append(part)

    uplifts_final = (
        pd.concat(uplift_parts)
        .drop(columns=["target_id"])
        .reset_index(drop=True)
    )
    uplifts_final = uplifts_final[
        ["optimization_id", "model_id", "block_id", "treatment", "uplift"]
    ]
    log.info("uplifts_final: %d rows across %d scenarios",
             len(uplifts_final), len(scenario_results))

    # ── 10. Summary — one row per scenario, with the real objective value ────
    run_at = datetime.datetime.now(tz=datetime.timezone.utc)
    opt_summary = pd.DataFrame([
        {
            "optimization_id":          res_k["optimization_id"],
            "cost_used":                float(cost[res_k["sel"]].sum()),
            "cost_limit":               res_k["cost_limit"],
            "objective_function_value": res_k["objective"],
            "solver_status":            res_k["solver_status"],
            "solution_status":          res_k["solution_status"],
            "proximity_criterion":      config.proximity_criterion,
            "spillover_criterion":      config.spillover_criterion,
            "treatment_spillovers":     config.spill_on_treated,
            "clusters_number":          config.n_clusters,
            "time_limit":               config.time_limit,
            "run_at":                   run_at,
        }
        for res_k in scenario_results
    ])

    # ── 11. Weights, long format — one row per (optimization, model, uplift) ─
    weights_df = pd.DataFrame([
        {
            "optimization_id": res_k["optimization_id"],
            "model_id":        target_to_model[col[:-5] if col.endswith("_d1nq") else col[:-4]],
            "uplift":          col,
            "uplift_weight":   weight,
        }
        for res_k in scenario_results
        for col, weight in weights.items()
    ])
    log.info("weights_df: %d rows (%d uplift columns × %d scenarios)",
             len(weights_df), len(weights), len(scenario_results))

    return OptimizationResult(
        uplifts_final=uplifts_final,
        summary=opt_summary,
        weights_df=weights_df,
    )
