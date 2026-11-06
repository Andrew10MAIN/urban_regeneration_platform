"""
Data loading and preparation for the CausalForest ML pipeline.

Replicates the "Data preparation" section of ml_experimenting.ipynb, with two
additions needed because the variables in mined.variables do not all cover the
same years or the same blocks:

  * Year resolution — a variable is read at pre_period when available; if not,
    the loader falls back within +/- pre_period_deviation years and, failing
    that, to the nearest year still earlier than post_period. The year actually
    used is logged and stored in results.features / results.models.

  * Block alignment — Y, X and T are built on the intersection of the block
    sets of the target (both periods), every confounder and urban_blocks at
    post_period. Variables with partial coverage therefore narrow the sample
    instead of aborting the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import reduce

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.ml.config import ModelConfig

log = logging.getLogger(__name__)


@dataclass
class DataBundle:
    """All arrays and dataframes needed by the model."""
    Y: np.ndarray             # (N, 1) – target: post - pre difference
    X: np.ndarray             # (N, k) – confounders at their resolved year
    T: np.ndarray             # (N, 2) – treatment indicators [d1nq, 1nq]
    uplift_base: pd.DataFrame # block_id + treatment columns, sorted by block_id
    actual_confounders: list[str]        # confounders that had data (some may be skipped)
    confounder_years: dict[str, int]     # {var_id: year actually used}
    n_blocks: int
    target_pre_year: int = 0             # year actually used as the pre-period
    target_post_year: int = 0
    block_ids: list[int] = field(default_factory=list)


def _years_available(mined: pd.DataFrame, var: str) -> list[int]:
    return sorted(mined.loc[mined["var_id"] == var, "year"].dt.year.unique().tolist())


def _slice_year(mined: pd.DataFrame, var: str, year: int, col_name: str) -> pd.DataFrame:
    return (
        mined[(mined["var_id"] == var) &
              (mined["year"] == pd.Timestamp(f"{year}-01-01"))]
        [["block_id", "value"]]
        .rename(columns={"value": col_name})
        .reset_index(drop=True)
    )


def _resolve_year(
    mined: pd.DataFrame,
    var: str,
    pre_period: int,
    deviation: int,
    post_period: int,
    role: str,
) -> int | None:
    """
    Pick the year to read `var` from, as close to pre_period as the data allows.

    Order of preference:
      1. pre_period itself
      2. pre_period -/+ d for d = 1..deviation (earlier year first)
      3. the available year closest to pre_period that is still earlier than
         post_period — years before pre_period win ties, since a pre-period
         measurement must precede the outcome

    Returns None when the variable has no usable year at all.
    """
    available = _years_available(mined, var)
    if not available:
        log.warning("%s %s: not present in mined.variables at all", role, var)
        return None

    if pre_period in available:
        return pre_period

    for d in range(1, deviation + 1):
        for candidate in (pre_period - d, pre_period + d):
            if candidate in available and candidate < post_period:
                log.info(
                    "%s %s: no data for %d — using %d (within +/-%d)",
                    role, var, pre_period, candidate, deviation,
                )
                return candidate

    usable = [y for y in available if y < post_period]
    if not usable:
        log.warning(
            "%s %s: only years %s available, none earlier than post_period %d",
            role, var, available, post_period,
        )
        return None

    # closest to pre_period; on a tie prefer the earlier year
    chosen = min(usable, key=lambda y: (abs(y - pre_period), y > pre_period))
    log.warning(
        "%s %s: no data within +/-%d of %d — falling back to %d "
        "(years available: %s)",
        role, var, deviation, pre_period, chosen, available,
    )
    return chosen


def load_data(config: ModelConfig, engine: Engine) -> DataBundle:
    """Load urban_blocks and mined.variables, then build Y / X / T arrays."""
    log.info("Loading urban_blocks and mined.variables from DB...")

    urban_blocks_ng = pd.read_sql("SELECT * FROM core.urban_blocks", engine)
    mined_variables = pd.read_sql("SELECT * FROM mined.variables", engine)
    mined_variables["year"] = pd.to_datetime(mined_variables["year"])

    log.info(
        "Loaded %d block-year rows from urban_blocks, %d variable rows from mined.variables",
        len(urban_blocks_ng), len(mined_variables),
    )

    target = config.target_variable
    post_period = config.post_period

    # ── Target: resolve both periods ─────────────────────────────────────────
    if post_period not in _years_available(mined_variables, target):
        raise ValueError(
            f"Target {target} has no data for post_period {post_period}. "
            f"Years available: {_years_available(mined_variables, target)}"
        )

    pre_year = _resolve_year(
        mined_variables, target, config.pre_period,
        config.pre_period_deviation, post_period, role="Target",
    )
    if pre_year is None:
        raise ValueError(
            f"Target {target} has no usable pre-period value before {post_period} "
            f"— the post-minus-pre difference cannot be computed. "
            f"Years available: {_years_available(mined_variables, target)}"
        )
    if pre_year != config.pre_period:
        log.warning(
            "Target %s: pre-period is %d, not the configured %d",
            target, pre_year, config.pre_period,
        )

    target_post = _slice_year(mined_variables, target, post_period, "post")
    target_pre  = _slice_year(mined_variables, target, pre_year,    "pre")
    log.info(
        "Target %s: %d blocks at %d (post), %d blocks at %d (pre)",
        target, len(target_post), post_period, len(target_pre), pre_year,
    )

    target_df = target_post.merge(target_pre, on="block_id", how="inner")
    target_df[target] = target_df["post"] - target_df["pre"]
    target_df = target_df[["block_id", target]].dropna()

    # ── Confounders: resolve year per variable ───────────────────────────────
    confounder_dfs: list[pd.DataFrame] = []
    actual_confounders: list[str] = []
    confounder_years: dict[str, int] = {}

    for var in config.confounders:
        year_used = _resolve_year(
            mined_variables, var, config.pre_period,
            config.pre_period_deviation, post_period, role="Confounder",
        )
        if year_used is None:
            log.warning("Confounder %s: skipped — no usable year", var)
            continue
        df_c = _slice_year(mined_variables, var, year_used, var).dropna()
        if df_c.empty:
            log.warning("Confounder %s: skipped — no rows for %d", var, year_used)
            continue
        confounder_dfs.append(df_c)
        actual_confounders.append(var)
        confounder_years[var] = year_used

    if not confounder_dfs:
        raise ValueError("No confounder data available — cannot train model.")

    merged_confounders = reduce(
        lambda l, r: l.merge(r, on="block_id", how="inner"),
        confounder_dfs,
    ).dropna()

    # ── Block alignment: intersect every source ──────────────────────────────
    blocks_post = urban_blocks_ng[urban_blocks_ng["year"] == pd.Timestamp(f"{post_period}-01-01")]

    sources = {
        f"target {target}":   set(target_df["block_id"]),
        "confounders":        set(merged_confounders["block_id"]),
        f"urban_blocks@{post_period}": set(blocks_post["block_id"]),
    }
    common = set.intersection(*sources.values())

    log.info("Block coverage per source:")
    for name, ids in sources.items():
        log.info("  %-32s %4d blocks  (%d not in the common set)",
                 name, len(ids), len(ids - common))
    log.info("  %-32s %4d blocks  <- model is estimated on this set",
             "INTERSECTION", len(common))

    if not common:
        raise ValueError(
            "Target, confounders and urban_blocks share no block_id — "
            f"coverage was {{{', '.join(f'{k}: {len(v)}' for k, v in sources.items())}}}"
        )
    if len(common) < 30:
        log.warning(
            "Only %d blocks in the common set — treatment effects will be very "
            "imprecise. Check the coverage of %s.", len(common), target,
        )

    block_ids = sorted(common)

    target_aligned = (
        target_df[target_df["block_id"].isin(common)]
        .sort_values("block_id").reset_index(drop=True)
    )
    confounders_aligned = (
        merged_confounders[merged_confounders["block_id"].isin(common)]
        .sort_values("block_id").reset_index(drop=True)
    )
    uplift_base = (
        blocks_post[blocks_post["block_id"].isin(common)]
        .sort_values("block_id").reset_index(drop=True)
        .drop(columns=["year", "treated_all"])
        .copy()
    )

    Y = target_aligned[[target]].values
    X = confounders_aligned[actual_confounders].values
    T = uplift_base[["treated_d1nq", "treated_1nq"]].to_numpy()

    log.info("Y (target) shape: %s  pre=%d post=%d", Y.shape, pre_year, post_period)
    log.info(
        "X (confounders) shape: %s  features: %s",
        X.shape,
        {v: confounder_years[v] for v in actual_confounders},
    )
    n_d1nq, n_1nq = int((T[:, 0] == 1).sum()), int((T[:, 1] == 1).sum())
    log.info(
        "T (treatment) shape: %s  |  d1nq treated: %d  |  1nq treated: %d",
        T.shape, n_d1nq, n_1nq,
    )
    # Narrowing the sample can leave a treatment arm with (almost) no treated
    # blocks. The ATT of such an arm is not estimable, so it will fail the
    # significance test and be dropped from the optimization objective.
    for arm, n_arm in (("d1nq", n_d1nq), ("1nq", n_1nq)):
        if n_arm < 2:
            log.warning(
                "Only %d treated block(s) in arm %s after alignment — its ATT "
                "cannot be estimated and the arm will not enter the optimization.",
                n_arm, arm,
            )

    # ── Sanity checks ────────────────────────────────────────────────────────
    if not (Y.shape[0] == X.shape[0] == T.shape[0] == len(block_ids)):
        raise ValueError(
            f"Shape mismatch after alignment: Y={Y.shape[0]}, X={X.shape[0]}, "
            f"T={T.shape[0]}, blocks={len(block_ids)}."
        )
    if (
        target_aligned["block_id"].tolist() != block_ids
        or confounders_aligned["block_id"].tolist() != block_ids
        or uplift_base["block_id"].tolist() != block_ids
    ):
        raise ValueError("Row order mismatch after alignment — block_ids do not line up.")
    if np.isnan(Y).any() or np.isnan(X).any():
        raise ValueError("NaN left in Y or X after alignment.")
    log.info("Alignment OK — Y/X/T all have %d rows, identical block order", len(block_ids))

    return DataBundle(
        Y=Y,
        X=X,
        T=T,
        uplift_base=uplift_base,
        actual_confounders=actual_confounders,
        confounder_years=confounder_years,
        n_blocks=len(block_ids),
        target_pre_year=pre_year,
        target_post_year=post_period,
        block_ids=block_ids,
    )
