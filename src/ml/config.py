"""
Configuration dataclasses for the ML + Optimization pipeline.

PipelineConfig     — full pipeline (loaded from YAML)
ModelConfig        — per-model settings (built from common fields + per-model overrides)
OptimizationConfig — ILP optimization parameters (parallel budget sweep)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _num(value: Any) -> float:
    """Parse a YAML number that may be written with underscores (250_000_000)."""
    return float(str(value).replace("_", ""))


@dataclass
class ModelConfig:
    """
    Per-model configuration — passed unchanged to data_loader and model modules.
    Identical field set as before, so data_loader.py and model.py need no changes.
    """
    model_id: str
    model_type: str
    target_variable: str
    pre_period: int
    post_period: int
    pre_period_deviation: int
    confounders: list[str]
    hyperparameters: dict[str, Any]
    random_seed: int

    def override(self, **kwargs: Any) -> "ModelConfig":
        return dataclasses.replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    def summary(self) -> str:
        return (
            f"  model_id={self.model_id}  target={self.target_variable}  "
            f"pre={self.pre_period}  post={self.post_period}  "
            f"confounders={self.confounders}"
        )


@dataclass
class OptimizationConfig:
    """
    ILP optimization parameters.

    The optimizer solves one ILP per budget scenario. Scenarios are generated
    from budget_min / budget_max / budget_step (budget_max exclusive, as in
    Python's range) and solved in parallel, n_jobs at a time, each capped at
    time_limit seconds of CBC runtime.
    """
    opt_id_start: str
    rf_model_id: str

    budget_min: float
    budget_max: float
    budget_step: float

    n_jobs: int = 12
    time_limit: int = 240

    spill_on_treated: bool = True
    n_clusters: int = 1
    # Contiguity of the directly treated cluster:
    #   "rook"  — blocks must share a boundary segment of positive length
    #   "queen" — a shared corner point is enough
    proximity_criterion: str = "rook"
    # Reach of the indirect effect around the treated cluster. Queen by
    # default: a spillover crosses a corner even though a corner does not make
    # two blocks one contiguous area.
    spillover_criterion: str = "queen"
    # True runs the budgets as a ladder: each step seeds one solve with the
    # previous winner and runs a second from scratch, keeping the better of the
    # two (and of the previous solution itself). Guarantees a non-decreasing
    # objective. False restores the earlier all-cold parallel sweep.
    sequential: bool = True
    # Scale uplift columns by max|value| before the ILP (sign preserved, so a
    # negative uplift lowers the objective).
    normalize: bool = True
    purge_previous: bool = False

    normalize_weights: bool = True
    # Weights keyed by uplift column: "<target_variable>_<treatment>"
    uplift_weights: dict[str, float] = field(default_factory=dict)

    # Target variables considered by the optimizer. Defaults to the targets of
    # the CF models defined in the same YAML, so stale models left in
    # results.models from earlier runs are ignored.
    target_variables: list[str] = field(default_factory=list)

    @property
    def budget_limits(self) -> list[float]:
        """Budget scenarios, ascending. budget_max is exclusive."""
        if self.budget_step <= 0:
            raise ValueError("budget_step must be > 0")
        limits: list[float] = []
        value = self.budget_min
        while value < self.budget_max:
            limits.append(float(value))
            value += self.budget_step
        if not limits:
            raise ValueError(
                f"Empty budget sweep: min={self.budget_min}, max={self.budget_max}, "
                f"step={self.budget_step}"
            )
        return limits

    @property
    def opt_ids(self) -> list[str]:
        """Sequential optimization ids, one per budget scenario."""
        prefix = self.opt_id_start.rstrip("0123456789")
        digits = self.opt_id_start[len(prefix):]
        if not digits:
            raise ValueError(
                f"opt_id_start must end with digits, got {self.opt_id_start!r}"
            )
        start = int(digits)
        width = len(digits)
        return [
            f"{prefix}{start + k:0{width}d}"
            for k in range(len(self.budget_limits))
        ]

    def summary(self) -> str:
        limits = self.budget_limits
        return (
            f"  opt_id_start={self.opt_id_start}  rf_model_id={self.rf_model_id}\n"
            f"  budgets: {len(limits)} scenarios  "
            f"{limits[0]:,.0f} .. {limits[-1]:,.0f}  step {self.budget_step:,.0f}\n"
            f"  mode={'sequential ladder' if self.sequential else 'parallel'}  "
            f"n_jobs={self.n_jobs}  time_limit={self.time_limit}s\n"
            f"  contiguity={self.proximity_criterion}  "
            f"spillover={self.spillover_criterion}  n_clusters={self.n_clusters}  "
            f"normalize={self.normalize}  purge_previous={self.purge_previous}\n"
            f"  targets={self.target_variables}\n"
            f"  normalize_weights={self.normalize_weights}  "
            f"uplift_weights={self.uplift_weights}"
        )


@dataclass
class PipelineConfig:
    """
    Full pipeline configuration loaded from YAML.
    Contains N ModelConfigs (one per CF model) + one OptimizationConfig.
    """
    model_configs: list[ModelConfig]
    optimization: OptimizationConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Common fields shared by all models
        common_hyperparameters = raw.get("hyperparameters", {})
        common = dict(
            model_type=raw["model_type"],
            pre_period=raw["pre_period"],
            post_period=raw["post_period"],
            pre_period_deviation=raw.get("pre_period_deviation", 0),
            random_seed=raw.get("random_seed", 42),
        )

        # A model may override single hyperparameters; anything it does not
        # mention is inherited from the shared block. Needed because the right
        # forest settings depend on the sample size, which differs per model
        # when a target or confounder has partial block coverage.
        model_configs = [
            ModelConfig(
                model_id=m["model_id"],
                target_variable=m["target_variable"],
                confounders=m.get("confounders", []),
                hyperparameters={
                    **common_hyperparameters,
                    **(m.get("hyperparameters") or {}),
                },
                **common,
            )
            for m in raw["models"]
        ]

        opt_raw = raw.get("optimization", {})

        # Budget sweep. A single `cost_limit` (old config format) is accepted
        # and treated as a one-scenario sweep.
        if "budget_min" in opt_raw:
            budget_min  = _num(opt_raw["budget_min"])
            budget_max  = _num(opt_raw.get("budget_max", budget_min + 1))
            budget_step = _num(opt_raw.get("budget_step", 1))
        elif "cost_limit" in opt_raw:
            budget_min  = _num(opt_raw["cost_limit"])
            budget_max  = budget_min + 1
            budget_step = 1.0
        else:
            raise KeyError(
                "optimization: either budget_min/budget_max/budget_step "
                "or cost_limit must be set"
            )

        optimization = OptimizationConfig(
            opt_id_start=opt_raw.get("opt_id_start") or opt_raw["opt_id"],
            rf_model_id=opt_raw["rf_model_id"],
            budget_min=budget_min,
            budget_max=budget_max,
            budget_step=budget_step,
            n_jobs=int(opt_raw.get("n_jobs", 12)),
            time_limit=int(opt_raw.get("time_limit", 240)),
            spill_on_treated=opt_raw.get("spill_on_treated", True),
            n_clusters=int(opt_raw.get("n_clusters", 1)),
            proximity_criterion=str(opt_raw.get("proximity_criterion", "rook")).lower(),
            spillover_criterion=str(opt_raw.get("spillover_criterion", "queen")).lower(),
            sequential=opt_raw.get("sequential", True),
            normalize=opt_raw.get("normalize", True),
            purge_previous=opt_raw.get("purge_previous", False),
            normalize_weights=opt_raw.get("normalize_weights", True),
            uplift_weights={
                str(k): float(v) for k, v in (opt_raw.get("uplift_weights") or {}).items()
            },
            target_variables=[
                str(t) for t in (
                    opt_raw.get("target_variables")
                    or [m.target_variable for m in model_configs]
                )
            ],
        )

        return cls(model_configs=model_configs, optimization=optimization)

    def summary(self) -> str:
        lines = [f"Pipeline — {len(self.model_configs)} model(s):"]
        for mc in self.model_configs:
            lines.append(mc.summary())
        lines.append("Optimization:")
        lines.append(self.optimization.summary())
        return "\n".join(lines)
