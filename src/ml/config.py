"""
Configuration dataclasses for the ML + Optimization pipeline.

PipelineConfig     — full pipeline (loaded from YAML)
ModelConfig        — per-model settings (built from common fields + per-model overrides)
OptimizationConfig — ILP optimization parameters
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
    """ILP optimization parameters."""
    opt_id: str
    rf_model_id: str
    cost_limit: float
    spill_on_treated: bool = True
    n_clusters: int = 1
    time_limit: int = 300
    normalize: bool = True
    model_weights: dict[str, float] = field(default_factory=dict)  # {model_id: weight}

    def summary(self) -> str:
        return (
            f"  opt_id={self.opt_id}  rf_model_id={self.rf_model_id}  "
            f"cost_limit={self.cost_limit:,.0f}  n_clusters={self.n_clusters}  "
            f"time_limit={self.time_limit}s  normalize={self.normalize}  "
            f"weights={self.model_weights}"
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
        common = dict(
            model_type=raw["model_type"],
            pre_period=raw["pre_period"],
            post_period=raw["post_period"],
            pre_period_deviation=raw.get("pre_period_deviation", 0),
            hyperparameters=raw.get("hyperparameters", {}),
            random_seed=raw.get("random_seed", 42),
        )

        model_configs = [
            ModelConfig(
                model_id=m["model_id"],
                target_variable=m["target_variable"],
                confounders=m.get("confounders", []),
                **common,
            )
            for m in raw["models"]
        ]

        opt_raw = raw.get("optimization", {})
        optimization = OptimizationConfig(
            opt_id=opt_raw["opt_id"],
            rf_model_id=opt_raw["rf_model_id"],
            cost_limit=float(str(opt_raw["cost_limit"]).replace("_", "")),
            spill_on_treated=opt_raw.get("spill_on_treated", True),
            n_clusters=opt_raw.get("n_clusters", 1),
            time_limit=opt_raw.get("time_limit", 300),
            normalize=opt_raw.get("normalize", True),
            model_weights={str(k): float(v) for k, v in opt_raw.get("model_weights", {}).items()},
        )

        return cls(model_configs=model_configs, optimization=optimization)

    def summary(self) -> str:
        lines = [f"Pipeline — {len(self.model_configs)} model(s):"]
        for mc in self.model_configs:
            lines.append(mc.summary())
        lines.append("Optimization:")
        lines.append(self.optimization.summary())
        return "\n".join(lines)
