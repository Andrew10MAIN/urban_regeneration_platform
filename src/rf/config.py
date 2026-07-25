"""
Configuration dataclass for the RF Regeneration Cost pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RFConfig:
    model_id: str
    model_type: str
    random_seed: int
    pre_period: int
    post_period: int
    hyperparameters: dict[str, Any]
    excluded_vars: list[str] = field(default_factory=list)
    special_year_vars: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RFConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            model_id=raw["model_id"],
            model_type=raw["model_type"],
            random_seed=raw["random_seed"],
            pre_period=raw["pre_period"],
            post_period=raw["post_period"],
            hyperparameters=raw.get("hyperparameters", {}),
            excluded_vars=raw.get("excluded_vars", []),
            special_year_vars=raw.get("special_year_vars", {}),
        )

    def override(self, **kwargs: Any) -> "RFConfig":
        """Return a new config with selected fields overridden (e.g. from CLI args)."""
        import copy
        new = copy.deepcopy(self)
        for k, v in kwargs.items():
            if v is not None and hasattr(new, k):
                setattr(new, k, v)
        return new

    def summary(self) -> str:
        lines = [
            f"model_id    : {self.model_id}",
            f"model_type  : {self.model_type}",
            f"pre_period  : {self.pre_period}",
            f"post_period : {self.post_period}",
            f"random_seed : {self.random_seed}",
            f"hyperparams : {self.hyperparameters}",
            f"excluded    : {self.excluded_vars}",
            f"special_yrs : {self.special_year_vars}",
        ]
        return "\n".join(lines)
