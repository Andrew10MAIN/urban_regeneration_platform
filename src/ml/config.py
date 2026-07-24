"""
ModelConfig — dataclass wrapping the YAML run configuration.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    model_id: str
    model_type: str
    target_variable: str
    pre_period: int
    post_period: int
    pre_period_deviation: int
    confounders: list[str]
    hyperparameters: dict[str, Any]
    random_seed: int

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def override(self, **kwargs) -> "ModelConfig":
        """Return a copy with specific fields overridden (for CLI overrides)."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)

    def summary(self) -> str:
        lines = [
            f"  model_id        : {self.model_id}",
            f"  model_type      : {self.model_type}",
            f"  target          : {self.target_variable}",
            f"  pre_period      : {self.pre_period}  (±{self.pre_period_deviation} yr tolerance)",
            f"  post_period     : {self.post_period}",
            f"  confounders     : {self.confounders}",
            f"  hyperparameters : {self.hyperparameters}",
            f"  random_seed     : {self.random_seed}",
        ]
        return "\n".join(lines)
