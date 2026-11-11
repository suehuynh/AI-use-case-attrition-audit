from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """Typed view over config.yaml. All parameters live in the YAML file."""

    raw: dict[str, Any]
    root: Path

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        path = Path(path).resolve()
        with open(path) as f:
            return cls(raw=yaml.safe_load(f), root=path.parent)

    @property
    def raw_dir(self) -> Path:
        return self.root / self.raw["paths"]["raw_dir"]

    @property
    def output_dir(self) -> Path:
        out = self.root / self.raw["paths"]["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        return out

    def inventory(self, year: str) -> dict[str, Any]:
        return self.raw["inventories"][year]

    @property
    def filters(self) -> dict[str, bool]:
        return self.raw["filters"]

    @property
    def baseline(self) -> dict[str, Any]:
        return self.raw["baseline"]

    @property
    def targets(self) -> dict[str, dict[str, int]]:
        return self.raw["targets"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def dissolved_agencies(self) -> list[str]:
        return self.raw.get("known_dissolved_agencies", [])