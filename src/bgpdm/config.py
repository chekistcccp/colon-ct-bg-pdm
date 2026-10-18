from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    for section in ("data", "features", "experiment", "memory", "evaluation", "runtime"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
    return cfg


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
