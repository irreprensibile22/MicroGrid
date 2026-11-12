"""Project configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML configuration file (`config/default.yaml` by default)."""
    with Path(path or DEFAULT_CONFIG_PATH).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Turn a path from the configuration into an absolute one, anchored at the project root."""
    return PROJECT_ROOT / relative
