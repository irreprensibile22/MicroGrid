"""Lettura della configurazione del progetto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Legge il file YAML di configurazione (di default `config/default.yaml`)."""
    with Path(path or DEFAULT_CONFIG_PATH).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Trasforma un percorso relativo della configurazione in percorso assoluto."""
    return PROJECT_ROOT / relative
