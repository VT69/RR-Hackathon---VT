"""Configuration loading. Every threshold, label and path comes from ``config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config and return it as a plain dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str) -> Path:
    """Resolve a config path relative to the project root."""
    return (PROJECT_ROOT / relative).resolve()
