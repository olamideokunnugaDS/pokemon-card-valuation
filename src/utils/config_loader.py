"""Small helper for loading the project's YAML config files."""

from pathlib import Path
from typing import Union

import yaml


def load_config(path: Union[str, Path]) -> dict:
    """Load a single YAML config file into a dict."""
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_configs(*paths: Union[str, Path]) -> dict:
    """Load and shallow-merge multiple YAML config files.

    Later paths take precedence on key conflicts. Useful for combining a
    base data config with a module-specific config (e.g. vision + data).
    """
    merged: dict = {}
    for p in paths:
        merged.update(load_config(p))
    return merged
