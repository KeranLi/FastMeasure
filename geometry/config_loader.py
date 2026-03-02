from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml


def load_geometry_config(config_path: str = "configs/geometry.yaml") -> Dict[str, Any]:
    """
    Read geometry parameter configuration file geometry_config.yaml
    """
    path = Path(config_path)
    if not path.exists():
        print(f"geometry_config file not found: {config_path}, using empty config")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        print(f"geometry_config loaded successfully: {path}")
        return cfg
    except Exception as e:
        print(f"geometry_config failed to load: {e}")
        return {}
