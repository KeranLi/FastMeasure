from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml


def load_geometry_config(config_path: str = "geometry_config.yaml") -> Dict[str, Any]:
    """
    读取几何参数配置文件 geometry_config.yaml
    """
    path = Path(config_path)
    if not path.exists():
        print(f"geometry_config 文件不存在: {config_path}，将使用空配置")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        print(f"geometry_config 加载成功: {path}")
        return cfg
    except Exception as e:
        print(f"geometry_config 加载失败: {e}")
        return {}
