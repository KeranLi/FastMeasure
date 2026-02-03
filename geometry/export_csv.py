from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional


def select_columns_for_grain_statistics_csv(
    grain_data: pd.DataFrame,
    geometry_config: Dict[str, Any],
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """
    根据 geometry_config.yaml 控制 grain_statistics.csv 输出列
    """
    if grain_data is None or grain_data.empty:
        return grain_data

    csv_cfg = geometry_config.get("grain_statistics_csv", {})
    enabled = csv_cfg.get("enabled", True)
    if not enabled:
        return grain_data

    keep_cols: Optional[List[str]] = csv_cfg.get("keep_columns", None)
    drop_cols: List[str] = csv_cfg.get("drop_columns", []) or []

    df = grain_data.copy()

    # keep_columns：只保留存在的列
    if keep_cols:
        exist = [c for c in keep_cols if c in df.columns]
        missing = [c for c in keep_cols if c not in df.columns]

        if missing:
            msg = f"geometry_config.keep_columns 中这些列不存在，已忽略: {missing}"
            if strict:
                raise KeyError(msg)
            print(msg)

        df = df[exist]

    # drop_columns：删掉存在的列
    if drop_cols:
        drop_exist = [c for c in drop_cols if c in df.columns]
        if drop_exist:
            df = df.drop(columns=drop_exist)

    return df
