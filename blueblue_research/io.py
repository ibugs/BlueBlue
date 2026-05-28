"""订单流 Bar 的读取和 CSV 输出。

阶段二只消费已经通过阶段一校验的 5 分钟订单流 Bar，不重新读取 tick。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import BASE_COLUMNS, Stage2Config


def contract_from_orderflow_filename(path: Path) -> Optional[str]:
    """从 period_of_5_*_SHFE.cu2604_*.csv 中解析合约名。"""

    match = re.search(r"_(SHFE\.cu\d+)_\d+\.csv$", path.name)
    return match.group(1) if match else None


def read_csv(path: Path, parse_datetime: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
    if parse_datetime and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def load_orderflow_bars(config: Stage2Config) -> pd.DataFrame:
    """批量读取阶段一生成并校验通过的 5 分钟订单流 Bar。"""

    frames = []
    for path in sorted(config.input_dir.glob("period_of_5_*_SHFE.cu*.csv")):
        contract = contract_from_orderflow_filename(path)
        if contract is None:
            continue
        if config.contracts is not None and contract not in config.contracts:
            continue

        df = read_csv(path)
        df["contract"] = contract
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M", errors="coerce")
        frames.append(df[[col for col in BASE_COLUMNS if col in df.columns]].copy())

    if not frames:
        raise FileNotFoundError(f"No order-flow CSV files found in {config.input_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["datetime"]).sort_values(["contract", "datetime"]).reset_index(drop=True)
    out = out[out["datetime"] >= config.start_date].reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "poc", "delta", "open_interest"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
