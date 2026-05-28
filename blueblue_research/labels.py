"""阶段二标签生成。

标签只在单合约内部向未来取值。策略假设是当前 Bar 收盘后形成信号，
下一根 Bar 开盘入场，持有固定 Bar 数后按收盘退出。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage2Config


def forward_max(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def forward_min(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def add_labels_for_contract(group: pd.DataFrame, config: Stage2Config) -> pd.DataFrame:
    g = group.copy()
    for horizon in (1, 3, 5, 10):
        g[f"future_return_{horizon}"] = g["close"].shift(-horizon) / g["close"].replace(0, np.nan) - 1

    future_high_10 = forward_max(g["high"], 10)
    future_low_10 = forward_min(g["low"], 10)
    g["mfe_10_long"] = future_high_10 / g["close"].replace(0, np.nan) - 1
    g["mae_10_long"] = future_low_10 / g["close"].replace(0, np.nan) - 1

    holding = config.holding_bars
    g["entry_open_next"] = g["open"].shift(-1)
    g["exit_close_after_5"] = g["close"].shift(-holding)
    g["trade_return_5_gross"] = g["exit_close_after_5"] / g["entry_open_next"].replace(0, np.nan) - 1
    g["trade_return_5_net"] = g["trade_return_5_gross"] - config.roundtrip_cost_price / g["entry_open_next"].replace(0, np.nan)
    return g


def build_labels(features: pd.DataFrame, config: Stage2Config) -> pd.DataFrame:
    frames = [add_labels_for_contract(group, config) for _, group in features.groupby("contract", sort=True)]
    return pd.concat(frames, ignore_index=True)
