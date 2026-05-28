"""阶段二策略装配。

策略仍然是事件研究级别：训练集确定特征方向、标准化参数和分位阈值，
测试集只做验证，不根据测试结果反向调参。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Stage2Config
from .features import FEATURE_GROUP_MAP, standardize_selected_features


def _strategy_metrics(trades: pd.DataFrame, split_name: str, return_col: str = "net_return") -> Dict[str, Any]:
    if trades.empty:
        return {
            "split": split_name,
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "win_rate": np.nan,
            "avg_net_return": np.nan,
            "median_net_return": np.nan,
            "total_net_return_sum": 0.0,
            "per_trade_sharpe": np.nan,
            "max_drawdown_sum": np.nan,
        }
    returns = pd.to_numeric(trades[return_col], errors="coerce").dropna()
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    sharpe = np.nan
    if len(returns) >= 2 and returns.std(ddof=1) != 0:
        sharpe = float(np.sqrt(len(returns)) * returns.mean() / returns.std(ddof=1))
    return {
        "split": split_name,
        "trades": int(len(trades)),
        "long_trades": int((trades["side"] == "long").sum()),
        "short_trades": int((trades["side"] == "short").sum()),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_net_return": float(returns.mean()) if len(returns) else np.nan,
        "median_net_return": float(returns.median()) if len(returns) else np.nan,
        "total_net_return_sum": float(returns.sum()) if len(returns) else 0.0,
        "per_trade_sharpe": sharpe,
        "max_drawdown_sum": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def add_signal_scores(df: pd.DataFrame, selected_features: pd.DataFrame, transform_params: pd.DataFrame) -> pd.DataFrame:
    """按组内 IC 加权、组间等权生成综合信号。"""

    out = df.copy()
    if selected_features.empty:
        out["signal_score"] = np.nan
        return out

    features = selected_features["feature"].tolist()
    z = standardize_selected_features(out, transform_params, features)
    group_scores = []
    for group, part in selected_features.groupby("group"):
        cols = part["feature"].tolist()
        weights = part["abs_spearman_ic_5"].astype(float)
        weight_sum = float(weights.sum())
        if weight_sum == 0:
            weights = pd.Series([1.0 / len(cols)] * len(cols), index=part.index)
        else:
            weights = weights / weight_sum

        signed_parts = []
        for (_, row), weight in zip(part.iterrows(), weights):
            direction = np.sign(float(row["spearman_ic_5"]))
            signed_parts.append(direction * weight * z[row["feature"]])
        group_col = f"group_score_{group}"
        out[group_col] = pd.concat(signed_parts, axis=1).sum(axis=1, min_count=1)
        group_scores.append(group_col)

    out["signal_score"] = out[group_scores].mean(axis=1)
    return out


def add_risk_filter(scored: pd.DataFrame, config: Stage2Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """训练集拟合固定风控阈值，避免极端Bar直接进入事件研究。"""

    out = scored.copy()
    train_mask = out["datetime"] <= config.train_end
    range_threshold = float(out.loc[train_mask, "true_range_ticks"].quantile(0.995))
    poc_shift_threshold = float(out.loc[train_mask, "poc_shift_ticks"].abs().quantile(0.995))
    out["risk_filter_pass"] = (
        out["signal_score"].notna()
        & (out["volume"] > 0)
        & (out["true_range_ticks"] <= range_threshold)
        & (out["poc_shift_ticks"].abs() <= poc_shift_threshold)
        & out["entry_open_next"].notna()
        & out["exit_close_after_5"].notna()
    )
    return out, {"range_ticks_p995": range_threshold, "abs_poc_shift_ticks_p995": poc_shift_threshold}


def generate_trades(scored: pd.DataFrame, config: Stage2Config, long_threshold: float, short_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for contract, group in scored.groupby("contract", sort=True):
        g = group.sort_values("datetime").reset_index(drop=True)
        next_allowed_i = 0
        for i, row in g.iterrows():
            if i < next_allowed_i or not bool(row.get("risk_filter_pass", False)):
                continue
            score = row["signal_score"]
            side = None
            if score >= long_threshold:
                side = "long"
            elif score <= short_threshold:
                side = "short"
            if side is None:
                continue

            entry_i = i + 1
            exit_i = i + config.holding_bars
            if exit_i >= len(g) or entry_i >= len(g):
                continue
            entry_price = g.loc[entry_i, "open"]
            exit_price = g.loc[exit_i, "close"]
            exit_datetime = g.loc[exit_i, "datetime"]
            # 训练集末尾信号如果持仓跨入测试集，直接跳过，避免边界污染。
            if row["datetime"] <= config.train_end and exit_datetime >= config.test_start:
                continue
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0 or exit_price == 0:
                continue
            # SHFE.cu 是线性期货合约，空头收益按价格绝对变动线性结算。
            gross_return = exit_price / entry_price - 1 if side == "long" else 1 - exit_price / entry_price
            net_return = gross_return - config.roundtrip_cost_price / entry_price
            rows.append(
                {
                    "contract": contract,
                    "side": side,
                    "signal_datetime": row["datetime"],
                    "entry_datetime": g.loc[entry_i, "datetime"],
                    "exit_datetime": exit_datetime,
                    "signal_score": score,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "holding_bars": config.holding_bars,
                    "split": "train" if row["datetime"] <= config.train_end else "test",
                    "long_threshold": long_threshold,
                    "short_threshold": short_threshold,
                }
            )
            next_allowed_i = i + config.holding_bars
    return pd.DataFrame(rows)


def build_strategy(df: pd.DataFrame, selected_features: pd.DataFrame, transform_params: pd.DataFrame, config: Stage2Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    scored = add_signal_scores(df, selected_features, transform_params)
    scored, risk_thresholds = add_risk_filter(scored, config)
    train_scores = scored.loc[(scored["datetime"] <= config.train_end) & scored["risk_filter_pass"], "signal_score"].dropna()
    if train_scores.empty or selected_features.empty:
        trades = pd.DataFrame()
        summary = pd.DataFrame([_strategy_metrics(trades, "train"), _strategy_metrics(trades, "test"), _strategy_metrics(trades, "all")])
        equity = pd.DataFrame(columns=["exit_datetime", "contract", "side", "net_return", "equity_sum"])
        cost_sensitivity = pd.DataFrame()
        return scored, trades, summary, equity, cost_sensitivity, risk_thresholds

    long_threshold = float(train_scores.quantile(config.long_quantile))
    short_threshold = float(train_scores.quantile(config.short_quantile))
    trades = generate_trades(scored, config, long_threshold, short_threshold)
    if not trades.empty:
        trades["selected_features"] = ",".join(selected_features["feature"].tolist())

    train_trades = trades[trades["split"] == "train"] if not trades.empty else trades
    test_trades = trades[trades["split"] == "test"] if not trades.empty else trades
    summary = pd.DataFrame([_strategy_metrics(train_trades, "train"), _strategy_metrics(test_trades, "test"), _strategy_metrics(trades, "all")])
    if not trades.empty:
        equity = trades.sort_values("exit_datetime")[["exit_datetime", "contract", "side", "net_return"]].copy()
        equity["equity_sum"] = equity["net_return"].cumsum()
    else:
        equity = pd.DataFrame(columns=["exit_datetime", "contract", "side", "net_return", "equity_sum"])

    sensitivity_rows = []
    for cost_ticks in (1.0, 2.0, 3.0, 4.0):
        if trades.empty:
            continue
        temp = trades.copy()
        temp["net_return_cost_case"] = temp["gross_return"] - (cost_ticks * config.tick_size) / temp["entry_price"]
        for split in ("train", "test", "all"):
            part = temp if split == "all" else temp[temp["split"] == split]
            row = _strategy_metrics(part, split, return_col="net_return_cost_case")
            row["cost_ticks_roundtrip"] = cost_ticks
            sensitivity_rows.append(row)
    cost_sensitivity = pd.DataFrame(sensitivity_rows)
    return scored, trades, summary, equity, cost_sensitivity, risk_thresholds


def check_no_overlap(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame(columns=["contract", "overlap_count"])
    for contract, group in trades.groupby("contract"):
        g = group.sort_values("entry_datetime")
        overlaps = (g["entry_datetime"].shift(-1) < g["exit_datetime"]).sum()
        rows.append({"contract": contract, "overlap_count": int(overlaps)})
    return pd.DataFrame(rows)
