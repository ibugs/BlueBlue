#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 MVP research pipeline for validated 5-minute order-flow bars.

The pipeline intentionally stays simple:
validated bars -> features -> labels -> single-feature tests -> tiny strategy -> report.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("/Users/wangrendong/Projects/BlueBlue/orderflow_data/SHFE.cu")
DEFAULT_OUTPUT_DIR = Path("/Users/wangrendong/Projects/BlueBlue/stage1_outputs/SHFE.cu")
DEFAULT_START_DATE = "2023-05-27"
DEFAULT_TRAIN_END = "2025-05-26 23:59:59"
DEFAULT_TEST_START = "2025-05-27"

BASE_COLUMNS = [
    "contract",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "poc",
    "delta",
    "open_interest",
]

FEATURE_COLUMNS = [
    "bar_return",
    "range_ticks",
    "body_ratio",
    "close_position",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "delta_strength",
    "delta_zscore_20",
    "volume_zscore_20",
    "cvd_change_10",
    "poc_distance_ticks",
    "poc_shift_ticks",
    "open_interest_change",
    "open_interest_zscore_20",
    "trend_return_12",
    "volatility_20",
    "price_delta_agreement",
]

LABEL_COLUMNS = [
    "future_return_1",
    "future_return_3",
    "future_return_5",
    "future_return_10",
    "mfe_10_long",
    "mae_10_long",
    "entry_open_next",
    "exit_close_after_5",
    "trade_return_5_gross",
    "trade_return_5_net",
]


@dataclass
class RunConfig:
    input_dir: Path
    output_dir: Path
    start_date: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    primary_horizon: int
    holding_bars: int
    cost_ticks_roundtrip: float
    tick_size: float
    contracts: Optional[List[str]]

    @property
    def roundtrip_cost_price(self) -> float:
        return self.cost_ticks_roundtrip * self.tick_size


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Stage 1 MVP order-flow feature and strategy research pipeline")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--train_end", type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--test_start", type=str, default=DEFAULT_TEST_START)
    parser.add_argument("--primary_horizon", type=int, default=5)
    parser.add_argument("--holding_bars", type=int, default=5)
    parser.add_argument("--cost_ticks_roundtrip", type=float, default=2.0)
    parser.add_argument("--tick_size", type=float, default=10.0)
    parser.add_argument("--contracts", type=str, default=None, help="Comma-separated contracts, e.g. SHFE.cu2604 or cu2604")
    args = parser.parse_args()

    contracts = None
    if args.contracts:
        contracts = [normalize_contract(c.strip()) for c in args.contracts.split(",") if c.strip()]

    return RunConfig(
        input_dir=Path(args.input_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        start_date=pd.Timestamp(args.start_date),
        train_end=pd.Timestamp(args.train_end),
        test_start=pd.Timestamp(args.test_start),
        primary_horizon=args.primary_horizon,
        holding_bars=args.holding_bars,
        cost_ticks_roundtrip=args.cost_ticks_roundtrip,
        tick_size=args.tick_size,
        contracts=contracts,
    )


def normalize_contract(contract: str) -> str:
    contract = contract.strip()
    return contract if contract.startswith("SHFE.") else f"SHFE.{contract}"


def contract_from_filename(path: Path) -> Optional[str]:
    match = re.search(r"_(SHFE\.cu\d+)_\d+\.csv$", path.name)
    return match.group(1) if match else None


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill: Optional[float] = np.nan) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.fillna(fill) if fill is not None else out


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def forward_max(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def forward_min(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def load_orderflow_bars(config: RunConfig) -> pd.DataFrame:
    frames = []
    for path in sorted(config.input_dir.glob("period_of_5_*_SHFE.cu*.csv")):
        contract = contract_from_filename(path)
        if contract is None:
            continue
        if config.contracts is not None and contract not in config.contracts:
            continue

        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
        df["contract"] = contract
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M", errors="coerce")
        keep = [col for col in BASE_COLUMNS if col in df.columns]
        df = df[keep].copy()
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No order-flow CSV files found in {config.input_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["datetime"]).sort_values(["contract", "datetime"]).reset_index(drop=True)
    out = out[out["datetime"] >= config.start_date].reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "poc", "delta", "open_interest"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_features_for_contract(group: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    g = group.copy()
    price_range = g["high"] - g["low"]
    body = (g["close"] - g["open"]).abs()
    upper_shadow = g["high"] - np.maximum(g["open"], g["close"])
    lower_shadow = np.minimum(g["open"], g["close"]) - g["low"]
    bar_direction = np.sign(g["close"] - g["open"])
    delta_direction = np.sign(g["delta"])
    oi_change = g["open_interest"].diff()
    bar_return = g["close"].pct_change()
    cvd = g["delta"].cumsum()

    g["bar_return"] = g["close"] / g["open"].replace(0, np.nan) - 1
    g["range_ticks"] = price_range / tick_size
    g["body_ratio"] = safe_divide(body, price_range, fill=0.0)
    g["close_position"] = safe_divide(g["close"] - g["low"], price_range, fill=0.5)
    g["upper_shadow_ratio"] = safe_divide(upper_shadow, price_range, fill=0.0)
    g["lower_shadow_ratio"] = safe_divide(lower_shadow, price_range, fill=0.0)
    g["delta_strength"] = safe_divide(g["delta"], g["volume"], fill=0.0)
    g["delta_zscore_20"] = rolling_zscore(g["delta"], 20)
    g["volume_zscore_20"] = rolling_zscore(g["volume"], 20)
    g["cvd_change_10"] = cvd - cvd.shift(10)
    g["poc_distance_ticks"] = (g["close"] - g["poc"]) / tick_size
    g["poc_shift_ticks"] = (g["poc"] - g["poc"].shift(1)) / tick_size
    g["open_interest_change"] = oi_change
    g["open_interest_zscore_20"] = rolling_zscore(oi_change, 20)
    g["trend_return_12"] = g["close"] / g["close"].shift(12).replace(0, np.nan) - 1
    g["volatility_20"] = bar_return.rolling(20, min_periods=20).std(ddof=0)
    g["price_delta_agreement"] = bar_direction * delta_direction
    return g


def build_features(bars: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    frames = [add_features_for_contract(group, tick_size) for _, group in bars.groupby("contract", sort=True)]
    return pd.concat(frames, ignore_index=True)


def add_labels_for_contract(group: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
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


def build_labels(features: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    frames = [add_labels_for_contract(group, config) for _, group in features.groupby("contract", sort=True)]
    return pd.concat(frames, ignore_index=True)


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="pearson")) if len(x) >= 20 else np.nan


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="spearman")) if len(x) >= 20 else np.nan


def summarize_one_feature(df: pd.DataFrame, feature: str, label_col: str, train_mask: pd.Series, test_mask: pd.Series) -> Dict[str, Any]:
    valid = df[[feature, label_col, "mfe_10_long", "mae_10_long"]].dropna(subset=[feature, label_col])
    train = df.loc[train_mask, [feature, label_col]].dropna()
    test = df.loc[test_mask, [feature, label_col]].dropna()
    full = df[[feature, label_col]].dropna()

    result: Dict[str, Any] = {
        "feature": feature,
        "non_null_count": int(df[feature].notna().sum()),
        "valid_count": int(len(valid)),
        "coverage": float(len(valid) / len(df)) if len(df) else np.nan,
        "pearson_ic_5": pearson_corr(train[feature], train[label_col]) if len(train) else np.nan,
        "spearman_ic_5": spearman_corr(train[feature], train[label_col]) if len(train) else np.nan,
        "full_pearson_ic_5": pearson_corr(full[feature], full[label_col]) if len(full) else np.nan,
        "full_spearman_ic_5": spearman_corr(full[feature], full[label_col]) if len(full) else np.nan,
        "test_spearman_ic_5": spearman_corr(test[feature], test[label_col]) if len(test) else np.nan,
    }
    result["ic_direction"] = int(np.sign(result["spearman_ic_5"])) if pd.notna(result["spearman_ic_5"]) else 0

    quintile_stats = feature_quintile_stats(df, feature, label_col)
    if not quintile_stats.empty and set(["Q1", "Q5"]).issubset(set(quintile_stats["quintile"])):
        q1 = quintile_stats.loc[quintile_stats["quintile"] == "Q1"].iloc[0]
        q5 = quintile_stats.loc[quintile_stats["quintile"] == "Q5"].iloc[0]
        result["quintile_spread_mean_return"] = float(q5["mean_return"] - q1["mean_return"])
        result["top_quintile_win_rate"] = float(q5["win_rate"])
        result["top_quintile_mean_mfe"] = float(q5["mean_mfe_10_long"])
        result["top_quintile_mean_mae"] = float(q5["mean_mae_10_long"])
    else:
        result["quintile_spread_mean_return"] = np.nan
        result["top_quintile_win_rate"] = np.nan
        result["top_quintile_mean_mfe"] = np.nan
        result["top_quintile_mean_mae"] = np.nan

    return result


def feature_quintile_stats(df: pd.DataFrame, feature: str, label_col: str) -> pd.DataFrame:
    valid = df[[feature, label_col, "mfe_10_long", "mae_10_long"]].dropna(subset=[feature, label_col]).copy()
    if valid[feature].nunique(dropna=True) < 5:
        return pd.DataFrame()
    try:
        valid["quintile_num"] = pd.qcut(valid[feature], 5, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.DataFrame()

    rows = []
    for q, part in valid.groupby("quintile_num"):
        rows.append(
            {
                "feature": feature,
                "quintile": f"Q{int(q)}",
                "count": int(len(part)),
                "mean_feature": float(part[feature].mean()),
                "mean_return": float(part[label_col].mean()),
                "median_return": float(part[label_col].median()),
                "win_rate": float((part[label_col] > 0).mean()),
                "mean_mfe_10_long": float(part["mfe_10_long"].mean()),
                "mean_mae_10_long": float(part["mae_10_long"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_feature_tests(df: pd.DataFrame, config: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_col = f"future_return_{config.primary_horizon}"
    train_mask = df["datetime"] <= config.train_end
    test_mask = df["datetime"] >= config.test_start

    summary_rows = [summarize_one_feature(df, feature, label_col, train_mask, test_mask) for feature in FEATURE_COLUMNS]
    feature_summary = pd.DataFrame(summary_rows).sort_values("spearman_ic_5", key=lambda s: s.abs(), ascending=False)

    quintile_frames = [feature_quintile_stats(df, feature, label_col) for feature in FEATURE_COLUMNS]
    feature_quintiles = pd.concat([x for x in quintile_frames if not x.empty], ignore_index=True) if any(not x.empty for x in quintile_frames) else pd.DataFrame()

    monthly_rows = []
    working = df.copy()
    working["month"] = working["datetime"].dt.to_period("M").astype(str)
    for month, part in working.groupby("month"):
        for feature_name in FEATURE_COLUMNS:
            valid = part[[feature_name, label_col]].dropna()
            if len(valid) >= 20:
                monthly_rows.append(
                    {
                        "month": month,
                        "feature": feature_name,
                        "count": int(len(valid)),
                        "pearson_ic_5": pearson_corr(valid[feature_name], valid[label_col]),
                        "spearman_ic_5": spearman_corr(valid[feature_name], valid[label_col]),
                    }
                )
    monthly_ic = pd.DataFrame(monthly_rows)
    return feature_summary, feature_quintiles, monthly_ic


def choose_strategy_features(feature_summary: pd.DataFrame) -> pd.DataFrame:
    candidates = feature_summary.copy()
    candidates = candidates[candidates["spearman_ic_5"].notna()]
    candidates = candidates[candidates["coverage"] >= 0.5]
    candidates = candidates[candidates["spearman_ic_5"].abs() > 0]
    return candidates.reindex(candidates["spearman_ic_5"].abs().sort_values(ascending=False).index).head(5)


def add_signal_score(df: pd.DataFrame, selected_features: pd.DataFrame, config: RunConfig) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    out = df.copy()
    train_mask = out["datetime"] <= config.train_end
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    score_parts = []

    for _, row in selected_features.iterrows():
        feature = row["feature"]
        direction = np.sign(row["spearman_ic_5"])
        train_values = out.loc[train_mask, feature].dropna()
        if train_values.empty:
            continue
        mean = float(train_values.mean())
        std = float(train_values.std(ddof=0))
        if not math.isfinite(std) or std == 0:
            continue
        means[feature] = mean
        stds[feature] = std
        score_parts.append(direction * ((out[feature] - mean) / std))

    if score_parts:
        out["signal_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
    else:
        out["signal_score"] = np.nan
    return out, means, stds


def generate_trades(scored: pd.DataFrame, config: RunConfig, long_threshold: float, short_threshold: float) -> pd.DataFrame:
    rows = []
    cost = config.roundtrip_cost_price
    for contract, group in scored.groupby("contract", sort=True):
        g = group.sort_values("datetime").reset_index(drop=True)
        next_allowed_i = 0
        for i, row in g.iterrows():
            if i < next_allowed_i:
                continue
            score = row.get("signal_score")
            if pd.isna(score):
                continue
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
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                continue

            if side == "long":
                gross_return = exit_price / entry_price - 1
            else:
                gross_return = entry_price / exit_price - 1 if exit_price != 0 else np.nan
            net_return = gross_return - cost / entry_price

            rows.append(
                {
                    "contract": contract,
                    "side": side,
                    "signal_datetime": row["datetime"],
                    "entry_datetime": g.loc[entry_i, "datetime"],
                    "exit_datetime": g.loc[exit_i, "datetime"],
                    "signal_score": score,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "holding_bars": config.holding_bars,
                    "split": "train" if row["datetime"] <= config.train_end else "test",
                }
            )
            next_allowed_i = i + config.holding_bars
    return pd.DataFrame(rows)


def strategy_metrics(trades: pd.DataFrame, split_name: str) -> Dict[str, Any]:
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

    returns = trades["net_return"].dropna()
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


def build_strategy(df: pd.DataFrame, feature_summary: pd.DataFrame, config: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_features = choose_strategy_features(feature_summary)
    scored, _, _ = add_signal_score(df, selected_features, config)

    train_scores = scored.loc[scored["datetime"] <= config.train_end, "signal_score"].dropna()
    if train_scores.empty or selected_features.empty:
        trades = pd.DataFrame()
        summary = pd.DataFrame([strategy_metrics(trades, "train"), strategy_metrics(trades, "test"), strategy_metrics(trades, "all")])
        equity = pd.DataFrame(columns=["exit_datetime", "contract", "side", "net_return", "equity_sum"])
        return selected_features, trades, summary, equity

    long_threshold = float(train_scores.quantile(0.8))
    short_threshold = float(train_scores.quantile(0.2))
    trades = generate_trades(scored, config, long_threshold, short_threshold)
    if not trades.empty:
        trades["long_threshold"] = long_threshold
        trades["short_threshold"] = short_threshold
        trades["selected_features"] = ",".join(selected_features["feature"].tolist())

    train_trades = trades[trades["split"] == "train"] if not trades.empty else trades
    test_trades = trades[trades["split"] == "test"] if not trades.empty else trades
    summary = pd.DataFrame(
        [
            strategy_metrics(train_trades, "train"),
            strategy_metrics(test_trades, "test"),
            strategy_metrics(trades, "all"),
        ]
    )
    if not trades.empty:
        equity = trades.sort_values("exit_datetime")[["exit_datetime", "contract", "side", "net_return"]].copy()
        equity["equity_sum"] = equity["net_return"].cumsum()
    else:
        equity = pd.DataFrame(columns=["exit_datetime", "contract", "side", "net_return", "equity_sum"])
    return selected_features, trades, summary, equity


def check_no_overlap(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame(columns=["contract", "overlap_count"])
    for contract, group in trades.groupby("contract"):
        g = group.sort_values("entry_datetime")
        overlaps = (g["entry_datetime"].shift(-1) < g["exit_datetime"]).sum()
        rows.append({"contract": contract, "overlap_count": int(overlaps)})
    return pd.DataFrame(rows)


def write_markdown_report(
    config: RunConfig,
    bars: pd.DataFrame,
    feature_summary: pd.DataFrame,
    selected_features: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: Path,
) -> None:
    top = feature_summary.head(8)
    bottom = feature_summary.sort_values("spearman_ic_5", ascending=True).head(8)
    selected = selected_features[["feature", "spearman_ic_5", "coverage"]].copy() if not selected_features.empty else pd.DataFrame()

    lines = [
        "# Stage 1 MVP Report",
        "",
        "## Data Scope",
        f"- input_dir: `{config.input_dir}`",
        f"- start_date: `{config.start_date.date()}`",
        f"- contracts: `{bars['contract'].nunique()}`",
        f"- rows: `{len(bars)}`",
        f"- datetime_min: `{bars['datetime'].min()}`",
        f"- datetime_max: `{bars['datetime'].max()}`",
        "",
        "## Feature List",
        ", ".join(f"`{x}`" for x in FEATURE_COLUMNS),
        "",
        "## Label Definition",
        "- primary label: `future_return_5`",
        "- strategy label: next-bar open entry, 5-bar close exit, 2-tick roundtrip cost.",
        "",
        "## Top Features By Train Spearman IC",
        top.to_markdown(index=False),
        "",
        "## Most Negative Features By Train Spearman IC",
        bottom.to_markdown(index=False),
        "",
        "## Selected MVP Strategy Features",
        selected.to_markdown(index=False) if not selected.empty else "No feature selected. Check train sample or feature coverage.",
        "",
        "## MVP Strategy Summary",
        strategy_summary.to_markdown(index=False),
        "",
        "## First-Stage Findings",
    ]

    if selected_features.empty:
        lines.append("- The selected training universe did not produce usable features for strategy assembly.")
    else:
        lines.append("- The MVP now produces a repeatable feature-ranking and strategy-feedback loop.")
        lines.append("- Feature signs are chosen only from the training set to avoid test leakage.")
        lines.append("- Strategy performance should be treated as event-study feedback, not a production trading result.")

    if trades.empty:
        lines.append("- No trades were generated under the 80/20 threshold rule.")
    else:
        test_row = strategy_summary[strategy_summary["split"] == "test"].iloc[0].to_dict()
        lines.append(f"- Test trades: `{int(test_row['trades'])}`; test avg net return: `{test_row['avg_net_return']}`.")

    lines.extend(
        [
            "",
            "## Stage 2 Fix List",
            "- Replace the simple 2-tick cost proxy with exchange fee, slippage, and realistic execution assumptions.",
            "- Add market-regime segmentation: trend/range, high/low volatility, high/low volume.",
            "- Add walk-forward feature selection instead of one fixed train/test split.",
            "- Add richer order-flow features after the current 17-feature MVP is reviewed.",
            "- Add charts for IC stability, quintile monotonicity, and strategy equity.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    config: RunConfig,
    df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    feature_quintiles: pd.DataFrame,
    monthly_ic: pd.DataFrame,
    selected_features: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    equity_curve: pd.DataFrame,
    overlap_check: pd.DataFrame,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    df[BASE_COLUMNS + FEATURE_COLUMNS].to_csv(config.output_dir / "features.csv", index=False, encoding="utf-8-sig")
    df[BASE_COLUMNS + LABEL_COLUMNS].to_csv(config.output_dir / "labels.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(config.output_dir / "feature_summary.csv", index=False, encoding="utf-8-sig")
    feature_quintiles.to_csv(config.output_dir / "feature_quintiles.csv", index=False, encoding="utf-8-sig")
    monthly_ic.to_csv(config.output_dir / "monthly_ic.csv", index=False, encoding="utf-8-sig")
    selected_features.to_csv(config.output_dir / "selected_features.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(config.output_dir / "trades_mvp.csv", index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(config.output_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    equity_curve.to_csv(config.output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    overlap_check.to_csv(config.output_dir / "trade_overlap_check.csv", index=False, encoding="utf-8-sig")
    write_markdown_report(
        config=config,
        bars=df,
        feature_summary=feature_summary,
        selected_features=selected_features,
        strategy_summary=strategy_summary,
        trades=trades,
        output_path=config.output_dir / "stage1_mvp_report.md",
    )


def run(config: RunConfig) -> None:
    bars = load_orderflow_bars(config)
    features = build_features(bars, config.tick_size)
    labeled = build_labels(features, config)
    feature_summary, feature_quintiles, monthly_ic = build_feature_tests(labeled, config)
    selected_features, trades, strategy_summary, equity_curve = build_strategy(labeled, feature_summary, config)
    overlap_check = check_no_overlap(trades)
    write_outputs(
        config=config,
        df=labeled,
        feature_summary=feature_summary,
        feature_quintiles=feature_quintiles,
        monthly_ic=monthly_ic,
        selected_features=selected_features,
        trades=trades,
        strategy_summary=strategy_summary,
        equity_curve=equity_curve,
        overlap_check=overlap_check,
    )

    print(f"output_dir={config.output_dir}")
    print(f"contracts={labeled['contract'].nunique()}, rows={len(labeled)}")
    print(f"features={len(FEATURE_COLUMNS)}, selected_features={len(selected_features)}, trades={len(trades)}")


def main() -> int:
    config = parse_args()
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
