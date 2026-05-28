"""阶段二趋势回调做多专项实验。

本模块把“趋势向上、回踩不破、K线转强、量价确认、空间足够”的主观
交易 setup 转成可重复审计的规则门控实验。所有特征只使用当前 Bar 和
历史 Bar；交易收益才使用下一根开盘和未来退出价。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BASE_COLUMNS, DEFAULT_OUTPUT_DIR, LABEL_COLUMNS, Stage2Config, parse_contracts, resolve_time_split
from .io import read_csv, write_csv
from .strategy import check_no_overlap


DEFAULT_STAGE2_DIR = DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_SUBDIR = "trend_pullback_gain"
MIN_TEST_TRADES = 100
LOW_CONFIDENCE_TRADES = 300

TREND_PULLBACK_FEATURE_SPECS: List[Dict[str, str]] = [
    {"feature": "ema60", "group": "trend_filter", "description": "60根5分钟Bar指数均线", "formula_note": "close.ewm(span=60, min_periods=60)"},
    {"feature": "ema60_slope_12_ticks", "group": "trend_filter", "description": "EMA60过去12根Bar的每Bar斜率tick数", "formula_note": "(ema60-ema60.shift(12))/tick_size/12"},
    {"feature": "trendline_60", "group": "trend_filter", "description": "最近60根close线性回归趋势线当前值", "formula_note": "rolling linear regression fitted value at current bar"},
    {"feature": "trendline_slope_60_ticks", "group": "trend_filter", "description": "最近60根趋势线每Bar斜率tick数", "formula_note": "rolling regression slope/tick_size"},
    {"feature": "trend_filter_pass", "group": "trend_filter", "description": "趋势过滤通过", "formula_note": "ema60_slope_12_ticks>0 and trend_return_48>0 and trend_strength_48>0"},
    {"feature": "atr20_ticks", "group": "pullback_structure", "description": "20期平均真实波幅tick数，用作回踩自适应尺度", "formula_note": "rolling_mean(true_range_ticks, 20)"},
    {"feature": "adaptive_pullback_tolerance_ticks", "group": "pullback_structure", "description": "波动率自适应回踩容差tick数", "formula_note": "0.5 * atr20_ticks"},
    {"feature": "distance_to_ema60_ticks", "group": "pullback_structure", "description": "收盘价距离EMA60的tick数", "formula_note": "(close-ema60)/tick_size"},
    {"feature": "low_distance_to_ema60_ticks", "group": "pullback_structure", "description": "最低价距离EMA60的tick数", "formula_note": "(low-ema60)/tick_size"},
    {"feature": "distance_to_trendline_ticks", "group": "pullback_structure", "description": "收盘价距离60期趋势线的tick数", "formula_note": "(close-trendline_60)/tick_size"},
    {"feature": "low_distance_to_trendline_ticks", "group": "pullback_structure", "description": "最低价距离60期趋势线的tick数", "formula_note": "(low-trendline_60)/tick_size"},
    {"feature": "ema60_hold_flag", "group": "pullback_structure", "description": "回踩EMA60附近但收盘重新站稳", "formula_note": "low within ema60 +/- 0.5*ATR20 and close>=ema60"},
    {"feature": "trendline_hold_flag", "group": "pullback_structure", "description": "回踩上升趋势线附近但收盘重新站稳", "formula_note": "low within trendline +/- 0.5*ATR20 and close>=trendline"},
    {"feature": "pullback_structure_pass", "group": "pullback_structure", "description": "回踩结构通过", "formula_note": "ema60_hold_flag or trendline_hold_flag"},
    {"feature": "bullish_bar_flag", "group": "kline_turn", "description": "当前Bar收阳", "formula_note": "close>open"},
    {"feature": "strong_bullish_close_flag", "group": "kline_turn", "description": "阳线且收在Bar上半部偏强区域", "formula_note": "bullish_bar_flag and close_location_value>=0.4"},
    {"feature": "body_quality_flag", "group": "kline_turn", "description": "实体大小适中", "formula_note": "0.20<=body_ratio<=0.80"},
    {"feature": "bullish_engulfing_flag", "group": "kline_turn", "description": "阳包阴近似形态", "formula_note": "current bullish body engulfs previous bearish body"},
    {"feature": "morning_star_proxy", "group": "kline_turn", "description": "希望之星近似形态", "formula_note": "weak bar, small/indecision bar, then bullish close above first body midpoint"},
    {"feature": "weak_to_strong_3bar_flag", "group": "kline_turn", "description": "最近2-3根K线由弱转强", "formula_note": "recent weak close then current strong bullish close"},
    {"feature": "kline_turn_pass", "group": "kline_turn", "description": "K线转强通过", "formula_note": "strong_bullish_close and body_quality and one reversal pattern"},
    {"feature": "delta_confirm_long", "group": "flow_confirm", "description": "做多Delta确认", "formula_note": "delta>0 and delta_strength>0"},
    {"feature": "volume_normal_or_better_flag", "group": "flow_confirm", "description": "成交量不明显低迷", "formula_note": "volume_zscore_20>=-0.5"},
    {"feature": "volume_expand_flag", "group": "flow_confirm", "description": "成交量正常偏放大", "formula_note": "volume_zscore_20>=0 or volume_burst_flag==1"},
    {"feature": "flow_confirm_pass", "group": "flow_confirm", "description": "量价/订单流确认通过", "formula_note": "delta_confirm_long and delta_price_agreement>=0 and volume_normal_or_better_flag"},
    {"feature": "pullback_low_10", "group": "space_filter", "description": "最近10根回调低点", "formula_note": "rolling_min(low,10)"},
    {"feature": "previous_swing_high_20", "group": "space_filter", "description": "当前Bar之前20根局部高点", "formula_note": "rolling_max(high.shift(1),20)"},
    {"feature": "stop_price_signal", "group": "space_filter", "description": "信号Bar视角下的止损价", "formula_note": "pullback_low_10 - tick_size"},
    {"feature": "stop_distance_ticks_signal", "group": "space_filter", "description": "信号收盘价到止损价的tick距离", "formula_note": "(close-stop_price_signal)/tick_size"},
    {"feature": "reward_ticks_signal", "group": "space_filter", "description": "信号收盘价到前高的tick空间", "formula_note": "(previous_swing_high_20-close)/tick_size"},
    {"feature": "reward_to_risk_proxy", "group": "space_filter", "description": "信号Bar视角下的潜在盈亏比", "formula_note": "reward_ticks_signal/stop_distance_ticks_signal"},
    {"feature": "space_filter_pass", "group": "space_filter", "description": "空间过滤通过", "formula_note": "previous_swing_high_20>close and 2<=stop_distance<=30 and reward_to_risk>=1.5"},
]

TREND_PULLBACK_FEATURE_COLUMNS = [item["feature"] for item in TREND_PULLBACK_FEATURE_SPECS]

COMBO_DEFINITIONS: List[Dict[str, Any]] = [
    {"combo_name": "trend_only", "required_filters": ["trend_filter_pass"], "description": "只要求趋势向上"},
    {"combo_name": "trend_pullback", "required_filters": ["trend_filter_pass", "pullback_structure_pass"], "description": "趋势向上并回踩支撑不破"},
    {"combo_name": "trend_pullback_kline", "required_filters": ["trend_filter_pass", "pullback_structure_pass", "kline_turn_pass"], "description": "增加K线由弱转强"},
    {"combo_name": "trend_pullback_kline_flow", "required_filters": ["trend_filter_pass", "pullback_structure_pass", "kline_turn_pass", "flow_confirm_pass"], "description": "增加Delta和成交量确认"},
    {"combo_name": "trend_pullback_full_space", "required_filters": ["trend_filter_pass", "pullback_structure_pass", "kline_turn_pass", "flow_confirm_pass", "space_filter_pass"], "description": "增加止损空间和盈亏比过滤"},
]

COMBO_LABELS = {
    "trend_only": "趋势过滤",
    "trend_pullback": "趋势+回踩",
    "trend_pullback_kline": "趋势+回踩+K线转强",
    "trend_pullback_kline_flow": "趋势+回踩+K线转强+量价确认",
    "trend_pullback_full_space": "趋势+回踩+K线转强+量价确认+空间过滤",
}

VALUE_LABELS = {
    "train": "训练集",
    "test": "测试集",
    "all": "全部",
    "enough_samples": "样本充足",
    "low_confidence": "低置信",
    "insufficient_samples": "样本不足",
    "no_samples": "无样本",
    "absolute_effective": "绝对有效",
    "incremental_effective": "增量有效",
    "quality_improved_not_scalable": "质量改善但不可扩展",
    "win_rate_only": "胜率伪增益",
    "cost_failed": "成本失败",
    "unstable": "月度不稳定",
    "ineffective": "无效",
    "PASS": "通过",
    "FAIL": "失败",
    "WARN": "警告",
    "BLOCKER": "阻断",
    "WARNING": "警告",
    "OBSERVATION": "观察",
    True: "是",
    False: "否",
}

COLUMN_LABELS = {
    "combo_name": "组合名",
    "description": "说明",
    "required_filters": "过滤条件",
    "split": "切分",
    "rows": "样本行数",
    "trade_ready_rows": "可交易行数",
    "passed_rows": "通过行数",
    "previous_passed_rows": "上一层通过行数",
    "pass_rate_total": "总样本通过率",
    "pass_rate_vs_previous": "相对上一层保留率",
    "trades": "交易笔数",
    "long_trades": "多头笔数",
    "win_rate": "胜率",
    "avg_gross_return": "平均毛收益",
    "avg_net_return": "平均净收益",
    "median_net_return": "中位净收益",
    "total_net_return_sum": "总净收益",
    "per_trade_sharpe": "按笔Sharpe",
    "max_drawdown_sum": "最大回撤",
    "benchmark_combo": "比较基准",
    "win_rate_gain_vs_benchmark": "胜率增量",
    "avg_net_return_gain_vs_benchmark": "平均净收益增量",
    "total_net_return_gain_vs_benchmark": "总净收益增量",
    "positive_month_share": "正收益月份占比",
    "sample_status": "样本状态",
    "effectiveness": "有效性判定",
    "min_required_trades": "最低样本要求",
    "win_rate_standard_error": "胜率标准误",
    "win_rate_ci95_low": "胜率95%下界",
    "win_rate_ci95_high": "胜率95%上界",
    "win_rate_diff_z": "胜率差Z值",
    "win_rate_diff_p_value": "胜率差P值",
    "avg_net_diff_bootstrap_ci95_low": "平均净收益差95%下界",
    "avg_net_diff_bootstrap_ci95_high": "平均净收益差95%上界",
    "cost_ticks_roundtrip": "往返成本tick",
    "reward_to_risk_threshold": "盈亏比阈值",
    "severity": "级别",
    "check_id": "检查项",
    "status": "状态",
    "message": "说明",
    "row_count": "行数",
    "detail": "细节",
    "month": "月份",
    "gross_return": "毛收益",
    "net_return": "净收益",
    "feature": "特征",
    "group": "特征组",
    "formula_note": "公式说明",
}


@dataclass(frozen=True)
class TrendPullbackConfig:
    stage2_dir: Path
    output_dir: Path
    train_end: Optional[pd.Timestamp]
    test_start: Optional[pd.Timestamp]
    primary_horizon: int
    holding_bars: int
    cost_ticks_roundtrip: float
    tick_size: float
    split_train_ratio: float
    contracts: Optional[List[str]]
    reward_to_risk_threshold: float
    bootstrap_samples: int
    bootstrap_seed: int


def parse_args(argv: Optional[Sequence[str]] = None) -> TrendPullbackConfig:
    parser = argparse.ArgumentParser(description="Stage 2 trend-pullback long-only gain test")
    parser.add_argument("--stage2_dir", type=str, default=str(DEFAULT_STAGE2_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--primary_horizon", type=int, default=5)
    parser.add_argument("--holding_bars", type=int, default=5)
    parser.add_argument("--cost_ticks_roundtrip", type=float, default=2.0)
    parser.add_argument("--tick_size", type=float, default=10.0)
    parser.add_argument("--split_train_ratio", type=float, default=0.8)
    parser.add_argument("--contracts", type=str, default=None)
    parser.add_argument("--reward_to_risk_threshold", type=float, default=1.5)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260528)
    args = parser.parse_args(argv)

    stage2_dir = Path(args.stage2_dir).expanduser()
    return TrendPullbackConfig(
        stage2_dir=stage2_dir,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else stage2_dir / DEFAULT_OUTPUT_SUBDIR,
        train_end=pd.Timestamp(args.train_end) if args.train_end else None,
        test_start=pd.Timestamp(args.test_start) if args.test_start else None,
        primary_horizon=args.primary_horizon,
        holding_bars=args.holding_bars,
        cost_ticks_roundtrip=args.cost_ticks_roundtrip,
        tick_size=args.tick_size,
        split_train_ratio=args.split_train_ratio,
        contracts=parse_contracts(args.contracts),
        reward_to_risk_threshold=float(args.reward_to_risk_threshold),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )


def feature_catalog() -> pd.DataFrame:
    return pd.DataFrame(TREND_PULLBACK_FEATURE_SPECS)


def make_stage2_config(config: TrendPullbackConfig, data: pd.DataFrame) -> Stage2Config:
    base_config = Stage2Config(
        input_dir=Path(""),
        output_dir=config.stage2_dir,
        start_date=pd.Timestamp("1900-01-01"),
        train_end=config.train_end,
        test_start=config.test_start,
        primary_horizon=config.primary_horizon,
        holding_bars=config.holding_bars,
        cost_ticks_roundtrip=config.cost_ticks_roundtrip,
        tick_size=config.tick_size,
        contracts=config.contracts,
        max_selected_features=18,
        corr_threshold=0.85,
        max_group_share=0.45,
        min_selected_groups=5,
        long_quantile=0.85,
        short_quantile=0.15,
        split_train_ratio=config.split_train_ratio,
    )
    return resolve_time_split(base_config, data)


def load_stage2_pack(config: TrendPullbackConfig) -> pd.DataFrame:
    features = read_csv(config.stage2_dir / "stage2_features.csv", parse_datetime=True)
    labels = read_csv(config.stage2_dir / "stage2_labels.csv", parse_datetime=True)
    if config.contracts is not None:
        features = features[features["contract"].isin(config.contracts)].reset_index(drop=True)
        labels = labels[labels["contract"].isin(config.contracts)].reset_index(drop=True)

    label_only = [col for col in LABEL_COLUMNS if col in labels.columns]
    data = features.merge(labels[["contract", "datetime"] + label_only], on=["contract", "datetime"], how="left", validate="one_to_one")
    data = data.sort_values(["contract", "datetime"]).reset_index(drop=True)
    return data


def _rolling_trendline(close: pd.Series, tick_size: float, window: int = 60) -> Tuple[pd.Series, pd.Series]:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.dot(x_centered, values) / denominator)

    slope_price = close.rolling(window, min_periods=window).apply(slope, raw=True)
    line = close.rolling(window, min_periods=window).mean() + slope_price * (window - 1 - x.mean())
    return line, slope_price / tick_size


def add_trend_pullback_features_for_contract(group: pd.DataFrame, config: TrendPullbackConfig) -> pd.DataFrame:
    """单合约内生成趋势回调 setup 特征，避免 rolling/EMA 跨合约污染。"""

    g = group.sort_values("datetime").copy()
    tick = config.tick_size
    prev_open = g["open"].shift(1)
    prev_close = g["close"].shift(1)
    prev2_open = g["open"].shift(2)
    prev2_close = g["close"].shift(2)

    g["ema60"] = g["close"].ewm(span=60, adjust=False, min_periods=60).mean()
    g["ema60_slope_12_ticks"] = (g["ema60"] - g["ema60"].shift(12)) / tick / 12.0
    g["trendline_60"], g["trendline_slope_60_ticks"] = _rolling_trendline(g["close"], tick, window=60)
    g["trend_filter_pass"] = (
        (g["ema60_slope_12_ticks"] > 0)
        & (g["trendline_slope_60_ticks"] > 0)
        & (g["trend_return_48"] > 0)
        & (g["trend_strength_48"] > 0)
    )

    g["atr20_ticks"] = g["true_range_ticks"].rolling(20, min_periods=20).mean()
    g["adaptive_pullback_tolerance_ticks"] = 0.5 * g["atr20_ticks"]
    adaptive_tolerance_price = g["adaptive_pullback_tolerance_ticks"] * tick
    g["distance_to_ema60_ticks"] = (g["close"] - g["ema60"]) / tick
    g["low_distance_to_ema60_ticks"] = (g["low"] - g["ema60"]) / tick
    g["distance_to_trendline_ticks"] = (g["close"] - g["trendline_60"]) / tick
    g["low_distance_to_trendline_ticks"] = (g["low"] - g["trendline_60"]) / tick
    g["ema60_hold_flag"] = (
        g["ema60"].notna()
        & adaptive_tolerance_price.notna()
        & (g["low"] <= g["ema60"] + adaptive_tolerance_price)
        & (g["low"] >= g["ema60"] - adaptive_tolerance_price)
        & (g["close"] >= g["ema60"])
    )
    g["trendline_hold_flag"] = (
        g["trendline_60"].notna()
        & adaptive_tolerance_price.notna()
        & (g["low"] <= g["trendline_60"] + adaptive_tolerance_price)
        & (g["low"] >= g["trendline_60"] - adaptive_tolerance_price)
        & (g["close"] >= g["trendline_60"])
    )
    g["pullback_structure_pass"] = g["ema60_hold_flag"] | g["trendline_hold_flag"]

    g["bullish_bar_flag"] = g["close"] > g["open"]
    g["strong_bullish_close_flag"] = g["bullish_bar_flag"] & (g["close_location_value"] >= 0.4)
    g["body_quality_flag"] = (g["body_ratio"] >= 0.20) & (g["body_ratio"] <= 0.80)
    prev_bearish = prev_close < prev_open
    g["bullish_engulfing_flag"] = g["bullish_bar_flag"] & prev_bearish & (g["open"] <= prev_close) & (g["close"] >= prev_open)

    prev2_bearish = prev2_close < prev2_open
    prev_small_or_indecision = (g["body_ratio"].shift(1) <= 0.35) | (g["close_location_value"].shift(1).abs() <= 0.2)
    prev2_body_mid = (prev2_open + prev2_close) / 2.0
    g["morning_star_proxy"] = prev2_bearish & prev_small_or_indecision & g["bullish_bar_flag"] & (g["close"] > prev2_body_mid)

    recent_weak = (
        (g["bar_return"].shift(1) < 0)
        | (g["bar_return"].shift(2) < 0)
        | (g["close_location_value"].shift(1) < -0.2)
        | (g["close_location_value"].shift(2) < -0.2)
    )
    g["weak_to_strong_3bar_flag"] = recent_weak & g["strong_bullish_close_flag"] & (g["bar_return"] > g["bar_return"].shift(1).fillna(0))
    turn_pattern = g["bullish_engulfing_flag"] | g["morning_star_proxy"] | g["weak_to_strong_3bar_flag"]
    g["kline_turn_pass"] = g["strong_bullish_close_flag"] & g["body_quality_flag"] & turn_pattern

    g["delta_confirm_long"] = (g["delta"] > 0) & (g["delta_strength"] > 0)
    g["volume_normal_or_better_flag"] = g["volume_zscore_20"] >= -0.5
    g["volume_expand_flag"] = (g["volume_zscore_20"] >= 0.0) | (g["volume_burst_flag"] == 1)
    g["flow_confirm_pass"] = g["delta_confirm_long"] & (g["delta_price_agreement"] >= 0) & g["volume_normal_or_better_flag"]

    g["pullback_low_10"] = g["low"].rolling(10, min_periods=10).min()
    g["previous_swing_high_20"] = g["high"].shift(1).rolling(20, min_periods=20).max()
    g["stop_price_signal"] = g["pullback_low_10"] - tick
    g["stop_distance_ticks_signal"] = (g["close"] - g["stop_price_signal"]) / tick
    g["reward_ticks_signal"] = (g["previous_swing_high_20"] - g["close"]) / tick
    g["reward_to_risk_proxy"] = g["reward_ticks_signal"] / g["stop_distance_ticks_signal"].replace(0, np.nan)
    g["space_filter_pass"] = (
        (g["previous_swing_high_20"] > g["close"])
        & (g["stop_distance_ticks_signal"] >= 2.0)
        & (g["stop_distance_ticks_signal"] <= 30.0)
        & (g["reward_to_risk_proxy"] >= config.reward_to_risk_threshold)
    )

    for combo in COMBO_DEFINITIONS:
        g[f"{combo['combo_name']}_pass"] = g[combo["required_filters"]].all(axis=1)
    g["trade_ready"] = (g["volume"] > 0) & g["entry_open_next"].notna() & g["exit_close_after_5"].notna()
    return g


def build_trend_pullback_features(data: pd.DataFrame, config: TrendPullbackConfig) -> pd.DataFrame:
    frames = [add_trend_pullback_features_for_contract(group, config) for _, group in data.groupby("contract", sort=True)]
    out = pd.concat(frames, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _metrics(trades: pd.DataFrame, split_name: str, return_col: str = "net_return") -> Dict[str, Any]:
    if trades.empty:
        return {
            "split": split_name,
            "trades": 0,
            "long_trades": 0,
            "win_rate": np.nan,
            "avg_gross_return": np.nan,
            "avg_net_return": np.nan,
            "median_net_return": np.nan,
            "total_net_return_sum": 0.0,
            "per_trade_sharpe": np.nan,
            "max_drawdown_sum": np.nan,
        }
    returns = pd.to_numeric(trades[return_col], errors="coerce").dropna()
    gross = pd.to_numeric(trades["gross_return"], errors="coerce").dropna()
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    sharpe = np.nan
    if len(returns) >= 2 and returns.std(ddof=1) != 0:
        sharpe = float(np.sqrt(len(returns)) * returns.mean() / returns.std(ddof=1))
    return {
        "split": split_name,
        "trades": int(len(trades)),
        "long_trades": int((trades["side"] == "long").sum()),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_gross_return": float(gross.mean()) if len(gross) else np.nan,
        "avg_net_return": float(returns.mean()) if len(returns) else np.nan,
        "median_net_return": float(returns.median()) if len(returns) else np.nan,
        "total_net_return_sum": float(returns.sum()) if len(returns) else 0.0,
        "per_trade_sharpe": sharpe,
        "max_drawdown_sum": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def _generate_trades_for_signal(features: pd.DataFrame, stage2_config: Stage2Config, combo_name: str, signal_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for contract, group in features.groupby("contract", sort=True):
        g = group.sort_values("datetime").reset_index(drop=True)
        next_allowed_i = 0
        for i, row in g.iterrows():
            if i < next_allowed_i or not bool(row.get(signal_col, False)) or not bool(row.get("trade_ready", False)):
                continue
            entry_i = i + 1
            exit_i = i + stage2_config.holding_bars
            if exit_i >= len(g) or entry_i >= len(g):
                continue
            entry_price = g.loc[entry_i, "open"]
            exit_price = g.loc[exit_i, "close"]
            exit_datetime = g.loc[exit_i, "datetime"]
            if row["datetime"] <= stage2_config.train_end and exit_datetime >= stage2_config.test_start:
                continue
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                continue
            gross_return = exit_price / entry_price - 1.0
            net_return = gross_return - stage2_config.roundtrip_cost_price / entry_price
            rows.append(
                {
                    "combo_name": combo_name,
                    "contract": contract,
                    "side": "long",
                    "signal_datetime": row["datetime"],
                    "entry_datetime": g.loc[entry_i, "datetime"],
                    "exit_datetime": exit_datetime,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "holding_bars": stage2_config.holding_bars,
                    "split": "train" if row["datetime"] <= stage2_config.train_end else "test",
                    "stop_price_signal": row.get("stop_price_signal", np.nan),
                    "stop_distance_ticks_signal": row.get("stop_distance_ticks_signal", np.nan),
                    "reward_ticks_signal": row.get("reward_ticks_signal", np.nan),
                    "reward_to_risk_proxy": row.get("reward_to_risk_proxy", np.nan),
                }
            )
            next_allowed_i = i + stage2_config.holding_bars
    return pd.DataFrame(rows)


def generate_long_trades(features: pd.DataFrame, stage2_config: Stage2Config) -> pd.DataFrame:
    frames = []
    for combo in COMBO_DEFINITIONS:
        trades = _generate_trades_for_signal(features, stage2_config, combo["combo_name"], f"{combo['combo_name']}_pass")
        if not trades.empty:
            frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_filter_counts(features: pd.DataFrame, stage2_config: Stage2Config) -> pd.DataFrame:
    masks = {
        "train": features["datetime"] <= stage2_config.train_end,
        "test": features["datetime"] >= stage2_config.test_start,
        "all": pd.Series(True, index=features.index),
    }
    rows: List[Dict[str, Any]] = []
    previous_col: Optional[str] = None
    for combo in COMBO_DEFINITIONS:
        combo_col = f"{combo['combo_name']}_pass"
        for split, mask in masks.items():
            part = features.loc[mask]
            total = int(len(part))
            ready = int(part["trade_ready"].sum())
            passed = int((part[combo_col] & part["trade_ready"]).sum())
            previous_passed = total if previous_col is None else int((part[previous_col] & part["trade_ready"]).sum())
            rows.append(
                {
                    "combo_name": combo["combo_name"],
                    "description": combo["description"],
                    "required_filters": ",".join(combo["required_filters"]),
                    "split": split,
                    "rows": total,
                    "trade_ready_rows": ready,
                    "passed_rows": passed,
                    "previous_passed_rows": previous_passed,
                    "pass_rate_total": passed / total if total else np.nan,
                    "pass_rate_vs_previous": passed / previous_passed if previous_passed else np.nan,
                }
            )
        previous_col = combo_col
    return pd.DataFrame(rows)


def build_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for combo in COMBO_DEFINITIONS:
        combo_trades = trades[trades["combo_name"] == combo["combo_name"]] if not trades.empty else trades
        for split in ("train", "test", "all"):
            part = combo_trades if split == "all" else combo_trades[combo_trades["split"] == split] if not combo_trades.empty else combo_trades
            row = _metrics(part, split)
            row["combo_name"] = combo["combo_name"]
            row["description"] = combo["description"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    work["month"] = pd.to_datetime(work["exit_datetime"]).dt.to_period("M").astype(str)
    rows: List[Dict[str, Any]] = []
    for (combo_name, split, month), part in work.groupby(["combo_name", "split", "month"]):
        row = _metrics(part, split)
        row["combo_name"] = combo_name
        row["month"] = month
        rows.append(row)
    return pd.DataFrame(rows)


def build_cost_sensitivity(trades: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame()
    for combo_name, combo_trades in trades.groupby("combo_name"):
        for cost_ticks in (1.0, 2.0, 3.0, 4.0):
            temp = combo_trades.copy()
            temp["net_return_cost_case"] = temp["gross_return"] - (cost_ticks * tick_size) / temp["entry_price"]
            for split in ("train", "test", "all"):
                part = temp if split == "all" else temp[temp["split"] == split]
                row = _metrics(part, split, return_col="net_return_cost_case")
                row["combo_name"] = combo_name
                row["cost_ticks_roundtrip"] = cost_ticks
                rows.append(row)
    return pd.DataFrame(rows)


def build_reward_risk_sensitivity(features: pd.DataFrame, stage2_config: Stage2Config) -> pd.DataFrame:
    """只针对完整趋势回调链条，比较不同盈亏比阈值下的结果。"""

    rows: List[Dict[str, Any]] = []
    base_signal = features["trend_filter_pass"] & features["pullback_structure_pass"] & features["kline_turn_pass"] & features["flow_confirm_pass"]
    for threshold in (1.0, 1.5, 2.0):
        signal_col = "_reward_risk_sensitivity_pass"
        work = features.copy()
        work[signal_col] = (
            base_signal
            & (work["previous_swing_high_20"] > work["close"])
            & (work["stop_distance_ticks_signal"] >= 2.0)
            & (work["stop_distance_ticks_signal"] <= 30.0)
            & (work["reward_to_risk_proxy"] >= threshold)
        )
        trades = _generate_trades_for_signal(work, stage2_config, "trend_pullback_full_space", signal_col)
        for split in ("train", "test", "all"):
            part = trades if split == "all" else trades[trades["split"] == split] if not trades.empty else trades
            row = _metrics(part, split)
            mask = (
                pd.Series(True, index=work.index)
                if split == "all"
                else (work["datetime"] <= stage2_config.train_end)
                if split == "train"
                else (work["datetime"] >= stage2_config.test_start)
            )
            row["combo_name"] = "trend_pullback_full_space"
            row["reward_to_risk_threshold"] = threshold
            row["passed_rows"] = int((work.loc[mask, signal_col] & work.loc[mask, "trade_ready"]).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def build_sample_confidence(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in summary.iterrows():
        trades = int(row["trades"])
        win_rate = float(row["win_rate"]) if pd.notna(row["win_rate"]) else np.nan
        if trades == 0 or pd.isna(win_rate):
            se = np.nan
            ci_low = np.nan
            ci_high = np.nan
            status = "insufficient_samples" if row["split"] == "test" else "no_samples"
        else:
            se = math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / trades)
            ci_low = max(0.0, win_rate - 1.96 * se)
            ci_high = min(1.0, win_rate + 1.96 * se)
            if row["split"] == "test" and trades < MIN_TEST_TRADES:
                status = "insufficient_samples"
            elif row["split"] == "test" and trades < LOW_CONFIDENCE_TRADES:
                status = "low_confidence"
            else:
                status = "enough_samples"
        rows.append(
            {
                "combo_name": row["combo_name"],
                "split": row["split"],
                "trades": trades,
                "min_required_trades": MIN_TEST_TRADES,
                "sample_status": status,
                "win_rate": win_rate,
                "win_rate_standard_error": se,
                "win_rate_ci95_low": ci_low,
                "win_rate_ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def _two_proportion_p_value(wins_a: int, n_a: int, wins_b: int, n_b: int) -> Tuple[float, float]:
    if n_a == 0 or n_b == 0:
        return np.nan, np.nan
    p_a = wins_a / n_a
    p_b = wins_b / n_b
    pooled = (wins_a + wins_b) / (n_a + n_b)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / n_a + 1 / n_b), 0.0))
    if se == 0:
        return np.nan, np.nan
    z = (p_a - p_b) / se
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p_value)


def _bootstrap_mean_diff_ci(current: pd.Series, benchmark: pd.Series, samples: int, seed: int) -> Tuple[float, float]:
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    benchmark_values = pd.to_numeric(benchmark, errors="coerce").dropna().to_numpy(dtype=float)
    if len(current_values) == 0 or len(benchmark_values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = np.empty(samples, dtype=float)
    for i in range(samples):
        current_sample = rng.choice(current_values, size=len(current_values), replace=True)
        benchmark_sample = rng.choice(benchmark_values, size=len(benchmark_values), replace=True)
        diffs[i] = current_sample.mean() - benchmark_sample.mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _positive_month_share(monthly: pd.DataFrame) -> Dict[str, float]:
    if monthly.empty:
        return {}
    out: Dict[str, float] = {}
    test = monthly[monthly["split"] == "test"]
    for combo_name, part in test.groupby("combo_name"):
        out[combo_name] = float((part["avg_net_return"] > 0).mean()) if len(part) else np.nan
    return out


def add_gain_diagnostics(summary: pd.DataFrame, trades: pd.DataFrame, monthly: pd.DataFrame, sample_confidence: pd.DataFrame, config: TrendPullbackConfig) -> pd.DataFrame:
    out = summary.copy()
    sample_cols = ["combo_name", "split", "sample_status", "min_required_trades", "win_rate_standard_error", "win_rate_ci95_low", "win_rate_ci95_high"]
    out = out.merge(sample_confidence[sample_cols], on=["combo_name", "split"], how="left")
    out["benchmark_combo"] = ""
    out["win_rate_gain_vs_benchmark"] = np.nan
    out["avg_net_return_gain_vs_benchmark"] = np.nan
    out["total_net_return_gain_vs_benchmark"] = np.nan
    out["win_rate_diff_z"] = np.nan
    out["win_rate_diff_p_value"] = np.nan
    out["avg_net_diff_bootstrap_ci95_low"] = np.nan
    out["avg_net_diff_bootstrap_ci95_high"] = np.nan
    out["positive_month_share"] = out["combo_name"].map(_positive_month_share(monthly))
    out["effectiveness"] = ""

    test = out[out["split"] == "test"].set_index("combo_name")
    test_trades = trades[trades["split"] == "test"] if not trades.empty else trades
    previous_combo: Optional[str] = None
    for idx, row in out.iterrows():
        if row["split"] != "test":
            continue
        combo = row["combo_name"]
        sample_status = row["sample_status"]
        benchmark = previous_combo
        label = "ineffective"
        win_gain = np.nan
        net_gain = np.nan
        total_gain = np.nan
        if benchmark and benchmark in test.index:
            base = test.loc[benchmark]
            win_gain = row["win_rate"] - base["win_rate"]
            net_gain = row["avg_net_return"] - base["avg_net_return"]
            total_gain = row["total_net_return_sum"] - base["total_net_return_sum"]
            out.loc[idx, "benchmark_combo"] = benchmark
            out.loc[idx, "win_rate_gain_vs_benchmark"] = win_gain
            out.loc[idx, "avg_net_return_gain_vs_benchmark"] = net_gain
            out.loc[idx, "total_net_return_gain_vs_benchmark"] = total_gain
            current_part = test_trades[test_trades["combo_name"] == combo]
            base_part = test_trades[test_trades["combo_name"] == benchmark]
            z, p_value = _two_proportion_p_value(
                int((current_part["net_return"] > 0).sum()),
                len(current_part),
                int((base_part["net_return"] > 0).sum()),
                len(base_part),
            )
            low, high = _bootstrap_mean_diff_ci(current_part["net_return"], base_part["net_return"], config.bootstrap_samples, config.bootstrap_seed)
            out.loc[idx, "win_rate_diff_z"] = z
            out.loc[idx, "win_rate_diff_p_value"] = p_value
            out.loc[idx, "avg_net_diff_bootstrap_ci95_low"] = low
            out.loc[idx, "avg_net_diff_bootstrap_ci95_high"] = high

        if sample_status in {"insufficient_samples", "low_confidence"}:
            out.loc[idx, "effectiveness"] = sample_status
            previous_combo = combo
            continue

        if benchmark and benchmark in test.index:
            if win_gain >= 0.01 and net_gain > 0 and total_gain >= 0:
                label = "incremental_effective"
            elif win_gain > 0 and net_gain > 0 and total_gain < 0:
                label = "quality_improved_not_scalable"
            elif win_gain > 0 and net_gain <= 0:
                label = "win_rate_only"
            elif row["avg_gross_return"] > 0 and row["avg_net_return"] <= 0:
                label = "cost_failed"
        elif row["win_rate"] > 0.5 and row["avg_net_return"] > 0 and row["total_net_return_sum"] > 0:
            label = "absolute_effective"
        elif row["avg_gross_return"] > 0 and row["avg_net_return"] <= 0:
            label = "cost_failed"

        if label in {"absolute_effective", "incremental_effective"} and pd.notna(row["positive_month_share"]) and row["positive_month_share"] < 0.5:
            label = "unstable"
        out.loc[idx, "effectiveness"] = label
        previous_combo = combo
    return out


def run_audit(features: pd.DataFrame, trades: pd.DataFrame, summary: pd.DataFrame, stage2_config: Stage2Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    missing_cols = [col for col in TREND_PULLBACK_FEATURE_COLUMNS if col not in features.columns]
    rows.append(_audit_row("BLOCKER", "ALL", "required_feature_columns", "PASS" if not missing_cols else "FAIL", "趋势回调特征列完整", len(missing_cols), ",".join(missing_cols)))

    duplicate_count = int(features.duplicated(["contract", "datetime"]).sum())
    rows.append(_audit_row("BLOCKER", "ALL", "duplicate_contract_datetime", "PASS" if duplicate_count == 0 else "FAIL", "contract+datetime 不重复", duplicate_count))

    first_window_bad = 0
    for _, group in features.groupby("contract", sort=True):
        ordered = group.sort_values("datetime")
        trend_head = ordered.head(59)
        atr_head = ordered.head(19)
        first_window_bad += int(trend_head["ema60"].notna().sum() + trend_head["trendline_60"].notna().sum() + atr_head["atr20_ticks"].notna().sum())
    rows.append(_audit_row("BLOCKER", "ALL", "rolling_contract_isolation", "PASS" if first_window_bad == 0 else "FAIL", "ATR/EMA/趋势线在单合约内初始化，不跨合约污染", first_window_bad))

    forbidden_signal_cols = ["entry_open_next", "exit_close_after_5", "future_return_1", "future_return_3", "future_return_5", "future_return_10", "mfe_10_long", "mae_10_long"]
    filter_source = ",".join(col for combo in COMBO_DEFINITIONS for col in combo["required_filters"])
    forbidden_used = [col for col in forbidden_signal_cols if col in filter_source]
    rows.append(_audit_row("BLOCKER", "ALL", "no_future_columns_in_filters", "PASS" if not forbidden_used else "FAIL", "过滤条件不使用未来收益或入场退出字段", len(forbidden_used), ",".join(forbidden_used)))

    if trades.empty:
        rows.append(_audit_row("WARNING", "ALL", "no_trades", "WARN", "所有组合均无交易", 0))
    else:
        for combo_name, part in trades.groupby("combo_name"):
            overlap = check_no_overlap(part)
            overlap_count = int(overlap["overlap_count"].sum()) if not overlap.empty else 0
            boundary_bad = int(((part["split"] == "train") & (pd.to_datetime(part["exit_datetime"]) >= stage2_config.test_start)).sum())
            side_bad = int((part["side"] != "long").sum())
            rows.append(_audit_row("BLOCKER", combo_name, "trade_overlap", "PASS" if overlap_count == 0 else "FAIL", "同合约交易不重叠", overlap_count))
            rows.append(_audit_row("BLOCKER", combo_name, "train_test_boundary", "PASS" if boundary_bad == 0 else "FAIL", "训练集交易不穿越测试边界", boundary_bad))
            rows.append(_audit_row("BLOCKER", combo_name, "long_only", "PASS" if side_bad == 0 else "FAIL", "专项实验只做多", side_bad))

    test_summary = summary[summary["split"] == "test"]
    for _, row in test_summary.iterrows():
        if row.get("sample_status") == "insufficient_samples":
            rows.append(_audit_row("WARNING", row["combo_name"], "insufficient_samples", "WARN", "测试集交易样本不足，不能认定有效", int(row["trades"])))
        elif row.get("sample_status") == "low_confidence":
            rows.append(_audit_row("WARNING", row["combo_name"], "low_confidence_samples", "WARN", "测试集交易样本偏少，结论置信度较低", int(row["trades"])))
    return pd.DataFrame(rows)


def _audit_row(severity: str, combo_name: str, check_id: str, status: str, message: str, row_count: int, detail: str = "") -> Dict[str, Any]:
    return {
        "severity": severity,
        "combo_name": combo_name,
        "check_id": check_id,
        "status": status,
        "message": message,
        "row_count": int(row_count),
        "detail": detail,
    }


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "无数据。"
    view = df.head(max_rows).copy()
    original_columns = list(view.columns)
    for col in original_columns:
        if col in {"combo_name", "benchmark_combo", "split", "sample_status", "effectiveness", "severity", "status"}:
            view[col] = view[col].map(lambda x: VALUE_LABELS.get(x, COMBO_LABELS.get(x, x)))
        elif col == "required_filters":
            view[col] = view[col].map(lambda x: str(x).replace(",", " + "))
    columns = [COLUMN_LABELS.get(col, col) for col in original_columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in original_columns) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\n", " ")
    try:
        if pd.isna(value):
            return "无"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    return str(value).replace("\n", " ")


def _explain_combo(row: pd.Series) -> str:
    combo = row["combo_name"]
    label = COMBO_LABELS.get(combo, combo)
    status = row.get("effectiveness", "ineffective")
    trades = int(row.get("trades", 0))
    win_rate = row.get("win_rate", np.nan)
    avg_net = row.get("avg_net_return", np.nan)
    total_net = row.get("total_net_return_sum", np.nan)
    benchmark = row.get("benchmark_combo", "")
    if status == "insufficient_samples":
        return f"- `{label}`：样本不足。测试集 `{trades}` 笔，低于 `{MIN_TEST_TRADES}` 笔最低要求，不能认定有效。"
    if status == "low_confidence":
        return f"- `{label}`：低置信。测试集 `{trades}` 笔，低于 `{LOW_CONFIDENCE_TRADES}` 笔，结论只能作为观察。"
    if status == "absolute_effective":
        return f"- `{label}`：绝对有效。胜率 `{win_rate:.2%}`，平均净收益 `{avg_net:.8f}`，总净收益 `{total_net:.6f}`。"
    if status == "incremental_effective":
        return f"- `{label}`：增量有效。相对 `{COMBO_LABELS.get(benchmark, benchmark)}`，胜率和净收益同步改善，且总净收益没有下降。"
    if status == "quality_improved_not_scalable":
        return f"- `{label}`：单笔质量改善但不可扩展。胜率和平均净收益改善，但交易频率下降导致总净收益下降。"
    if status == "win_rate_only":
        return f"- `{label}`：胜率伪增益。胜率提升但平均净收益没有同步提升。"
    if status == "cost_failed":
        gross = row.get("avg_gross_return", np.nan)
        return f"- `{label}`：成本失败。平均毛收益 `{gross:.8f}`，扣 2 tick 后平均净收益 `{avg_net:.8f}`。"
    if status == "unstable":
        return f"- `{label}`：月度不稳定。虽然总体满足部分有效条件，但测试集正收益月份占比低于 50%。"
    if benchmark:
        win_gain = row.get("win_rate_gain_vs_benchmark", np.nan)
        net_gain = row.get("avg_net_return_gain_vs_benchmark", np.nan)
        total_gain = row.get("total_net_return_gain_vs_benchmark", np.nan)
        return f"- `{label}`：无效。相对 `{COMBO_LABELS.get(benchmark, benchmark)}`，胜率增量 `{win_gain:.2%}`，平均净收益增量 `{net_gain:.8f}`，总净收益增量 `{total_gain:.6f}`。"
    return f"- `{label}`：无效。胜率 `{win_rate:.2%}`，平均净收益 `{avg_net:.8f}`，总净收益 `{total_net:.6f}`。"


def write_report(
    output_path: Path,
    stage2_config: Stage2Config,
    features: pd.DataFrame,
    filter_counts: pd.DataFrame,
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    sample_confidence: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    reward_risk_sensitivity: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    test = summary[summary["split"] == "test"].copy()
    blockers = audit[(audit["severity"] == "BLOCKER") & (audit["status"] == "FAIL")]
    lines: List[str] = []
    lines.append("# 阶段二趋势回调做多增益测试报告\n")
    lines.append("## 结论")
    final_row = test[test["combo_name"] == "trend_pullback_full_space"]
    final_status = final_row.iloc[0]["effectiveness"] if not final_row.empty else "ineffective"
    if final_status in {"absolute_effective", "incremental_effective"}:
        conclusion = "成立"
    elif final_status in {"insufficient_samples", "low_confidence"}:
        conclusion = "样本不足，暂不能确认"
    else:
        conclusion = "不成立"
    lines.append(f"- 趋势回调做多完整链条：`{conclusion}`。")
    lines.append(f"- 审计阻断项：`{len(blockers)}`。")
    lines.append("- 回踩结构使用 `±0.5 * ATR20` 自适应容差，不再使用固定 tick 容差。")
    lines.append(f"- 数据范围：`{features['datetime'].min()}` 到 `{features['datetime'].max()}`，合约数 `{features['contract'].nunique()}`，样本行数 `{len(features)}`。")
    lines.append(f"- 训练/测试切分：训练集截至 `{stage2_config.train_end}`，测试集从 `{stage2_config.test_start}` 开始。")
    lines.append("")

    lines.append("## 测试集增益曲线")
    summary_cols = [
        "combo_name",
        "trades",
        "sample_status",
        "win_rate",
        "win_rate_ci95_low",
        "win_rate_ci95_high",
        "avg_gross_return",
        "avg_net_return",
        "total_net_return_sum",
        "max_drawdown_sum",
        "benchmark_combo",
        "win_rate_gain_vs_benchmark",
        "avg_net_return_gain_vs_benchmark",
        "total_net_return_gain_vs_benchmark",
        "positive_month_share",
        "win_rate_diff_p_value",
        "avg_net_diff_bootstrap_ci95_low",
        "avg_net_diff_bootstrap_ci95_high",
        "effectiveness",
    ]
    lines.append(_md_table(test[summary_cols], max_rows=10))
    lines.append("")

    lines.append("## 组合逐项解释")
    for _, row in test.iterrows():
        lines.append(_explain_combo(row))
    lines.append("")

    lines.append("## 过滤漏斗")
    funnel = filter_counts[filter_counts["split"] == "test"][
        ["combo_name", "description", "required_filters", "rows", "trade_ready_rows", "passed_rows", "previous_passed_rows", "pass_rate_total", "pass_rate_vs_previous"]
    ]
    lines.append(_md_table(funnel, max_rows=10))
    lines.append("")

    lines.append("## 样本置信度")
    sample_test = sample_confidence[sample_confidence["split"] == "test"][
        ["combo_name", "split", "trades", "min_required_trades", "sample_status", "win_rate", "win_rate_standard_error", "win_rate_ci95_low", "win_rate_ci95_high"]
    ]
    lines.append(_md_table(sample_test, max_rows=10))
    lines.append("")

    lines.append("## 成本敏感性")
    cost_test = cost_sensitivity[cost_sensitivity["split"] == "test"][
        ["combo_name", "cost_ticks_roundtrip", "trades", "win_rate", "avg_net_return", "total_net_return_sum"]
    ]
    lines.append(_md_table(cost_test, max_rows=30))
    lines.append("")

    lines.append("## 盈亏比阈值敏感性")
    rr_test = reward_risk_sensitivity[reward_risk_sensitivity["split"] == "test"][
        ["reward_to_risk_threshold", "passed_rows", "trades", "win_rate", "avg_net_return", "total_net_return_sum", "max_drawdown_sum"]
    ]
    lines.append(_md_table(rr_test, max_rows=10))
    lines.append("")

    lines.append("## 月度稳定性")
    monthly_test = monthly[monthly["split"] == "test"][["combo_name", "month", "trades", "win_rate", "avg_net_return", "total_net_return_sum"]]
    lines.append(_md_table(monthly_test, max_rows=40))
    lines.append("")

    lines.append("## 审计")
    lines.append(_md_table(audit, max_rows=60))
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(config: Optional[TrendPullbackConfig] = None) -> Dict[str, Any]:
    config = config or parse_args()
    data = load_stage2_pack(config)
    stage2_config = make_stage2_config(config, data)
    features = build_trend_pullback_features(data, config)
    trades = generate_long_trades(features, stage2_config)
    filter_counts = build_filter_counts(features, stage2_config)
    summary = build_summary(trades)
    monthly = build_monthly_metrics(trades)
    sample_confidence = build_sample_confidence(summary)
    cost_sensitivity = build_cost_sensitivity(trades, stage2_config.tick_size)
    reward_risk_sensitivity = build_reward_risk_sensitivity(features, stage2_config)
    summary = add_gain_diagnostics(summary, trades, monthly, sample_confidence, config)
    audit = run_audit(features, trades, summary, stage2_config)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_output_cols = BASE_COLUMNS + TREND_PULLBACK_FEATURE_COLUMNS + [f"{combo['combo_name']}_pass" for combo in COMBO_DEFINITIONS] + ["trade_ready"]
    write_csv(features[feature_output_cols], output_dir / "trend_pullback_features.csv")
    write_csv(feature_catalog(), output_dir / "trend_pullback_feature_catalog.csv")
    write_csv(filter_counts, output_dir / "trend_pullback_filter_counts.csv")
    write_csv(summary, output_dir / "trend_pullback_gain_summary.csv")
    write_csv(trades, output_dir / "trend_pullback_gain_trades.csv")
    write_csv(monthly, output_dir / "trend_pullback_gain_monthly.csv")
    write_csv(sample_confidence, output_dir / "trend_pullback_gain_sample_confidence.csv")
    write_csv(cost_sensitivity, output_dir / "trend_pullback_gain_cost_sensitivity.csv")
    write_csv(reward_risk_sensitivity, output_dir / "trend_pullback_reward_risk_sensitivity.csv")
    write_csv(audit, output_dir / "trend_pullback_gain_audit.csv")
    write_report(
        output_dir / "trend_pullback_gain_report_zh.md",
        stage2_config,
        features,
        filter_counts,
        summary,
        monthly,
        sample_confidence,
        cost_sensitivity,
        reward_risk_sensitivity,
        audit,
    )

    blockers = audit[(audit["severity"] == "BLOCKER") & (audit["status"] == "FAIL")]
    return {
        "output_dir": output_dir,
        "contracts": int(features["contract"].nunique()),
        "rows": int(len(features)),
        "combos": len(COMBO_DEFINITIONS),
        "train_end": stage2_config.train_end,
        "test_start": stage2_config.test_start,
        "trades": int(len(trades)),
        "audit_blockers": int(len(blockers)),
    }
