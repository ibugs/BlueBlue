"""趋势回调参数扫描与退出原因诊断。

本模块采用“统一策略 + 参数开关”的方式扫描趋势回调 setup：
多头基础信号为“上涨趋势 + 回踩 + 量价确认”，空头基础信号为
“下跌趋势 + 反抽 + 量价确认”。K线过滤和空间过滤仅作为参数开关，
避免策略层级与 K线模式交叉后产生逻辑重复。
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Stage2Config, parse_contracts
from .io import write_csv
from .strategy import check_no_overlap
from .trend_pullback import (
    LOW_CONFIDENCE_TRADES,
    MIN_TEST_TRADES,
    TrendPullbackConfig,
    build_trend_pullback_features,
    load_stage2_pack,
    make_stage2_config,
)


DEFAULT_STAGE2_DIR = Path("/Users/wangrendong/Projects/BlueBlue/stage2_outputs/SHFE.cu")
DEFAULT_OUTPUT_SUBDIR = "trend_pullback_sweep_ls"

DEFAULT_TIME_BARRIERS = (5, 12, 24, 36)
DEFAULT_STOP_ATR_MULTIPLES = (0.5, 0.75, 1.0, 1.25)
DEFAULT_TAKE_PROFIT_MULTIPLES = (1.0, 1.5, 2.0, 2.5)
DEFAULT_KLINE_MODES = ("none", "strong_close_only", "full_patterns")

STRATEGY_NAME = "trend_pullback_flow"
STRATEGY_LABELS = {
    "long": "上涨趋势+回踩+量价确认",
    "short": "下跌趋势+反抽+量价确认",
}
SIDE_LABELS = {"long": "多头", "short": "空头"}
DEFAULT_SIDES = ("long", "short")

VALUE_LABELS = {
    "train": "训练集",
    "test": "测试集",
    "all": "全部",
    "long": "多头",
    "short": "空头",
    "none": "无K线过滤",
    "strong_close_only": "顺方向收盘",
    "full_patterns": "完整K线形态",
    "insufficient_samples": "样本不足",
    "low_confidence": "低置信度",
    "enough_samples": "样本充足",
    "no_samples": "无样本",
    "absolute_effective": "绝对有效",
    "cost_failed": "成本吞噬",
    "ineffective": "无效",
    "passes_bonferroni": "通过多重检验修正",
    "significant_below_50": "显著低于50%",
    "not_significant": "未通过多重检验修正",
    "not_tested": "未检验",
    "train_test_consistent": "训练测试一致改善",
    "train_only": "仅训练集改善",
    "train_not_passed": "训练平原也未通过",
    "test_only": "测试集偶然最优，不作为参数建议",
    "not_verified": "测试集未验证",
    "PASS": "通过",
    "FAIL": "失败",
    "WARN": "警告",
    "BLOCKER": "阻断",
    "WARNING": "警告",
    True: "开",
    False: "关",
}

COLUMN_LABELS = {
    "case_id": "参数组合ID",
    "train_rank": "训练排名",
    "side": "方向",
    "combo_name": "策略层级",
    "description": "说明",
    "split": "切分",
    "time_barrier_bars": "时间屏障",
    "stop_atr_multiple": "止损ATR倍数",
    "take_profit_multiple": "止盈倍数",
    "kline_mode": "K线模式",
    "space_filter_on": "空间过滤",
    "trades": "交易笔数",
    "win_rate": "胜率",
    "avg_gross_return": "平均毛收益",
    "avg_net_return": "平均净收益",
    "train_avg_net_return": "训练平均净收益",
    "train_total_net_return_sum": "训练总净收益",
    "median_net_return": "中位净收益",
    "total_net_return_sum": "总净收益",
    "per_trade_sharpe": "按笔Sharpe",
    "max_drawdown_sum": "最大回撤",
    "positive_month_share": "正收益月份占比",
    "sample_status": "样本状态",
    "effectiveness": "有效性判定",
    "win_rate_standard_error": "胜率标准误",
    "win_rate_ci95_low": "胜率95%下界",
    "win_rate_ci95_high": "胜率95%上界",
    "win_rate_p_value_vs_50": "胜率对50%P值",
    "win_rate_p_value_bonferroni": "Bonferroni修正P值",
    "multiple_testing_status": "多重检验状态",
    "exit_reason": "退出原因",
    "exit_count": "退出笔数",
    "exit_rate": "退出占比",
    "stop_loss_rate": "止损占比",
    "take_profit_rate": "止盈占比",
    "time_barrier_rate": "时间退出占比",
    "take_profit_to_stop_count_ratio": "止盈/止损次数比",
    "avg_stop_loss_net_return": "平均止损净收益",
    "avg_take_profit_net_return": "平均止盈净收益",
    "avg_time_barrier_net_return": "平均时间退出净收益",
    "avg_stop_distance_ticks": "平均止损距离tick",
    "avg_take_profit_distance_ticks": "平均止盈距离tick",
    "avg_holding_bars": "平均持有Bar",
    "month": "月份",
    "diagnosis": "诊断结论",
    "recommendation_status": "建议状态",
    "summary_bonferroni_p_value": "汇总Bonferroni修正P值",
    "summary_multiple_testing_status": "汇总多重检验状态",
    "parameter_overlap_score": "参数重叠度",
    "asymmetry_score": "多空非对称性得分",
    "time_barrier_similarity": "时间屏障相似度",
    "stop_similarity": "止损相似度",
    "take_profit_similarity": "止盈相似度",
    "kline_similarity": "K线模式相似度",
    "space_similarity": "空间过滤相似度",
    "selection_source": "选择来源",
    "total_trial_count": "总试验次数",
    "plateau_cases": "参数平原邻域数",
    "plateau_mean_avg_net_return": "平原平均净收益",
    "plateau_std_avg_net_return": "平原净收益标准差",
    "plateau_positive_share": "平原正收益占比",
    "plateau_score": "参数平原评分",
    "cases": "参数组合数",
    "mean_avg_net_return": "平均净收益均值",
    "median_avg_net_return": "平均净收益中位数",
    "mean_total_net_return": "总净收益均值",
    "mean_win_rate": "胜率均值",
    "mean_take_profit_rate": "止盈占比均值",
    "mean_stop_loss_rate": "止损占比均值",
    "mean_time_barrier_rate": "时间退出占比均值",
    "current_mode": "当前模式",
    "benchmark_mode": "基准模式",
    "net_gain": "净收益增量",
    "total_gain": "总收益增量",
    "benchmark_trades": "基准交易笔数",
    "benchmark_win_rate": "基准胜率",
    "benchmark_avg_net_return": "基准平均净收益",
    "benchmark_total_net_return_sum": "基准总净收益",
    "win_rate_gain_vs_benchmark": "胜率增量",
    "severity": "级别",
    "check_id": "检查项",
    "status": "状态",
    "message": "说明",
    "row_count": "行数",
    "detail": "细节",
}


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str, allowed: Iterable[str]) -> List[str]:
    allowed_set = set(allowed)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    bad = [item for item in values if item not in allowed_set]
    if bad:
        raise ValueError(f"Unsupported values: {bad}; allowed={sorted(allowed_set)}")
    return values


def _parse_sides(raw: str) -> List[str]:
    return _parse_str_list(raw, DEFAULT_SIDES)


def _parse_bool_list(raw: str) -> List[bool]:
    out: List[bool] = []
    for item in [part.strip().lower() for part in raw.split(",") if part.strip()]:
        if item in {"on", "true", "1", "yes"}:
            out.append(True)
        elif item in {"off", "false", "0", "no"}:
            out.append(False)
        else:
            raise ValueError(f"Unsupported space filter value: {item}")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend-pullback parameter sweep and exit diagnostics")
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
    parser.add_argument("--time_barrier_bars", type=str, default=",".join(map(str, DEFAULT_TIME_BARRIERS)))
    parser.add_argument("--stop_atr_multiples", type=str, default=",".join(map(str, DEFAULT_STOP_ATR_MULTIPLES)))
    parser.add_argument("--take_profit_multiples", type=str, default=",".join(map(str, DEFAULT_TAKE_PROFIT_MULTIPLES)))
    parser.add_argument("--kline_modes", type=str, default=",".join(DEFAULT_KLINE_MODES))
    parser.add_argument("--space_filters", type=str, default="off,on")
    parser.add_argument("--sides", type=str, default=",".join(DEFAULT_SIDES))
    parser.add_argument("--max_train_rank", type=int, default=10)
    parser.add_argument("--report_tag", type=str, default="", help="可选；非空时生成带 tag 的 Markdown 报告文件名，避免覆盖旧报告")
    return parser.parse_args(argv)


def _base_config(args: argparse.Namespace) -> TrendPullbackConfig:
    stage2_dir = Path(args.stage2_dir).expanduser()
    return TrendPullbackConfig(
        stage2_dir=stage2_dir,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else stage2_dir / DEFAULT_OUTPUT_SUBDIR,
        train_end=pd.Timestamp(args.train_end) if args.train_end else None,
        test_start=pd.Timestamp(args.test_start) if args.test_start else None,
        primary_horizon=int(args.primary_horizon),
        holding_bars=int(args.holding_bars),
        cost_ticks_roundtrip=float(args.cost_ticks_roundtrip),
        tick_size=float(args.tick_size),
        split_train_ratio=float(args.split_train_ratio),
        contracts=parse_contracts(args.contracts),
        reward_to_risk_threshold=float(args.reward_to_risk_threshold),
        bootstrap_samples=0,
        bootstrap_seed=20260528,
        bootstrap_block_size=0,
        exit_mode="triple_barrier",
        stop_atr_multiple=1.0,
        take_profit_multiple=2.0,
        time_barrier_bars=None,
    )


def _parameter_grid(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grid = itertools.product(
        _parse_int_list(args.time_barrier_bars),
        _parse_float_list(args.stop_atr_multiples),
        _parse_float_list(args.take_profit_multiples),
        _parse_str_list(args.kline_modes, DEFAULT_KLINE_MODES),
        _parse_bool_list(args.space_filters),
    )
    for case_num, (time_bars, stop_mult, take_mult, kline_mode, space_on) in enumerate(grid, start=1):
        rows.append(
            {
                "case_id": f"case_{case_num:04d}",
                "combo_name": STRATEGY_NAME,
                "description": "趋势回调量价确认",
                "time_barrier_bars": int(time_bars),
                "stop_atr_multiple": float(stop_mult),
                "take_profit_multiple": float(take_mult),
                "kline_mode": kline_mode,
                "space_filter_on": bool(space_on),
            }
        )
    return pd.DataFrame(rows)


def add_short_mirror_features(features: pd.DataFrame, config: TrendPullbackConfig) -> pd.DataFrame:
    """生成空头反抽 setup 的镜像特征，全部在单合约内计算。"""

    frames: List[pd.DataFrame] = []
    tick = float(config.tick_size)
    for _, group in features.groupby("contract", sort=True):
        g = group.sort_values("datetime").copy()
        tol_price = g["adaptive_pullback_tolerance_ticks"] * tick
        g["short_trend_filter_pass"] = (
            (g["ema60_slope_12_ticks"] < 0)
            & (g["trendline_slope_60_ticks"] < 0)
            & (g["trend_return_48"] < 0)
            & (g["trend_strength_48"] < 0)
        )
        g["short_distance_to_ema60_ticks"] = (g["ema60"] - g["close"]) / tick
        g["short_high_distance_to_ema60_ticks"] = (g["high"] - g["ema60"]) / tick
        g["short_distance_to_trendline_ticks"] = (g["trendline_60"] - g["close"]) / tick
        g["short_high_distance_to_trendline_ticks"] = (g["high"] - g["trendline_60"]) / tick
        g["short_ema60_reject_flag"] = (
            g["ema60"].notna()
            & tol_price.notna()
            & (g["high"] >= g["ema60"] - tol_price)
            & (g["high"] <= g["ema60"] + tol_price)
            & (g["close"] <= g["ema60"])
        )
        g["short_trendline_reject_flag"] = (
            g["trendline_60"].notna()
            & tol_price.notna()
            & (g["high"] >= g["trendline_60"] - tol_price)
            & (g["high"] <= g["trendline_60"] + tol_price)
            & (g["close"] <= g["trendline_60"])
        )
        g["short_pullback_structure_pass"] = g["short_ema60_reject_flag"] | g["short_trendline_reject_flag"]

        prev_open = g["open"].shift(1)
        prev_close = g["close"].shift(1)
        prev2_open = g["open"].shift(2)
        prev2_close = g["close"].shift(2)
        g["bearish_bar_flag"] = g["close"] < g["open"]
        g["strong_bearish_close_flag"] = g["bearish_bar_flag"] & (g["close_location_value"] <= -0.4)
        prev_bullish = prev_close > prev_open
        g["bearish_engulfing_flag"] = g["bearish_bar_flag"] & prev_bullish & (g["open"] >= prev_close) & (g["close"] <= prev_open)
        prev2_bullish = prev2_close > prev2_open
        prev_small_or_indecision = (g["body_ratio"].shift(1) <= 0.35) | (g["close_location_value"].shift(1).abs() <= 0.2)
        prev2_body_mid = (prev2_open + prev2_close) / 2.0
        g["evening_star_proxy"] = prev2_bullish & prev_small_or_indecision & g["bearish_bar_flag"] & (g["close"] < prev2_body_mid)
        recent_strong = (
            (g["bar_return"].shift(1) > 0)
            | (g["bar_return"].shift(2) > 0)
            | (g["close_location_value"].shift(1) > 0.2)
            | (g["close_location_value"].shift(2) > 0.2)
        )
        g["strong_to_weak_3bar_flag"] = recent_strong & g["strong_bearish_close_flag"] & (g["bar_return"] < g["bar_return"].shift(1).fillna(0))
        short_turn = g["bearish_engulfing_flag"] | g["evening_star_proxy"] | g["strong_to_weak_3bar_flag"]
        g["short_kline_turn_pass"] = g["strong_bearish_close_flag"] & g["body_quality_flag"] & short_turn

        g["short_delta_confirm"] = (g["delta"] < 0) & (g["delta_strength"] < 0)
        g["short_flow_confirm_pass"] = g["short_delta_confirm"] & (g["delta_price_agreement"] >= 0) & g["volume_normal_or_better_flag"]

        g["pullback_high_10"] = g["high"].rolling(10, min_periods=10).max()
        g["previous_swing_low_20"] = g["low"].shift(1).rolling(20, min_periods=20).min()
        g["short_stop_price_signal"] = g["pullback_high_10"] + tick
        g["short_stop_distance_ticks_signal"] = (g["short_stop_price_signal"] - g["close"]) / tick
        g["short_reward_ticks_signal"] = (g["close"] - g["previous_swing_low_20"]) / tick
        g["short_reward_to_risk_proxy"] = g["short_reward_ticks_signal"] / g["short_stop_distance_ticks_signal"].replace(0, np.nan)
        g["short_space_filter_pass"] = (
            (g["previous_swing_low_20"] < g["close"])
            & (g["short_stop_distance_ticks_signal"] >= 2.0)
            & (g["short_stop_distance_ticks_signal"] <= 30.0)
            & (g["short_reward_to_risk_proxy"] >= config.reward_to_risk_threshold)
        )
        frames.append(g)
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _kline_pass(features: pd.DataFrame, mode: str, side: str) -> pd.Series:
    if mode == "none":
        return pd.Series(True, index=features.index)
    if side == "long":
        if mode == "strong_close_only":
            return features["strong_bullish_close_flag"].fillna(False) & features["body_quality_flag"].fillna(False)
        if mode == "full_patterns":
            return features["kline_turn_pass"].fillna(False)
    if side == "short":
        if mode == "strong_close_only":
            return features["strong_bearish_close_flag"].fillna(False) & features["body_quality_flag"].fillna(False)
        if mode == "full_patterns":
            return features["short_kline_turn_pass"].fillna(False)
    raise ValueError(f"Unsupported kline mode: {mode}")


def _signal_mask(features: pd.DataFrame, side: str, kline_mode: str, space_filter_on: bool) -> pd.Series:
    if side == "long":
        base = (
            features["trend_filter_pass"].fillna(False)
            & features["pullback_structure_pass"].fillna(False)
            & features["flow_confirm_pass"].fillna(False)
            & features["trade_ready"].fillna(False)
        )
        signal = base & _kline_pass(features, kline_mode, side)
        if space_filter_on:
            signal = signal & features["space_filter_pass"].fillna(False)
        return signal
    if side == "short":
        base = (
            features["short_trend_filter_pass"].fillna(False)
            & features["short_pullback_structure_pass"].fillna(False)
            & features["short_flow_confirm_pass"].fillna(False)
            & features["trade_ready"].fillna(False)
        )
        signal = base & _kline_pass(features, kline_mode, side)
        if space_filter_on:
            signal = signal & features["short_space_filter_pass"].fillna(False)
        return signal
    raise ValueError(f"Unsupported side: {side}")


def _side_signal_values(g: pd.DataFrame, side: str) -> Dict[str, np.ndarray]:
    if side == "long":
        return {
            "signal_stop": g["stop_price_signal"].to_numpy(dtype=float),
            "stop_price_signal": g["stop_price_signal"].to_numpy(dtype=float),
            "stop_distance_ticks_signal": g["stop_distance_ticks_signal"].to_numpy(dtype=float),
            "reward_ticks_signal": g["reward_ticks_signal"].to_numpy(dtype=float),
            "reward_to_risk_proxy": g["reward_to_risk_proxy"].to_numpy(dtype=float),
        }
    if side == "short":
        return {
            "signal_stop": g["short_stop_price_signal"].to_numpy(dtype=float),
            "stop_price_signal": g["short_stop_price_signal"].to_numpy(dtype=float),
            "stop_distance_ticks_signal": g["short_stop_distance_ticks_signal"].to_numpy(dtype=float),
            "reward_ticks_signal": g["short_reward_ticks_signal"].to_numpy(dtype=float),
            "reward_to_risk_proxy": g["short_reward_to_risk_proxy"].to_numpy(dtype=float),
        }
    raise ValueError(f"Unsupported side: {side}")
    return signal


def _precompute_exit_vectors(
    g: pd.DataFrame,
    stage2_config: Stage2Config,
    time_bars: int,
    stop_atr_multiple: float,
    take_profit_multiple: float,
    side: str,
) -> Dict[str, np.ndarray]:
    """向量化预计算单合约内每根信号Bar的 Triple Barrier 退出结果。"""

    n = len(g)
    tick = float(stage2_config.tick_size)
    idx = np.arange(n)
    entry_i = idx + 1
    time_exit_i = idx + int(time_bars)
    valid = (entry_i < n) & (time_exit_i < n)

    opens = g["open"].to_numpy(dtype=float)
    closes = g["close"].to_numpy(dtype=float)
    highs = g["high"].to_numpy(dtype=float)
    lows = g["low"].to_numpy(dtype=float)
    entry_price = np.full(n, np.nan, dtype=float)
    entry_price[valid] = opens[entry_i[valid]]

    atr_ticks = g["atr20_ticks"].to_numpy(dtype=float)
    side_values = _side_signal_values(g, side)
    signal_stop = side_values["signal_stop"]
    if side == "long":
        signal_stop_distance = entry_price - signal_stop
    elif side == "short":
        signal_stop_distance = signal_stop - entry_price
    else:
        raise ValueError(f"Unsupported side: {side}")
    signal_stop_distance[~((signal_stop_distance > 0) & np.isfinite(signal_stop_distance))] = np.nan
    atr_stop_distance = stop_atr_multiple * atr_ticks * tick
    atr_stop_distance[~((atr_stop_distance > 0) & np.isfinite(atr_stop_distance))] = np.nan

    stacked = np.vstack(
        [
            np.where(np.isfinite(signal_stop_distance), signal_stop_distance, -np.inf),
            np.where(np.isfinite(atr_stop_distance), atr_stop_distance, -np.inf),
            np.full(n, tick, dtype=float),
        ]
    )
    source_idx = np.argmax(stacked, axis=0)
    stop_distance = stacked[source_idx, np.arange(n)]
    valid = valid & np.isfinite(entry_price) & (entry_price > 0) & np.isfinite(stop_distance) & (stop_distance > 0)

    if side == "long":
        stop_price = entry_price - stop_distance
        take_profit_price = entry_price + take_profit_multiple * stop_distance
    else:
        stop_price = entry_price + stop_distance
        take_profit_price = entry_price - take_profit_multiple * stop_distance
    exit_i = time_exit_i.copy()
    exit_price = np.full(n, np.nan, dtype=float)
    exit_price[valid] = closes[time_exit_i[valid]]
    reason_code = np.full(n, 0, dtype=np.int8)  # 0 invalid, 1 time, 2 stop, 3 take
    reason_code[valid] = 1
    resolved = ~valid.copy()

    for step in range(1, int(time_bars) + 1):
        j = idx + step
        in_range = j < n
        active = (~resolved) & in_range
        if not active.any():
            continue
        if side == "long":
            stop_hit = active & (lows[j.clip(max=n - 1)] <= stop_price)
        else:
            stop_hit = active & (highs[j.clip(max=n - 1)] >= stop_price)
        if stop_hit.any():
            exit_i[stop_hit] = j[stop_hit]
            open_at_stop = opens[j[stop_hit]]
            if side == "long":
                exit_price[stop_hit] = np.where(np.isfinite(open_at_stop), np.minimum(open_at_stop, stop_price[stop_hit]), stop_price[stop_hit])
            else:
                exit_price[stop_hit] = np.where(np.isfinite(open_at_stop), np.maximum(open_at_stop, stop_price[stop_hit]), stop_price[stop_hit])
            reason_code[stop_hit] = 2
            resolved[stop_hit] = True
        active = (~resolved) & in_range
        if side == "long":
            take_hit = active & (highs[j.clip(max=n - 1)] >= take_profit_price)
        else:
            take_hit = active & (lows[j.clip(max=n - 1)] <= take_profit_price)
        if take_hit.any():
            exit_i[take_hit] = j[take_hit]
            exit_price[take_hit] = take_profit_price[take_hit]
            reason_code[take_hit] = 3
            resolved[take_hit] = True

    return {
        "valid": valid,
        "entry_i": entry_i,
        "exit_i": exit_i,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "reason_code": reason_code,
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "stop_distance_ticks": stop_distance / tick,
        "take_profit_distance_ticks": np.abs(take_profit_price - entry_price) / tick,
        "stop_distance_source": np.array(["signal", "atr", "tick"], dtype=object)[source_idx],
        "stop_price_signal": side_values["stop_price_signal"],
        "stop_distance_ticks_signal": side_values["stop_distance_ticks_signal"],
        "reward_ticks_signal": side_values["reward_ticks_signal"],
        "reward_to_risk_proxy": side_values["reward_to_risk_proxy"],
    }


def _gross_return(entry_price: float, exit_price: float, side: str) -> float:
    if side == "long":
        return exit_price / entry_price - 1.0
    if side == "short":
        return 1.0 - exit_price / entry_price
    raise ValueError(f"Unsupported side: {side}")


def _generate_case_trades(features: pd.DataFrame, signal: pd.Series, stage2_config: Stage2Config, case: pd.Series, side: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    reason_map = {1: "time_barrier", 2: "stop_loss", 3: "take_profit"}
    signal = signal.fillna(False)
    for contract, group in features.groupby("contract", sort=True):
        sorted_group = group.sort_values("datetime")
        g = sorted_group.reset_index(drop=True)
        group_signal = signal.loc[sorted_group.index].to_numpy(dtype=bool)
        exits = _precompute_exit_vectors(
            g,
            stage2_config,
            int(case["time_barrier_bars"]),
            float(case["stop_atr_multiple"]),
            float(case["take_profit_multiple"]),
            side,
        )
        candidates = np.flatnonzero(group_signal & exits["valid"])
        next_allowed_i = 0
        for i in candidates:
            if i < next_allowed_i:
                continue
            entry_i = int(exits["entry_i"][i])
            exit_i = int(exits["exit_i"][i])
            signal_datetime = g.at[i, "datetime"]
            exit_datetime = g.at[exit_i, "datetime"]
            if signal_datetime <= stage2_config.train_end and exit_datetime >= stage2_config.test_start:
                continue
            if signal_datetime <= stage2_config.train_end:
                split = "train"
            elif signal_datetime >= stage2_config.test_start:
                split = "test"
            else:
                continue
            entry_price = float(exits["entry_price"][i])
            exit_price = float(exits["exit_price"][i])
            gross_return = _gross_return(entry_price, exit_price, side)
            net_return = gross_return - stage2_config.roundtrip_cost_price / entry_price
            rows.append(
                {
                    "case_id": case["case_id"],
                    "combo_name": STRATEGY_NAME,
                    "contract": contract,
                    "side": side,
                    "signal_datetime": signal_datetime,
                    "entry_datetime": g.at[entry_i, "datetime"],
                    "exit_datetime": exit_datetime,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "holding_bars": int(exit_i - entry_i + 1),
                    "exit_reason": reason_map.get(int(exits["reason_code"][i]), "invalid"),
                    "stop_price_barrier": exits["stop_price"][i],
                    "take_profit_price_barrier": exits["take_profit_price"][i],
                    "stop_distance_ticks_barrier": exits["stop_distance_ticks"][i],
                    "take_profit_distance_ticks_barrier": exits["take_profit_distance_ticks"][i],
                    "stop_distance_source": exits["stop_distance_source"][i],
                    "split": split,
                    "time_barrier_bars": case["time_barrier_bars"],
                    "stop_atr_multiple": case["stop_atr_multiple"],
                    "take_profit_multiple": case["take_profit_multiple"],
                    "kline_mode": case["kline_mode"],
                    "space_filter_on": case["space_filter_on"],
                    "stop_price_signal": exits["stop_price_signal"][i],
                    "stop_distance_ticks_signal": exits["stop_distance_ticks_signal"][i],
                    "reward_ticks_signal": exits["reward_ticks_signal"][i],
                    "reward_to_risk_proxy": exits["reward_to_risk_proxy"][i],
                }
            )
            next_allowed_i = exit_i + 1
    return pd.DataFrame(rows)


def _metrics(trades: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    if trades.empty:
        return {
            "split": split_name,
            "trades": 0,
            "win_rate": np.nan,
            "avg_gross_return": np.nan,
            "avg_net_return": np.nan,
            "median_net_return": np.nan,
            "total_net_return_sum": 0.0,
            "per_trade_sharpe": np.nan,
            "max_drawdown_sum": np.nan,
        }
    returns = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    gross = pd.to_numeric(trades["gross_return"], errors="coerce").dropna()
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    sharpe = np.nan
    if len(returns) >= 2 and returns.std(ddof=1) != 0:
        sharpe = float(np.sqrt(len(returns)) * returns.mean() / returns.std(ddof=1))
    return {
        "split": split_name,
        "trades": int(len(trades)),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_gross_return": float(gross.mean()) if len(gross) else np.nan,
        "avg_net_return": float(returns.mean()) if len(returns) else np.nan,
        "median_net_return": float(returns.median()) if len(returns) else np.nan,
        "total_net_return_sum": float(returns.sum()) if len(returns) else 0.0,
        "per_trade_sharpe": sharpe,
        "max_drawdown_sum": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def _sample_status(split: str, trades: int) -> str:
    if trades == 0:
        return "no_samples"
    if split == "test" and trades < MIN_TEST_TRADES:
        return "insufficient_samples"
    if split == "test" and trades < LOW_CONFIDENCE_TRADES:
        return "low_confidence"
    return "enough_samples"


def _win_rate_stats(split: str, trades: int, win_rate: float, trial_count: int) -> Dict[str, Any]:
    if trades <= 0 or pd.isna(win_rate):
        return {
            "win_rate_standard_error": np.nan,
            "win_rate_ci95_low": np.nan,
            "win_rate_ci95_high": np.nan,
            "win_rate_p_value_vs_50": np.nan,
            "win_rate_p_value_bonferroni": np.nan,
            "multiple_testing_status": "not_tested",
        }
    if split == "test" and trades < MIN_TEST_TRADES:
        return {
            "win_rate_standard_error": np.nan,
            "win_rate_ci95_low": np.nan,
            "win_rate_ci95_high": np.nan,
            "win_rate_p_value_vs_50": np.nan,
            "win_rate_p_value_bonferroni": np.nan,
            "multiple_testing_status": "not_tested",
        }
    se = math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / trades)
    ci_low = max(0.0, win_rate - 1.96 * se)
    ci_high = min(1.0, win_rate + 1.96 * se)
    if se == 0:
        p_value = 0.0 if win_rate != 0.5 else 1.0
    else:
        z = (win_rate - 0.5) / se
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    p_bonf = min(1.0, p_value * trial_count)
    if split == "test" and p_bonf < 0.05 and win_rate > 0.5:
        status = "passes_bonferroni"
    elif split == "test" and p_bonf < 0.05 and win_rate < 0.5:
        status = "significant_below_50"
    else:
        status = "not_significant"
    return {
        "win_rate_standard_error": se,
        "win_rate_ci95_low": ci_low,
        "win_rate_ci95_high": ci_high,
        "win_rate_p_value_vs_50": p_value,
        "win_rate_p_value_bonferroni": p_bonf,
        "multiple_testing_status": status,
    }


def build_monthly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    work["month"] = pd.to_datetime(work["exit_datetime"]).dt.to_period("M").astype(str)
    rows: List[Dict[str, Any]] = []
    for keys, part in work.groupby(["case_id", "combo_name", "split", "month"], sort=False):
        case_id, combo_name, split, month = keys
        row = _metrics(part, split)
        row["case_id"] = case_id
        row["combo_name"] = combo_name
        row["month"] = month
        row["side"] = part["side"].iloc[0]
        for col in ["time_barrier_bars", "stop_atr_multiple", "take_profit_multiple", "kline_mode", "space_filter_on"]:
            row[col] = part[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(trades: pd.DataFrame, parameter_grid: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groups = {(case_id, split): part for (case_id, split), part in trades.groupby(["case_id", "split"], sort=False)} if not trades.empty else {}
    all_groups = {case_id: part for case_id, part in trades.groupby("case_id", sort=False)} if not trades.empty else {}
    monthly_test = monthly[monthly["split"] == "test"] if not monthly.empty else pd.DataFrame()
    positive_share = monthly_test.groupby("case_id")["avg_net_return"].apply(lambda x: float((x > 0).mean())) if not monthly_test.empty else pd.Series(dtype=float)
    trial_count = len(parameter_grid)

    for _, case in parameter_grid.iterrows():
        for split in ("train", "test", "all"):
            part = all_groups.get(case["case_id"], pd.DataFrame()) if split == "all" else groups.get((case["case_id"], split), pd.DataFrame())
            row = _metrics(part, split)
            row.update(case.to_dict())
            row["sample_status"] = _sample_status(split, int(row["trades"]))
            row["positive_month_share"] = positive_share.get(case["case_id"], np.nan)
            row.update(_win_rate_stats(split, int(row["trades"]), row["win_rate"], trial_count))
            row["effectiveness"] = _effectiveness(row)
            rows.append(row)
    return pd.DataFrame(rows)


def _effectiveness(row: Dict[str, Any]) -> str:
    if row["split"] != "test":
        return ""
    if row["sample_status"] in {"insufficient_samples", "low_confidence", "no_samples"}:
        return row["sample_status"]
    if row["win_rate"] > 0.5 and row["avg_net_return"] > 0 and row["total_net_return_sum"] > 0:
        return "absolute_effective"
    if row["avg_gross_return"] > 0 and row["avg_net_return"] <= 0:
        return "cost_failed"
    return "ineffective"


def build_exit_reason_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for keys, part in trades.groupby(["case_id", "combo_name", "split"], sort=False):
        case_id, combo_name, split = keys
        total = len(part)
        counts = part["exit_reason"].value_counts()
        base = {
            "case_id": case_id,
            "combo_name": combo_name,
            "side": part["side"].iloc[0],
            "split": split,
            "time_barrier_bars": part["time_barrier_bars"].iloc[0],
            "stop_atr_multiple": part["stop_atr_multiple"].iloc[0],
            "take_profit_multiple": part["take_profit_multiple"].iloc[0],
            "kline_mode": part["kline_mode"].iloc[0],
            "space_filter_on": part["space_filter_on"].iloc[0],
            "trades": total,
            "stop_loss_rate": counts.get("stop_loss", 0) / total if total else np.nan,
            "take_profit_rate": counts.get("take_profit", 0) / total if total else np.nan,
            "time_barrier_rate": counts.get("time_barrier", 0) / total if total else np.nan,
            "take_profit_to_stop_count_ratio": counts.get("take_profit", 0) / counts.get("stop_loss", np.nan) if counts.get("stop_loss", 0) else np.nan,
            "avg_stop_loss_net_return": part.loc[part["exit_reason"] == "stop_loss", "net_return"].mean(),
            "avg_take_profit_net_return": part.loc[part["exit_reason"] == "take_profit", "net_return"].mean(),
            "avg_time_barrier_net_return": part.loc[part["exit_reason"] == "time_barrier", "net_return"].mean(),
            "avg_stop_distance_ticks": part["stop_distance_ticks_barrier"].mean(),
            "avg_take_profit_distance_ticks": part["take_profit_distance_ticks_barrier"].mean(),
            "avg_holding_bars": part["holding_bars"].mean(),
        }
        rows.append({**base, "exit_reason": np.nan, "exit_count": total, "exit_rate": 1.0})
        for reason, reason_part in part.groupby("exit_reason"):
            rows.append(
                {
                    **base,
                    "exit_reason": reason,
                    "exit_count": len(reason_part),
                    "exit_rate": len(reason_part) / total if total else np.nan,
                    "avg_net_return": reason_part["net_return"].mean(),
                }
            )
    return pd.DataFrame(rows)


def add_plateau_scores(summary: pd.DataFrame, parameter_grid: pd.DataFrame) -> pd.DataFrame:
    train = summary[summary["split"] == "train"].copy()
    time_order = {value: i for i, value in enumerate(sorted(parameter_grid["time_barrier_bars"].unique()))}
    stop_order = {value: i for i, value in enumerate(sorted(parameter_grid["stop_atr_multiple"].unique()))}
    take_order = {value: i for i, value in enumerate(sorted(parameter_grid["take_profit_multiple"].unique()))}
    train["time_idx"] = train["time_barrier_bars"].map(time_order)
    train["stop_idx"] = train["stop_atr_multiple"].map(stop_order)
    train["take_idx"] = train["take_profit_multiple"].map(take_order)

    rows: List[Dict[str, Any]] = []
    for idx, row in train.iterrows():
        neighbors = train[
            (train["kline_mode"] == row["kline_mode"])
            & (train["space_filter_on"] == row["space_filter_on"])
            & ((train["time_idx"] - row["time_idx"]).abs() <= 1)
            & ((train["stop_idx"] - row["stop_idx"]).abs() <= 1)
            & ((train["take_idx"] - row["take_idx"]).abs() <= 1)
        ]
        net = pd.to_numeric(neighbors["avg_net_return"], errors="coerce").dropna()
        plateau_mean = float(net.mean()) if len(net) else np.nan
        plateau_std = float(net.std(ddof=1)) if len(net) >= 2 else 0.0
        plateau_positive = float((net > 0).mean()) if len(net) else np.nan
        stability_penalty = plateau_std + abs(plateau_mean) * 0.1 + 1e-12
        score = plateau_mean * plateau_positive * math.log1p(max(float(row["trades"]), 1.0)) / stability_penalty if pd.notna(plateau_mean) and pd.notna(plateau_positive) else np.nan
        rows.append(
            {
                "case_id": row["case_id"],
                "plateau_cases": int(len(neighbors)),
                "plateau_mean_avg_net_return": plateau_mean,
                "plateau_std_avg_net_return": plateau_std,
                "plateau_positive_share": plateau_positive,
                "plateau_score": score,
            }
        )
    plateau = pd.DataFrame(rows)
    return summary.merge(plateau, on="case_id", how="left")


def build_parameter_ranking(summary: pd.DataFrame, max_rank: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = summary[(summary["split"] == "train") & (summary["trades"] >= LOW_CONFIDENCE_TRADES)].copy()
    train = train.sort_values(["plateau_score", "plateau_mean_avg_net_return", "avg_net_return"], ascending=False)
    ranking = train.head(max_rank).copy()
    ranking.insert(0, "train_rank", range(1, len(ranking) + 1))

    test = summary[summary["split"] == "test"].copy()
    train_cols = [
        "train_rank",
        "case_id",
        "avg_net_return",
        "total_net_return_sum",
        "plateau_mean_avg_net_return",
        "plateau_positive_share",
        "plateau_score",
    ]
    # 验证表保留训练集的参数平原指标；测试集只补测试表现，避免 merge 后
    # 出现 plateau_*_x / plateau_*_y 这种含义不清的列名。
    test_cols = [
        col
        for col in test.columns
        if col
        not in {
            "plateau_cases",
            "plateau_mean_avg_net_return",
            "plateau_std_avg_net_return",
            "plateau_positive_share",
            "plateau_score",
        }
    ]
    verification = ranking[train_cols].rename(
        columns={
            "avg_net_return": "train_avg_net_return",
            "total_net_return_sum": "train_total_net_return_sum",
        }
    ).merge(test[test_cols], on="case_id", how="left")
    train_good = (
        (verification["train_avg_net_return"] > 0)
        & (verification["train_total_net_return_sum"] > 0)
        & (verification["plateau_mean_avg_net_return"] > 0)
        & (verification["plateau_positive_share"] >= 0.5)
    )
    test_good = (verification["avg_net_return"] > 0) & (verification["total_net_return_sum"] > 0)
    verification["recommendation_status"] = np.where(
        train_good & test_good,
        "train_test_consistent",
        np.where(train_good & ~test_good, "not_verified", "train_not_passed"),
    )

    known = set(ranking["case_id"])
    test_best = test.sort_values(["avg_net_return", "total_net_return_sum", "win_rate"], ascending=False).head(max_rank).copy()
    test_only = test_best[~test_best["case_id"].isin(known)].copy()
    if not test_only.empty:
        test_only.insert(0, "train_rank", np.nan)
        test_only["recommendation_status"] = "test_only"
        verification = pd.concat([verification, test_only], ignore_index=True, sort=False)
    return ranking, verification


def _mean_summary(summary: pd.DataFrame, group_cols: List[str], exit_diag: pd.DataFrame) -> pd.DataFrame:
    test = summary[summary["split"] == "test"]
    agg = (
        test.groupby(group_cols)
        .agg(
            cases=("case_id", "nunique"),
            mean_win_rate=("win_rate", "mean"),
            mean_avg_net_return=("avg_net_return", "mean"),
            median_avg_net_return=("avg_net_return", "median"),
            mean_total_net_return=("total_net_return_sum", "mean"),
        )
        .reset_index()
    )
    if not exit_diag.empty:
        exit_base = exit_diag[(exit_diag["split"] == "test") & (exit_diag["exit_reason"].isna())]
        exit_agg = (
            exit_base.groupby(group_cols)
            .agg(
                mean_take_profit_rate=("take_profit_rate", "mean"),
                mean_stop_loss_rate=("stop_loss_rate", "mean"),
                mean_time_barrier_rate=("time_barrier_rate", "mean"),
            )
            .reset_index()
        )
        agg = agg.merge(exit_agg, on=group_cols, how="left")
    return agg


def build_time_barrier_diagnostics(summary: pd.DataFrame, exit_diag: pd.DataFrame) -> pd.DataFrame:
    out = _mean_summary(summary, ["time_barrier_bars"], exit_diag)
    out["diagnosis"] = ""
    base = out[out["time_barrier_bars"] == 5]
    if not base.empty:
        base_net = float(base.iloc[0]["mean_avg_net_return"])
        for idx, row in out.iterrows():
            if row["time_barrier_bars"] in {24, 36} and row["mean_avg_net_return"] > base_net:
                out.at[idx, "diagnosis"] = "长时间屏障改善平均净收益，原5Bar可能偏短"
    return out


def build_stop_take_diagnostics(summary: pd.DataFrame, exit_diag: pd.DataFrame) -> pd.DataFrame:
    out = _mean_summary(summary, ["stop_atr_multiple", "take_profit_multiple"], exit_diag)
    out["diagnosis"] = ""
    for idx, row in out.iterrows():
        notes = []
        if row["stop_atr_multiple"] == 0.5 and row.get("mean_stop_loss_rate", np.nan) > 0.35:
            notes.append("0.5ATR止损偏紧")
        if row["take_profit_multiple"] >= 2.0 and row.get("mean_take_profit_rate", np.nan) < 0.1:
            notes.append("2R以上止盈触发偏低")
        out.at[idx, "diagnosis"] = "；".join(notes)
    return out


def build_kline_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    test = summary[summary["split"] == "test"]
    rows: List[Dict[str, Any]] = []
    group_cols = ["time_barrier_bars", "stop_atr_multiple", "take_profit_multiple", "space_filter_on"]
    for _, part in test.groupby(group_cols):
        baseline = part[part["kline_mode"] == "none"]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for _, row in part.iterrows():
            rows.append(
                {
                    "time_barrier_bars": row["time_barrier_bars"],
                    "stop_atr_multiple": row["stop_atr_multiple"],
                    "take_profit_multiple": row["take_profit_multiple"],
                    "space_filter_on": row["space_filter_on"],
                    "current_mode": row["kline_mode"],
                    "benchmark_mode": "none",
                    "trades": row["trades"],
                    "win_rate": row["win_rate"],
                    "avg_net_return": row["avg_net_return"],
                    "total_net_return_sum": row["total_net_return_sum"],
                    "win_rate_gain_vs_benchmark": row["win_rate"] - base["win_rate"],
                    "net_gain": row["avg_net_return"] - base["avg_net_return"],
                    "total_gain": row["total_net_return_sum"] - base["total_net_return_sum"],
                    "diagnosis": "K线过滤有害" if row["kline_mode"] != "none" and row["avg_net_return"] < base["avg_net_return"] and row["win_rate"] < base["win_rate"] else "",
                }
            )
    return pd.DataFrame(rows)


def build_space_filter_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    test = summary[summary["split"] == "test"]
    rows: List[Dict[str, Any]] = []
    group_cols = ["time_barrier_bars", "stop_atr_multiple", "take_profit_multiple", "kline_mode"]
    for _, part in test.groupby(group_cols):
        off = part[part["space_filter_on"] == False]
        on = part[part["space_filter_on"] == True]
        if off.empty or on.empty:
            continue
        base = off.iloc[0]
        row = on.iloc[0]
        diagnosis = ""
        if row["trades"] < MIN_TEST_TRADES:
            diagnosis = "空间过滤导致样本不足"
        elif row["win_rate"] > base["win_rate"] and row["total_net_return_sum"] < base["total_net_return_sum"]:
            diagnosis = "空间过滤提高胜率但不可扩展"
        elif row["win_rate"] < base["win_rate"] and row["avg_net_return"] < base["avg_net_return"]:
            diagnosis = "空间过滤有害"
        rows.append(
            {
                "time_barrier_bars": row["time_barrier_bars"],
                "stop_atr_multiple": row["stop_atr_multiple"],
                "take_profit_multiple": row["take_profit_multiple"],
                "kline_mode": row["kline_mode"],
                "trades": row["trades"],
                "win_rate": row["win_rate"],
                "avg_net_return": row["avg_net_return"],
                "total_net_return_sum": row["total_net_return_sum"],
                "benchmark_trades": base["trades"],
                "benchmark_win_rate": base["win_rate"],
                "benchmark_avg_net_return": base["avg_net_return"],
                "benchmark_total_net_return_sum": base["total_net_return_sum"],
                "win_rate_gain_vs_benchmark": row["win_rate"] - base["win_rate"],
                "net_gain": row["avg_net_return"] - base["avg_net_return"],
                "total_gain": row["total_net_return_sum"] - base["total_net_return_sum"],
                "diagnosis": diagnosis,
            }
        )
    return pd.DataFrame(rows)


def _audit_row(severity: str, check_id: str, status: str, message: str, row_count: int = 0, detail: str = "") -> Dict[str, Any]:
    return {"severity": severity, "check_id": check_id, "status": status, "message": message, "row_count": int(row_count), "detail": detail}


def run_audit(features: pd.DataFrame, trades: pd.DataFrame, parameter_grid: pd.DataFrame, stage2_config: Stage2Config, side: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append(_audit_row("BLOCKER", "parameter_cases", "PASS" if len(parameter_grid) == 384 else "WARN", "默认参数网格应为384组；小网格测试可忽略", len(parameter_grid)))
    duplicate_count = int(features.duplicated(["contract", "datetime"]).sum())
    rows.append(_audit_row("BLOCKER", "duplicate_contract_datetime", "PASS" if duplicate_count == 0 else "FAIL", "contract+datetime 不重复", duplicate_count))
    required = (
        ["trend_filter_pass", "pullback_structure_pass", "flow_confirm_pass", "space_filter_pass"]
        if side == "long"
        else ["short_trend_filter_pass", "short_pullback_structure_pass", "short_flow_confirm_pass", "short_space_filter_pass"]
    )
    missing = [col for col in required if col not in features.columns]
    rows.append(_audit_row("BLOCKER", "required_signal_columns", "PASS" if not missing else "FAIL", "扫描所需信号列完整", len(missing), ",".join(missing)))
    if trades.empty:
        rows.append(_audit_row("WARNING", "no_trades", "WARN", "扫描没有生成任何交易"))
        return pd.DataFrame(rows)
    boundary_bad = int(((trades["split"] == "train") & (pd.to_datetime(trades["exit_datetime"]) >= stage2_config.test_start)).sum())
    rows.append(_audit_row("BLOCKER", "train_test_boundary", "PASS" if boundary_bad == 0 else "FAIL", "训练集交易不穿越测试边界", boundary_bad))
    side_bad = int((trades["side"] != side).sum())
    rows.append(_audit_row("BLOCKER", "side_route", "PASS" if side_bad == 0 else "FAIL", f"扫描实验只包含{SIDE_LABELS.get(side, side)}", side_bad))

    expected_gross = np.where(
        trades["side"] == "long",
        trades["exit_price"] / trades["entry_price"] - 1.0,
        1.0 - trades["exit_price"] / trades["entry_price"],
    )
    return_bad = int((np.abs(trades["gross_return"].to_numpy(dtype=float) - expected_gross) > 1e-12).sum())
    rows.append(_audit_row("BLOCKER", "linear_futures_return_formula", "PASS" if return_bad == 0 else "FAIL", "线性期货多空收益公式逐笔复算正确", return_bad))

    if side == "long":
        barrier_bad = int(
            (
                (trades["stop_price_barrier"] > trades["entry_price"])
                | (trades["take_profit_price_barrier"] < trades["entry_price"])
            ).sum()
        )
        barrier_message = "多头屏障满足 stop<=entry<=take_profit"
    else:
        barrier_bad = int(
            (
                (trades["take_profit_price_barrier"] > trades["entry_price"])
                | (trades["stop_price_barrier"] < trades["entry_price"])
            ).sum()
        )
        barrier_message = "空头屏障满足 take_profit<=entry<=stop"
    rows.append(_audit_row("BLOCKER", "side_barrier_geometry", "PASS" if barrier_bad == 0 else "FAIL", barrier_message, barrier_bad))

    overlap_total = 0
    for _, part in trades.groupby("case_id", sort=False):
        overlap = check_no_overlap(part)
        overlap_total += int(overlap["overlap_count"].sum()) if not overlap.empty else 0
    rows.append(_audit_row("BLOCKER", "trade_overlap", "PASS" if overlap_total == 0 else "FAIL", "每个方向每个参数组合内同合约交易不重叠", overlap_total))
    return pd.DataFrame(rows)


def _format_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return "无"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    if isinstance(value, (bool, np.bool_)):
        return VALUE_LABELS.get(bool(value), str(value))
    text = str(value)
    return VALUE_LABELS.get(text, text).replace("\n", " ")


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "无数据。"
    view = df.head(max_rows).copy()
    columns = list(view.columns)
    header = [COLUMN_LABELS.get(col, col) for col in columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _answer_lines(summary: pd.DataFrame, exit_diag: pd.DataFrame, kline_ablation: pd.DataFrame, space_ablation: pd.DataFrame, verification: pd.DataFrame, side: str) -> List[str]:
    lines: List[str] = []
    time_diag = build_time_barrier_diagnostics(summary, exit_diag)
    improved_time = time_diag[time_diag["diagnosis"].astype(str).str.len() > 0] if not time_diag.empty else pd.DataFrame()
    lines.append(f"- 5 Bar 时间屏障：{'偏短迹象明显' if not improved_time.empty else '未发现稳定偏短证据'}。")
    exit_base = exit_diag[(exit_diag["split"] == "test") & (exit_diag["exit_reason"].isna())] if not exit_diag.empty else pd.DataFrame()
    if not exit_base.empty:
        far_tp = exit_base[(exit_base["take_profit_multiple"] >= 2.0) & (exit_base["take_profit_rate"] < 0.1)]
        tight_stop = exit_base[(exit_base["stop_atr_multiple"] == 0.5) & (exit_base["stop_loss_rate"] > 0.35)]
        lines.append(f"- 2R 止盈：{'触发率偏低，需要降低止盈倍数或拉长时间屏障验证' if not far_tp.empty else '未显示普遍过远'}。")
        lines.append(f"- 0.5 ATR 止损：{'多数组合止损偏频，存在过紧风险' if not tight_stop.empty else '没有普遍止损过频'}。")
    harmful_kline = kline_ablation[kline_ablation["diagnosis"] == "K线过滤有害"] if not kline_ablation.empty else pd.DataFrame()
    strong = kline_ablation[kline_ablation["current_mode"] == "strong_close_only"] if not kline_ablation.empty else pd.DataFrame()
    full = kline_ablation[kline_ablation["current_mode"] == "full_patterns"] if not kline_ablation.empty else pd.DataFrame()
    strong_better = bool(strong["net_gain"].median() > full["net_gain"].median()) if not strong.empty and not full.empty else False
    lines.append(f"- 完整 K线形态：{'存在破坏样本质量的证据' if not harmful_kline.empty else '未构成稳定负贡献'}。")
    lines.append(f"- 顺方向收盘 vs 完整形态：{'顺方向收盘口径更健康' if strong_better else '没有稳定优于完整形态'}。")
    bad_space = space_ablation[space_ablation["diagnosis"].astype(str).str.len() > 0] if not space_ablation.empty else pd.DataFrame()
    lines.append(f"- 空间过滤：{'样本耗竭或不可扩展风险明显' if not bad_space.empty else '未显示稳定拖累'}。")
    verified = verification[verification["recommendation_status"] == "train_test_consistent"] if not verification.empty else pd.DataFrame()
    if verified.empty:
        lines.append("- 参数平原：训练集平原评分前排没有在测试集形成明确正收益验证，暂不建议固化参数。")
        lines.append(f"- 阶段三 walk-forward：{SIDE_LABELS.get(side, side)}可以纳入候选，但应作为否证性验证，不应直接继承单次切分最优参数。")
    else:
        best = verified.sort_values(["avg_net_return", "total_net_return_sum"], ascending=False).iloc[0]
        lines.append(
            f"- 最健康参数区域：时间屏障 `{best['time_barrier_bars']}`，止损 `{best['stop_atr_multiple']}` ATR，"
            f"止盈 `{best['take_profit_multiple']}`R，K线 `{VALUE_LABELS.get(best['kline_mode'], best['kline_mode'])}`，"
            f"空间过滤 `{VALUE_LABELS.get(best['space_filter_on'], best['space_filter_on'])}`。"
        )
        lines.append(f"- 阶段三 walk-forward：{SIDE_LABELS.get(side, side)}值得纳入候选，但必须继续验证参数稳定性。")
    return lines


def write_report(
    output_path: Path,
    side: str,
    features: pd.DataFrame,
    stage2_config: Stage2Config,
    parameter_grid: pd.DataFrame,
    summary: pd.DataFrame,
    exit_diag: pd.DataFrame,
    ranking: pd.DataFrame,
    verification: pd.DataFrame,
    kline_ablation: pd.DataFrame,
    space_ablation: pd.DataFrame,
    time_diag: pd.DataFrame,
    stop_take_diag: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    blockers = audit[(audit["severity"] == "BLOCKER") & (audit["status"] == "FAIL")]
    test = summary[summary["split"] == "test"].sort_values(["avg_net_return", "total_net_return_sum", "win_rate"], ascending=False)
    lines: List[str] = []
    side_label = SIDE_LABELS.get(side, side)
    lines.append(f"# {side_label}趋势回调参数扫描与退出原因诊断报告\n")
    lines.append("## 结论摘要")
    lines.append(f"- 方向：`{side_label}`；参数组合数：`{len(parameter_grid)}`，统一策略为 `{STRATEGY_LABELS.get(side, STRATEGY_NAME)}`；K线和空间仅作为参数开关。")
    lines.append(f"- 数据范围：`{features['datetime'].min()}` 到 `{features['datetime'].max()}`；合约数 `{features['contract'].nunique()}`；样本行数 `{len(features)}`。")
    lines.append(f"- 训练/测试切分：训练集截至 `{stage2_config.train_end}`，测试集从 `{stage2_config.test_start}` 开始。")
    lines.append(f"- 审计阻断项：`{len(blockers)}`；胜率显著性已按 `{len(parameter_grid)}` 次试验做 Bonferroni 修正。")
    lines.extend(_answer_lines(summary, exit_diag, kline_ablation, space_ablation, verification, side))
    lines.append("")

    main_cols = [
        "case_id",
        "side",
        "time_barrier_bars",
        "stop_atr_multiple",
        "take_profit_multiple",
        "kline_mode",
        "space_filter_on",
        "trades",
        "win_rate",
        "avg_net_return",
        "total_net_return_sum",
        "sample_status",
        "multiple_testing_status",
        "effectiveness",
    ]
    lines.append("## 测试集最优观察")
    lines.append(_md_table(test[main_cols], max_rows=20))
    lines.append("")

    verify_cols = [
        "train_rank",
        "case_id",
        "side",
        "time_barrier_bars",
        "stop_atr_multiple",
        "take_profit_multiple",
        "kline_mode",
        "space_filter_on",
        "train_avg_net_return",
        "train_total_net_return_sum",
        "trades",
        "win_rate",
        "avg_net_return",
        "total_net_return_sum",
        "plateau_mean_avg_net_return",
        "plateau_positive_share",
        "plateau_score",
        "recommendation_status",
    ]
    lines.append("## 参数平原排名与测试验证")
    lines.append(_md_table(verification[[col for col in verify_cols if col in verification.columns]], max_rows=20))
    lines.append("")

    exit_cols = [
        "case_id",
        "side",
        "time_barrier_bars",
        "stop_atr_multiple",
        "take_profit_multiple",
        "kline_mode",
        "space_filter_on",
        "trades",
        "stop_loss_rate",
        "take_profit_rate",
        "time_barrier_rate",
        "take_profit_to_stop_count_ratio",
        "avg_stop_loss_net_return",
        "avg_take_profit_net_return",
        "avg_time_barrier_net_return",
        "avg_stop_distance_ticks",
        "avg_take_profit_distance_ticks",
    ]
    exit_base = exit_diag[(exit_diag["split"] == "test") & (exit_diag["exit_reason"].isna())].sort_values("take_profit_rate", ascending=False) if not exit_diag.empty else pd.DataFrame()
    lines.append("## 退出原因诊断")
    lines.append(_md_table(exit_base[[col for col in exit_cols if col in exit_base.columns]], max_rows=25))
    lines.append("")

    lines.append("## 时间屏障诊断")
    lines.append(_md_table(time_diag, max_rows=20))
    lines.append("")

    lines.append("## 止损止盈诊断")
    lines.append(_md_table(stop_take_diag, max_rows=30))
    lines.append("")

    kline_cols = ["time_barrier_bars", "stop_atr_multiple", "take_profit_multiple", "space_filter_on", "current_mode", "trades", "win_rate", "avg_net_return", "win_rate_gain_vs_benchmark", "net_gain", "diagnosis"]
    lines.append("## K线过滤消融")
    lines.append(_md_table(kline_ablation[[col for col in kline_cols if col in kline_ablation.columns]], max_rows=30))
    lines.append("")

    space_cols = ["time_barrier_bars", "stop_atr_multiple", "take_profit_multiple", "kline_mode", "trades", "win_rate", "avg_net_return", "benchmark_trades", "benchmark_win_rate", "benchmark_avg_net_return", "net_gain", "total_gain", "diagnosis"]
    lines.append("## 空间过滤消融")
    lines.append(_md_table(space_ablation[[col for col in space_cols if col in space_ablation.columns]], max_rows=30))
    lines.append("")

    lines.append("## 审计")
    lines.append(_md_table(audit, max_rows=50))
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _side_parameter_grid(parameter_grid: pd.DataFrame, side: str) -> pd.DataFrame:
    out = parameter_grid.copy()
    out["side"] = side
    return out


def run_side(
    side: str,
    features: pd.DataFrame,
    stage2_config: Stage2Config,
    base_config: TrendPullbackConfig,
    parameter_grid: pd.DataFrame,
    output_dir: Path,
    max_train_rank: int,
    report_tag: str = "",
) -> Dict[str, Any]:
    side_grid = _side_parameter_grid(parameter_grid, side)
    trade_frames: List[pd.DataFrame] = []
    for _, case in side_grid.iterrows():
        signal = _signal_mask(features, side, str(case["kline_mode"]), bool(case["space_filter_on"]))
        trades = _generate_case_trades(features, signal, stage2_config, case, side)
        if not trades.empty:
            trade_frames.append(trades)

    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    monthly = build_monthly_metrics(trades_all)
    summary = build_summary(trades_all, side_grid, monthly)
    summary = add_plateau_scores(summary, side_grid)
    exit_diag = build_exit_reason_diagnostics(trades_all)
    ranking, verification = build_parameter_ranking(summary, max_train_rank)
    kline_ablation = build_kline_ablation(summary)
    space_ablation = build_space_filter_ablation(summary)
    time_diag = build_time_barrier_diagnostics(summary, exit_diag)
    stop_take_diag = build_stop_take_diagnostics(summary, exit_diag)
    audit = run_audit(features, trades_all, side_grid, stage2_config, side)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(side_grid, output_dir / "trend_pullback_parameter_grid.csv")
    write_csv(summary, output_dir / "trend_pullback_sweep_summary.csv")
    write_csv(trades_all, output_dir / "trend_pullback_sweep_trades.csv")
    write_csv(exit_diag, output_dir / "trend_pullback_exit_reason_diagnostics.csv")
    write_csv(ranking, output_dir / "trend_pullback_parameter_ranking_train.csv")
    write_csv(verification, output_dir / "trend_pullback_parameter_verification_test.csv")
    write_csv(kline_ablation, output_dir / "trend_pullback_kline_ablation.csv")
    write_csv(space_ablation, output_dir / "trend_pullback_space_filter_ablation.csv")
    write_csv(time_diag, output_dir / "trend_pullback_time_barrier_diagnostics.csv")
    write_csv(stop_take_diag, output_dir / "trend_pullback_stop_take_diagnostics.csv")
    write_csv(monthly, output_dir / "trend_pullback_sweep_monthly.csv")
    write_csv(audit, output_dir / "trend_pullback_sweep_audit.csv")
    report_name = "trend_pullback_sweep_report_zh.md"
    if report_tag:
        report_name = f"trend_pullback_sweep_{side}_report_{report_tag}_zh.md"
    write_report(
        output_dir / report_name,
        side,
        features,
        stage2_config,
        side_grid,
        summary,
        exit_diag,
        ranking,
        verification,
        kline_ablation,
        space_ablation,
        time_diag,
        stop_take_diag,
        audit,
    )

    blockers = audit[(audit["severity"] == "BLOCKER") & (audit["status"] == "FAIL")]
    return {
        "side": side,
        "output_dir": output_dir,
        "parameter_grid": side_grid,
        "trades": trades_all,
        "monthly": monthly,
        "summary": summary,
        "exit_diag": exit_diag,
        "ranking": ranking,
        "verification": verification,
        "kline_ablation": kline_ablation,
        "space_ablation": space_ablation,
        "time_diag": time_diag,
        "stop_take_diag": stop_take_diag,
        "audit": audit,
        "audit_blockers": int(len(blockers)),
        "trade_count": int(len(trades_all)),
    }


def _best_parameter_row(side_result: Dict[str, Any]) -> Tuple[pd.Series, str]:
    verification = side_result["verification"]
    verified = verification[verification["recommendation_status"] == "train_test_consistent"] if not verification.empty else pd.DataFrame()
    if not verified.empty:
        return verified.sort_values(["avg_net_return", "total_net_return_sum"], ascending=False).iloc[0], "训练测试一致改善"
    test = side_result["summary"][side_result["summary"]["split"] == "test"]
    if not test.empty:
        return test.sort_values(["avg_net_return", "total_net_return_sum", "win_rate"], ascending=False).iloc[0], "测试集观察最优"
    return pd.Series(dtype=object), "无可用参数"


def _grid_similarity(value_a: Any, value_b: Any, ordered_values: List[Any]) -> float:
    if pd.isna(value_a) or pd.isna(value_b):
        return np.nan
    if len(ordered_values) <= 1:
        return 1.0
    index = {value: i for i, value in enumerate(ordered_values)}
    distance = abs(index[value_a] - index[value_b]) / (len(ordered_values) - 1)
    return float(1.0 - distance)


def build_long_short_asymmetry(side_results: Dict[str, Dict[str, Any]], parameter_grid: pd.DataFrame) -> pd.DataFrame:
    if "long" not in side_results or "short" not in side_results:
        return pd.DataFrame()
    long_row, long_source = _best_parameter_row(side_results["long"])
    short_row, short_source = _best_parameter_row(side_results["short"])
    if long_row.empty or short_row.empty:
        return pd.DataFrame()

    time_values = sorted(parameter_grid["time_barrier_bars"].unique().tolist())
    stop_values = sorted(parameter_grid["stop_atr_multiple"].unique().tolist())
    take_values = sorted(parameter_grid["take_profit_multiple"].unique().tolist())
    kline_similarity = 1.0 if long_row["kline_mode"] == short_row["kline_mode"] else 0.0
    space_similarity = 1.0 if bool(long_row["space_filter_on"]) == bool(short_row["space_filter_on"]) else 0.0
    similarities = {
        "time_barrier_similarity": _grid_similarity(long_row["time_barrier_bars"], short_row["time_barrier_bars"], time_values),
        "stop_similarity": _grid_similarity(long_row["stop_atr_multiple"], short_row["stop_atr_multiple"], stop_values),
        "take_profit_similarity": _grid_similarity(long_row["take_profit_multiple"], short_row["take_profit_multiple"], take_values),
        "kline_similarity": kline_similarity,
        "space_similarity": space_similarity,
    }
    values = [value for value in similarities.values() if pd.notna(value)]
    overlap = float(np.mean(values)) if values else np.nan
    asymmetry = 1.0 - overlap if pd.notna(overlap) else np.nan
    diagnosis = "多空参数高度相似，可继续观察共用参数"
    if pd.notna(asymmetry) and asymmetry >= 0.5:
        diagnosis = "多空参数明显非对称，不建议固化同一套参数"
    if short_row["time_barrier_bars"] < long_row["time_barrier_bars"]:
        diagnosis += "；空头最佳时间屏障更短，做空动能速度更快"
    elif short_row["time_barrier_bars"] > long_row["time_barrier_bars"]:
        diagnosis += "；空头最佳时间屏障更长，做空并未体现更快动能"

    row = {
        "long_case_id": long_row["case_id"],
        "short_case_id": short_row["case_id"],
        "long_selection_source": long_source,
        "short_selection_source": short_source,
        "long_time_barrier_bars": long_row["time_barrier_bars"],
        "short_time_barrier_bars": short_row["time_barrier_bars"],
        "long_stop_atr_multiple": long_row["stop_atr_multiple"],
        "short_stop_atr_multiple": short_row["stop_atr_multiple"],
        "long_take_profit_multiple": long_row["take_profit_multiple"],
        "short_take_profit_multiple": short_row["take_profit_multiple"],
        "long_kline_mode": long_row["kline_mode"],
        "short_kline_mode": short_row["kline_mode"],
        "long_space_filter_on": long_row["space_filter_on"],
        "short_space_filter_on": short_row["space_filter_on"],
        "parameter_overlap_score": overlap,
        "asymmetry_score": asymmetry,
        "diagnosis": diagnosis,
    }
    row.update(similarities)
    return pd.DataFrame([row])


def build_long_short_comparison(side_results: Dict[str, Dict[str, Any]], total_trial_count: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for side, result in side_results.items():
        test = result["summary"][result["summary"]["split"] == "test"]
        if test.empty:
            continue
        best = test.sort_values(["avg_net_return", "total_net_return_sum", "win_rate"], ascending=False).iloc[0].copy()
        p_value = best.get("win_rate_p_value_vs_50", np.nan)
        summary_p = min(1.0, p_value * total_trial_count) if pd.notna(p_value) else np.nan
        if pd.isna(summary_p):
            status = "not_tested"
        elif summary_p < 0.05 and best["win_rate"] > 0.5:
            status = "passes_bonferroni"
        elif summary_p < 0.05 and best["win_rate"] < 0.5:
            status = "significant_below_50"
        else:
            status = "not_significant"
        rows.append(
            {
                "side": side,
                "case_id": best["case_id"],
                "time_barrier_bars": best["time_barrier_bars"],
                "stop_atr_multiple": best["stop_atr_multiple"],
                "take_profit_multiple": best["take_profit_multiple"],
                "kline_mode": best["kline_mode"],
                "space_filter_on": best["space_filter_on"],
                "trades": best["trades"],
                "win_rate": best["win_rate"],
                "avg_net_return": best["avg_net_return"],
                "total_net_return_sum": best["total_net_return_sum"],
                "sample_status": best["sample_status"],
                "effectiveness": best["effectiveness"],
                "summary_bonferroni_p_value": summary_p,
                "summary_multiple_testing_status": status,
            }
        )
    return pd.DataFrame(rows)


def build_long_short_best_plateau(side_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for side, result in side_results.items():
        verification = result["verification"].copy()
        if verification.empty:
            continue
        verification["side"] = side
        rows.append(verification.head(10))
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_long_short_exit_comparison(side_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for side, result in side_results.items():
        best, source = _best_parameter_row(result)
        if best.empty:
            continue
        exit_diag = result["exit_diag"]
        part = exit_diag[
            (exit_diag["split"] == "test")
            & (exit_diag["case_id"] == best["case_id"])
            & (exit_diag["exit_reason"].isna())
        ]
        if part.empty:
            continue
        row = part.iloc[0].copy()
        row["side"] = side
        row["selection_source"] = source
        rows.append(row)
    return pd.DataFrame(rows)


def build_long_short_audit(side_results: Dict[str, Dict[str, Any]], parameter_grid: pd.DataFrame, sides: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_trials = len(parameter_grid) * len(sides)
    rows.append(_audit_row("BLOCKER", "summary_trial_count", "PASS" if total_trials == len(parameter_grid) * len(sides) else "FAIL", "汇总层Bonferroni试验次数按参数组数*方向数计算", total_trials))
    for side, result in side_results.items():
        blockers = result["audit"][(result["audit"]["severity"] == "BLOCKER") & (result["audit"]["status"] == "FAIL")]
        rows.append(_audit_row("BLOCKER", f"{side}_blockers", "PASS" if blockers.empty else "FAIL", f"{SIDE_LABELS.get(side, side)}审计阻断项", len(blockers)))
    return pd.DataFrame(rows)


def write_long_short_summary_report(
    output_path: Path,
    features: pd.DataFrame,
    stage2_config: Stage2Config,
    parameter_grid: pd.DataFrame,
    sides: List[str],
    comparison: pd.DataFrame,
    best_plateau: pd.DataFrame,
    exit_comparison: pd.DataFrame,
    asymmetry: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    blockers = audit[(audit["severity"] == "BLOCKER") & (audit["status"] == "FAIL")]
    lines: List[str] = []
    lines.append("# 多空趋势回调参数扫描汇总报告\n")
    lines.append("## 结论摘要")
    lines.append(f"- 方向：`{','.join(SIDE_LABELS.get(side, side) for side in sides)}`；单方向参数组合 `{len(parameter_grid)}`，汇总试验次数 `{len(parameter_grid) * len(sides)}`。")
    lines.append(f"- 数据范围：`{features['datetime'].min()}` 到 `{features['datetime'].max()}`；合约数 `{features['contract'].nunique()}`；样本行数 `{len(features)}`。")
    lines.append(f"- 训练/测试切分：训练集截至 `{stage2_config.train_end}`，测试集从 `{stage2_config.test_start}` 开始。")
    lines.append(f"- 审计阻断项：`{len(blockers)}`。")
    if not asymmetry.empty:
        lines.append(f"- 多空非对称性得分：`{asymmetry.iloc[0]['asymmetry_score']:.4f}`；{asymmetry.iloc[0]['diagnosis']}。")
    if not comparison.empty:
        best_side = comparison.sort_values(["avg_net_return", "total_net_return_sum"], ascending=False).iloc[0]
        lines.append(f"- 测试集观察最优方向：`{SIDE_LABELS.get(best_side['side'], best_side['side'])}`，但仍需以训练平原和多重检验共同约束。")
    lines.append("")

    compare_cols = [
        "side",
        "case_id",
        "time_barrier_bars",
        "stop_atr_multiple",
        "take_profit_multiple",
        "kline_mode",
        "space_filter_on",
        "trades",
        "win_rate",
        "avg_net_return",
        "total_net_return_sum",
        "sample_status",
        "effectiveness",
        "summary_bonferroni_p_value",
        "summary_multiple_testing_status",
    ]
    lines.append("## 多空测试集观察最优对比")
    lines.append(_md_table(comparison[[col for col in compare_cols if col in comparison.columns]], max_rows=20))
    lines.append("")

    asym_cols = [
        "parameter_overlap_score",
        "asymmetry_score",
        "time_barrier_similarity",
        "stop_similarity",
        "take_profit_similarity",
        "kline_similarity",
        "space_similarity",
        "diagnosis",
    ]
    lines.append("## 多空非对称性")
    lines.append(_md_table(asymmetry[[col for col in asym_cols if col in asymmetry.columns]], max_rows=5))
    lines.append("")

    exit_cols = [
        "side",
        "selection_source",
        "case_id",
        "time_barrier_bars",
        "stop_atr_multiple",
        "take_profit_multiple",
        "kline_mode",
        "space_filter_on",
        "trades",
        "stop_loss_rate",
        "take_profit_rate",
        "time_barrier_rate",
        "avg_stop_loss_net_return",
        "avg_take_profit_net_return",
        "avg_time_barrier_net_return",
    ]
    lines.append("## 多空退出原因对比")
    lines.append(_md_table(exit_comparison[[col for col in exit_cols if col in exit_comparison.columns]], max_rows=20))
    lines.append("")

    plateau_cols = [
        "side",
        "train_rank",
        "case_id",
        "time_barrier_bars",
        "stop_atr_multiple",
        "take_profit_multiple",
        "kline_mode",
        "space_filter_on",
        "train_avg_net_return",
        "avg_net_return",
        "total_net_return_sum",
        "plateau_mean_avg_net_return",
        "plateau_positive_share",
        "recommendation_status",
    ]
    lines.append("## 多空参数平原前列")
    lines.append(_md_table(best_plateau[[col for col in plateau_cols if col in best_plateau.columns]], max_rows=30))
    lines.append("")

    lines.append("## 汇总审计")
    lines.append(_md_table(audit, max_rows=30))
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    args = args or parse_args()
    base_config = _base_config(args)
    parameter_grid = _parameter_grid(args)
    sides = _parse_sides(args.sides)
    data = load_stage2_pack(base_config)
    stage2_config = make_stage2_config(base_config, data)
    features = build_trend_pullback_features(data, base_config)
    if "short" in sides:
        features = add_short_mirror_features(features, base_config)

    output_dir = base_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    side_results: Dict[str, Dict[str, Any]] = {}
    report_tag = str(getattr(args, "report_tag", "") or "").strip()
    for side in sides:
        side_results[side] = run_side(side, features, stage2_config, base_config, parameter_grid, output_dir / side, int(args.max_train_rank), report_tag)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_long_short_comparison(side_results, len(parameter_grid) * len(sides))
    best_plateau = build_long_short_best_plateau(side_results)
    exit_comparison = build_long_short_exit_comparison(side_results)
    asymmetry = build_long_short_asymmetry(side_results, parameter_grid)
    summary_audit = build_long_short_audit(side_results, parameter_grid, sides)
    write_csv(comparison, summary_dir / "long_short_comparison.csv")
    write_csv(best_plateau, summary_dir / "long_short_best_plateau.csv")
    write_csv(exit_comparison, summary_dir / "long_short_exit_reason_comparison.csv")
    write_csv(asymmetry, summary_dir / "long_short_asymmetry.csv")
    write_csv(summary_audit, summary_dir / "long_short_audit.csv")
    summary_report_name = "trend_pullback_sweep_long_short_summary_zh.md"
    if report_tag:
        summary_report_name = f"trend_pullback_sweep_long_short_summary_{report_tag}_zh.md"
    write_long_short_summary_report(
        summary_dir / summary_report_name,
        features,
        stage2_config,
        parameter_grid,
        sides,
        comparison,
        best_plateau,
        exit_comparison,
        asymmetry,
        summary_audit,
    )

    audit_blockers = sum(result["audit_blockers"] for result in side_results.values())
    summary_blockers = summary_audit[(summary_audit["severity"] == "BLOCKER") & (summary_audit["status"] == "FAIL")]
    return {
        "output_dir": output_dir,
        "contracts": int(features["contract"].nunique()),
        "rows": int(len(features)),
        "parameter_cases": int(len(parameter_grid)),
        "sides": ",".join(sides),
        "train_end": stage2_config.train_end,
        "test_start": stage2_config.test_start,
        "trades": int(sum(result["trade_count"] for result in side_results.values())),
        "audit_blockers": int(audit_blockers + len(summary_blockers)),
    }
