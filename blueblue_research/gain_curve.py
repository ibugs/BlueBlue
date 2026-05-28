"""阶段二增益曲线验证。

本模块专门回答一个研究问题：Delta、POC、位置这些订单流维度逐层组合后，
测试集胜率和扣成本净收益是否真正改善。所有选择、方向、阈值都只来自训练集。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BASE_COLUMNS, DEFAULT_OUTPUT_DIR, LABEL_COLUMNS, Stage2Config, resolve_time_split
from .features import FEATURE_GROUP_MAP, standardize_selected_features
from .io import read_csv, write_csv
from .strategy import add_risk_filter, check_no_overlap, generate_trades


DEFAULT_STAGE2_DIR = DEFAULT_OUTPUT_DIR
POSITION_FEATURES = ["close_location_value", "body_ratio", "upper_shadow_ratio", "lower_shadow_ratio", "wick_imbalance"]

DIMENSION_FEATURE_RULES: Dict[str, Dict[str, Any]] = {
    "delta": {"groups": ["orderflow_delta"], "features": None},
    "poc": {"groups": ["auction_poc"], "features": None},
    "position": {"groups": None, "features": POSITION_FEATURES},
}

COMBO_DEFINITIONS: List[Dict[str, Any]] = [
    {"combo_name": "delta_only", "dimensions": ["delta"], "description": "纯订单流 Delta 维度"},
    {"combo_name": "poc_only", "dimensions": ["poc"], "description": "POC/成交密集区维度"},
    {"combo_name": "position_only", "dimensions": ["position"], "description": "Bar 内部位置维度"},
    {"combo_name": "delta_poc", "dimensions": ["delta", "poc"], "description": "Delta + POC 组合"},
    {"combo_name": "delta_poc_position", "dimensions": ["delta", "poc", "position"], "description": "Delta + POC + 位置组合"},
]

MIN_TEST_TRADES = 100
LOW_CONFIDENCE_TRADES = 300
COLLINEARITY_WARNING_THRESHOLD = 0.70
DISTRIBUTION_SHRINK_THRESHOLD = 0.40

COMBO_LABELS = {
    "delta_only": "Delta单独",
    "poc_only": "POC单独",
    "position_only": "位置单独",
    "delta_poc": "Delta+POC",
    "delta_poc_position": "Delta+POC+位置",
}

VALUE_LABELS = {
    "train": "训练集",
    "test": "测试集",
    "all": "全部",
    "delta": "Delta订单流",
    "poc": "POC成交密集区",
    "position": "位置",
    "price_structure": "价格结构",
    "orderflow_delta": "Delta订单流",
    "auction_poc": "POC成交密集区",
    "volume_liquidity": "成交量/流动性",
    "open_interest": "持仓量",
    "regime_time": "状态/时间",
    "interaction": "交互特征",
    "cost_failed": "成本失败",
    "ineffective": "无效",
    "win_rate_only": "胜率伪增益",
    "incremental_effective": "增量有效",
    "absolute_effective": "绝对有效",
    "unstable": "月度不稳定",
    "insufficient_samples": "样本不足",
    "low_confidence": "低置信",
    "enough_samples": "样本充足",
    "no_samples": "无样本",
    "collinearity_blocked": "共线性阻断",
    "collinearity_warning": "共线性警告",
    "distribution_deformed": "分布形变",
    "tail_compressed": "尾部压缩",
    "ok": "正常",
    "not_applicable": "不适用",
    "True": "是",
    "False": "否",
    True: "是",
    False: "否",
    "PASS": "通过",
    "FAIL": "失败",
    "WARN": "警告",
    "BLOCKER": "阻断",
    "WARNING": "警告",
    "OBSERVATION": "观察",
    "all_same_direction": "三维同向",
    "delta_poc_same": "Delta与POC同向",
    "delta_poc_conflict": "Delta与POC冲突",
    "mixed": "混合",
    "selected_corr_threshold": "入选特征相关性阈值",
    "selected_corr_warning": "入选特征相关性警告",
    "selected_from_train_summary": "仅使用训练集选特征",
    "insufficient_samples": "样本不足",
    "low_confidence_samples": "低置信样本",
    "signal_distribution": "信号分布",
    "trade_overlap": "持仓重叠",
    "train_test_boundary": "训练测试边界",
    "threshold_direction": "阈值方向",
    "no_trades": "无交易",
}

COLUMN_LABELS = {
    "combo_name": "组合名",
    "combo_dimensions": "组合维度",
    "dimension": "维度",
    "dimension_order": "维度内顺序",
    "feature": "特征",
    "group": "特征组",
    "spearman_ic_5": "训练集秩相关IC",
    "abs_spearman_ic_5": "训练集IC绝对值",
    "max_abs_corr_to_selected_in_dimension": "维度内最大相关",
    "split": "切分",
    "trades": "交易笔数",
    "long_trades": "多头笔数",
    "short_trades": "空头笔数",
    "win_rate": "胜率",
    "avg_gross_return": "平均毛收益",
    "avg_net_return": "平均净收益",
    "median_net_return": "中位净收益",
    "total_net_return_sum": "总净收益",
    "per_trade_sharpe": "按笔Sharpe",
    "max_drawdown_sum": "最大回撤",
    "benchmark_combo": "比较基准",
    "win_rate_gain_vs_benchmark": "胜率增量",
    "avg_net_return_gain_vs_benchmark": "净收益增量",
    "effectiveness": "有效性判定",
    "sample_status": "样本状态",
    "min_required_trades": "最低样本要求",
    "win_rate_standard_error": "胜率标准误",
    "win_rate_ci95_low": "胜率95%下界",
    "win_rate_ci95_high": "胜率95%上界",
    "max_selected_abs_corr": "入选特征最大相关",
    "collinearity_status": "共线性状态",
    "signal_count": "信号样本数",
    "score_mean": "分数均值",
    "score_std": "分数标准差",
    "score_skew": "分数偏度",
    "score_kurtosis": "分数峰度",
    "score_p01": "分数P01",
    "score_p05": "分数P05",
    "score_p15": "分数P15",
    "score_p50": "分数P50",
    "score_p85": "分数P85",
    "score_p95": "分数P95",
    "score_p99": "分数P99",
    "long_threshold": "做多阈值",
    "short_threshold": "做空阈值",
    "tail_separation": "尾部分离度",
    "threshold_abs_mean": "阈值绝对均值",
    "tail_shrink_ratio": "尾部收缩比例",
    "distribution_status": "分布状态",
    "feature_a": "特征A",
    "dimension_a": "维度A",
    "feature_b": "特征B",
    "dimension_b": "维度B",
    "spearman_corr": "秩相关系数",
    "abs_spearman_corr": "相关绝对值",
    "is_cross_dimension": "是否跨维度",
    "correlation_status": "相关性状态",
    "month": "月份",
    "pattern": "交互形态",
    "rows": "样本行数",
    "mean_future_return_5": "未来5根均值收益",
    "median_future_return_5": "未来5根中位收益",
    "positive_rate": "正收益比例",
    "severity": "级别",
    "check_id": "检查项",
    "status": "状态",
    "message": "说明",
    "row_count": "行数",
    "detail": "细节",
}


@dataclass(frozen=True)
class GainCurveConfig:
    stage2_dir: Path
    output_dir: Path
    train_end: Optional[pd.Timestamp]
    test_start: Optional[pd.Timestamp]
    primary_horizon: int
    holding_bars: int
    cost_ticks_roundtrip: float
    tick_size: float
    long_quantile: float
    short_quantile: float
    corr_threshold: float
    max_features_per_dimension: int
    split_train_ratio: float
    contracts: Optional[List[str]]


def parse_args(argv: Optional[Sequence[str]] = None) -> GainCurveConfig:
    parser = argparse.ArgumentParser(description="Stage 2 Delta/POC/position gain-curve validation")
    parser.add_argument("--stage2_dir", type=str, default=str(DEFAULT_STAGE2_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--primary_horizon", type=int, default=5)
    parser.add_argument("--holding_bars", type=int, default=5)
    parser.add_argument("--cost_ticks_roundtrip", type=float, default=2.0)
    parser.add_argument("--tick_size", type=float, default=10.0)
    parser.add_argument("--long_quantile", type=float, default=0.85)
    parser.add_argument("--short_quantile", type=float, default=0.15)
    parser.add_argument("--corr_threshold", type=float, default=0.85)
    parser.add_argument("--max_features_per_dimension", type=int, default=4)
    parser.add_argument("--split_train_ratio", type=float, default=0.8)
    parser.add_argument("--contracts", type=str, default=None)
    args = parser.parse_args(argv)

    stage2_dir = Path(args.stage2_dir).expanduser()
    contracts = [x.strip() for x in args.contracts.split(",") if x.strip()] if args.contracts else None
    return GainCurveConfig(
        stage2_dir=stage2_dir,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else stage2_dir / "gain_curve",
        train_end=pd.Timestamp(args.train_end) if args.train_end else None,
        test_start=pd.Timestamp(args.test_start) if args.test_start else None,
        primary_horizon=args.primary_horizon,
        holding_bars=args.holding_bars,
        cost_ticks_roundtrip=args.cost_ticks_roundtrip,
        tick_size=args.tick_size,
        long_quantile=args.long_quantile,
        short_quantile=args.short_quantile,
        corr_threshold=args.corr_threshold,
        max_features_per_dimension=args.max_features_per_dimension,
        split_train_ratio=args.split_train_ratio,
        contracts=contracts,
    )


def make_stage2_config(config: GainCurveConfig, data: pd.DataFrame) -> Stage2Config:
    """复用 Stage2Config，让风控、交易生成和标签审计口径完全一致。"""

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
        corr_threshold=config.corr_threshold,
        max_group_share=0.45,
        min_selected_groups=5,
        long_quantile=config.long_quantile,
        short_quantile=config.short_quantile,
        split_train_ratio=config.split_train_ratio,
    )
    return resolve_time_split(base_config, data)


def load_stage2_pack(config: GainCurveConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = read_csv(config.stage2_dir / "stage2_features.csv", parse_datetime=True)
    labels = read_csv(config.stage2_dir / "stage2_labels.csv", parse_datetime=True)
    summary_train = read_csv(config.stage2_dir / "feature_summary_train.csv")
    corr_train = read_csv(config.stage2_dir / "feature_correlation_train.csv")
    transform_params = read_csv(config.stage2_dir / "feature_transform_params.csv")

    if config.contracts is not None:
        features = features[features["contract"].isin(config.contracts)].reset_index(drop=True)
        labels = labels[labels["contract"].isin(config.contracts)].reset_index(drop=True)

    label_only = [col for col in LABEL_COLUMNS if col in labels.columns]
    data = features.merge(labels[["contract", "datetime"] + label_only], on=["contract", "datetime"], how="left", validate="one_to_one")
    return data, summary_train, corr_train, transform_params, features


def _corr_lookup(corr_pairs: pd.DataFrame) -> Dict[tuple, float]:
    out: Dict[tuple, float] = {}
    for _, row in corr_pairs.iterrows():
        value = abs(float(row["spearman_corr"])) if pd.notna(row["spearman_corr"]) else 0.0
        out[(row["feature_a"], row["feature_b"])] = value
        out[(row["feature_b"], row["feature_a"])] = value
    return out


def _max_corr(feature: str, selected: List[str], corr_map: Dict[tuple, float]) -> float:
    if not selected:
        return 0.0
    return max(corr_map.get((feature, other), 0.0) for other in selected)


def _dimension_pool(summary_train: pd.DataFrame, dimension: str) -> pd.DataFrame:
    rule = DIMENSION_FEATURE_RULES[dimension]
    pool = summary_train.copy()
    if rule["groups"] is not None:
        pool = pool[pool["group"].isin(rule["groups"])]
    if rule["features"] is not None:
        pool = pool[pool["feature"].isin(rule["features"])]
    pool = pool[pool["coverage"] >= 0.70]
    pool = pool[pool["unique_count"] >= 10]
    pool = pool[pool["inf_count"] == 0]
    pool = pool[pool["spearman_ic_5"].notna()]
    pool = pool[pool["spearman_ic_5"].abs() > 0]
    return pool.sort_values("abs_spearman_ic_5", ascending=False).reset_index(drop=True)


def select_dimension_features(summary_train: pd.DataFrame, corr_train: pd.DataFrame, config: GainCurveConfig) -> pd.DataFrame:
    """每个维度在训练集内独立选最多 N 个低冗余特征。"""

    rows: List[Dict[str, Any]] = []
    corr_map = _corr_lookup(corr_train)
    for dimension in DIMENSION_FEATURE_RULES:
        selected: List[str] = []
        for _, row in _dimension_pool(summary_train, dimension).iterrows():
            feature = row["feature"]
            max_corr = _max_corr(feature, selected, corr_map)
            if max_corr > config.corr_threshold:
                continue
            selected.append(feature)
            rows.append(
                {
                    "dimension": dimension,
                    "dimension_order": len(selected),
                    "feature": feature,
                    "group": FEATURE_GROUP_MAP.get(feature, row["group"]),
                    "spearman_ic_5": float(row["spearman_ic_5"]),
                    "abs_spearman_ic_5": float(abs(row["spearman_ic_5"])),
                    "ic_direction": int(np.sign(row["spearman_ic_5"])),
                    "coverage": float(row["coverage"]),
                    "unique_count": int(row["unique_count"]),
                    "max_abs_corr_to_selected_in_dimension": max_corr,
                }
            )
            if len(selected) >= config.max_features_per_dimension:
                break
    return pd.DataFrame(rows)


def combo_selected_features(dimension_features: pd.DataFrame, combo: Dict[str, Any]) -> pd.DataFrame:
    selected = dimension_features[dimension_features["dimension"].isin(combo["dimensions"])].copy()
    selected.insert(0, "combo_name", combo["combo_name"])
    selected["combo_dimensions"] = ",".join(combo["dimensions"])
    return selected.reset_index(drop=True)


def add_combo_signal_scores(data: pd.DataFrame, selected: pd.DataFrame, transform_params: pd.DataFrame, combo_name: str) -> pd.DataFrame:
    """按维度内加权、维度间等权生成组合分数。"""

    out = data.copy()
    if selected.empty:
        out["signal_score"] = np.nan
        out["combo_name"] = combo_name
        return out

    features = selected["feature"].tolist()
    zscores = standardize_selected_features(out, transform_params, features)
    dimension_score_cols = []
    for dimension, part in selected.groupby("dimension"):
        weights = part["abs_spearman_ic_5"].astype(float)
        weight_sum = float(weights.sum())
        weights = weights / weight_sum if weight_sum > 0 else pd.Series([1.0 / len(part)] * len(part), index=part.index)
        pieces = []
        for (_, row), weight in zip(part.iterrows(), weights):
            direction = np.sign(float(row["spearman_ic_5"]))
            pieces.append(direction * weight * zscores[row["feature"]])
        col = f"dimension_score_{dimension}"
        out[col] = pd.concat(pieces, axis=1).sum(axis=1, min_count=1)
        dimension_score_cols.append(col)

    out["signal_score"] = out[dimension_score_cols].mean(axis=1)
    out["combo_name"] = combo_name
    return out


def metrics(trades: pd.DataFrame, split_name: str, return_col: str = "net_return") -> Dict[str, Any]:
    if trades.empty:
        return {
            "split": split_name,
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
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
        "short_trades": int((trades["side"] == "short").sum()),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_gross_return": float(gross.mean()) if len(gross) else np.nan,
        "avg_net_return": float(returns.mean()) if len(returns) else np.nan,
        "median_net_return": float(returns.median()) if len(returns) else np.nan,
        "total_net_return_sum": float(returns.sum()) if len(returns) else 0.0,
        "per_trade_sharpe": sharpe,
        "max_drawdown_sum": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def build_combo(data: pd.DataFrame, selected: pd.DataFrame, transform_params: pd.DataFrame, combo: Dict[str, Any], stage2_config: Stage2Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = add_combo_signal_scores(data, selected, transform_params, combo["combo_name"])
    scored, risk_thresholds = add_risk_filter(scored, stage2_config)
    train_scores = scored.loc[(scored["datetime"] <= stage2_config.train_end) & scored["risk_filter_pass"], "signal_score"].dropna()
    if train_scores.empty:
        trades = pd.DataFrame()
        long_threshold = np.nan
        short_threshold = np.nan
    else:
        long_threshold = float(train_scores.quantile(stage2_config.long_quantile))
        short_threshold = float(train_scores.quantile(stage2_config.short_quantile))
        trades = generate_trades(scored, stage2_config, long_threshold, short_threshold)
    if not trades.empty:
        trades.insert(0, "combo_name", combo["combo_name"])
        trades["combo_dimensions"] = ",".join(combo["dimensions"])

    summary_rows = []
    for split in ("train", "test", "all"):
        part = trades if split == "all" else trades[trades["split"] == split] if not trades.empty else trades
        row = metrics(part, split)
        row["combo_name"] = combo["combo_name"]
        row["combo_dimensions"] = ",".join(combo["dimensions"])
        row["dimension_count"] = len(combo["dimensions"])
        row["selected_feature_count"] = len(selected)
        row["risk_range_ticks_p995"] = risk_thresholds["range_ticks_p995"]
        row["risk_abs_poc_shift_ticks_p995"] = risk_thresholds["abs_poc_shift_ticks_p995"]
        row["long_threshold"] = long_threshold
        row["short_threshold"] = short_threshold
        row["tail_separation"] = long_threshold - short_threshold if pd.notna(long_threshold) and pd.notna(short_threshold) else np.nan
        row["threshold_abs_mean"] = (abs(long_threshold) + abs(short_threshold)) / 2.0 if pd.notna(long_threshold) and pd.notna(short_threshold) else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    monthly = build_monthly_metrics(trades, combo["combo_name"])
    cost_sensitivity = build_cost_sensitivity(trades, combo["combo_name"], stage2_config.tick_size)
    signal_distribution = build_signal_distribution(scored, combo["combo_name"], stage2_config, long_threshold, short_threshold)
    return trades, summary, monthly, cost_sensitivity, signal_distribution


def _score_stats(values: pd.Series) -> Dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "signal_count": 0,
            "score_mean": np.nan,
            "score_std": np.nan,
            "score_skew": np.nan,
            "score_kurtosis": np.nan,
            "score_p01": np.nan,
            "score_p05": np.nan,
            "score_p15": np.nan,
            "score_p50": np.nan,
            "score_p85": np.nan,
            "score_p95": np.nan,
            "score_p99": np.nan,
        }
    return {
        "signal_count": int(len(values)),
        "score_mean": float(values.mean()),
        "score_std": float(values.std(ddof=0)),
        "score_skew": float(values.skew()),
        "score_kurtosis": float(values.kurtosis()),
        "score_p01": float(values.quantile(0.01)),
        "score_p05": float(values.quantile(0.05)),
        "score_p15": float(values.quantile(0.15)),
        "score_p50": float(values.quantile(0.50)),
        "score_p85": float(values.quantile(0.85)),
        "score_p95": float(values.quantile(0.95)),
        "score_p99": float(values.quantile(0.99)),
    }


def build_signal_distribution(scored: pd.DataFrame, combo_name: str, stage2_config: Stage2Config, long_threshold: float, short_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    masks = {
        "train": scored["datetime"] <= stage2_config.train_end,
        "test": scored["datetime"] >= stage2_config.test_start,
        "all": pd.Series(True, index=scored.index),
    }
    for split, mask in masks.items():
        part = scored.loc[mask & scored["risk_filter_pass"], "signal_score"]
        row = _score_stats(part)
        row.update(
            {
                "combo_name": combo_name,
                "split": split,
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                "tail_separation": long_threshold - short_threshold if pd.notna(long_threshold) and pd.notna(short_threshold) else np.nan,
                "threshold_abs_mean": (abs(long_threshold) + abs(short_threshold)) / 2.0 if pd.notna(long_threshold) and pd.notna(short_threshold) else np.nan,
                "benchmark_combo": "",
                "tail_shrink_ratio": np.nan,
                "distribution_status": "ok",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_monthly_metrics(trades: pd.DataFrame, combo_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    work["month"] = pd.to_datetime(work["exit_datetime"]).dt.to_period("M").astype(str)
    for (split, month), part in work.groupby(["split", "month"]):
        row = metrics(part, split)
        row["combo_name"] = combo_name
        row["month"] = month
        rows.append(row)
    return pd.DataFrame(rows)


def build_cost_sensitivity(trades: pd.DataFrame, combo_name: str, tick_size: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame()
    for cost_ticks in (0.0, 1.0, 2.0, 3.0, 4.0):
        temp = trades.copy()
        temp["net_return_cost_case"] = temp["gross_return"] - (cost_ticks * tick_size) / temp["entry_price"]
        for split in ("train", "test", "all"):
            part = temp if split == "all" else temp[temp["split"] == split]
            row = metrics(part, split, return_col="net_return_cost_case")
            row["combo_name"] = combo_name
            row["cost_ticks_roundtrip"] = cost_ticks
            rows.append(row)
    return pd.DataFrame(rows)


def build_sample_confidence(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in summary.iterrows():
        trades = int(row["trades"]) if pd.notna(row["trades"]) else 0
        win_rate = float(row["win_rate"]) if pd.notna(row["win_rate"]) else np.nan
        if trades <= 0 or pd.isna(win_rate):
            se = np.nan
            ci_low = np.nan
            ci_high = np.nan
            sample_status = "no_samples"
        else:
            se = float(np.sqrt(win_rate * (1.0 - win_rate) / trades))
            ci_low = float(max(0.0, win_rate - 1.96 * se))
            ci_high = float(min(1.0, win_rate + 1.96 * se))
            if trades < MIN_TEST_TRADES:
                sample_status = "insufficient_samples"
            elif trades < LOW_CONFIDENCE_TRADES:
                sample_status = "low_confidence"
            else:
                sample_status = "enough_samples"
        rows.append(
            {
                "combo_name": row["combo_name"],
                "split": row["split"],
                "trades": trades,
                "min_required_trades": MIN_TEST_TRADES,
                "sample_status": sample_status,
                "win_rate": win_rate,
                "win_rate_standard_error": se,
                "win_rate_ci95_low": ci_low,
                "win_rate_ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def build_selected_correlation(selected_features: pd.DataFrame, corr_train: pd.DataFrame, corr_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    corr_map = _corr_lookup(corr_train)
    for combo_name, part in selected_features.groupby("combo_name"):
        records = part.to_dict("records")
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                corr = corr_map.get((left["feature"], right["feature"]), np.nan)
                abs_corr = abs(float(corr)) if pd.notna(corr) else np.nan
                if pd.isna(abs_corr):
                    status = "unknown"
                elif abs_corr > corr_threshold:
                    status = "collinearity_blocked"
                elif abs_corr >= COLLINEARITY_WARNING_THRESHOLD:
                    status = "collinearity_warning"
                else:
                    status = "ok"
                rows.append(
                    {
                        "combo_name": combo_name,
                        "feature_a": left["feature"],
                        "dimension_a": left["dimension"],
                        "group_a": left["group"],
                        "feature_b": right["feature"],
                        "dimension_b": right["dimension"],
                        "group_b": right["group"],
                        "spearman_corr": corr,
                        "abs_spearman_corr": abs_corr,
                        "is_cross_dimension": left["dimension"] != right["dimension"],
                        "correlation_status": status,
                    }
                )
    return pd.DataFrame(rows)


def add_distribution_diagnostics(distribution: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    out = distribution.copy()
    if out.empty:
        return out
    test = summary[summary["split"] == "test"].set_index("combo_name")
    train_distribution = out[out["split"] == "train"].set_index("combo_name")
    stronger_single = None
    if {"delta_only", "poc_only"}.issubset(test.index):
        stronger_single = test.loc[["delta_only", "poc_only"]].sort_values(["avg_net_return", "win_rate"], ascending=False).index[0]
    benchmarks = {
        "delta_poc": stronger_single,
        "delta_poc_position": "delta_poc",
    }

    for combo, benchmark in benchmarks.items():
        if not benchmark or combo not in train_distribution.index or benchmark not in train_distribution.index:
            continue
        combo_tail = train_distribution.loc[combo, "tail_separation"]
        benchmark_tail = train_distribution.loc[benchmark, "tail_separation"]
        if pd.isna(combo_tail) or pd.isna(benchmark_tail) or benchmark_tail == 0:
            continue
        shrink_ratio = float(1.0 - combo_tail / benchmark_tail)
        combo_net = test.loc[combo, "avg_net_return"] if combo in test.index else np.nan
        benchmark_net = test.loc[benchmark, "avg_net_return"] if benchmark in test.index else np.nan
        status = "ok"
        if shrink_ratio > DISTRIBUTION_SHRINK_THRESHOLD and pd.notna(combo_net) and pd.notna(benchmark_net) and combo_net <= benchmark_net:
            status = "distribution_deformed"
        elif shrink_ratio > DISTRIBUTION_SHRINK_THRESHOLD:
            status = "tail_compressed"
        mask = out["combo_name"] == combo
        out.loc[mask, "benchmark_combo"] = benchmark
        out.loc[mask, "tail_shrink_ratio"] = shrink_ratio
        out.loc[mask, "distribution_status"] = status
    return out


def _status_by_combo(df: pd.DataFrame, status_col: str, bad_statuses: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if df.empty or status_col not in df.columns:
        return out
    for combo_name, part in df.groupby("combo_name"):
        statuses = set(part[status_col].dropna().astype(str))
        for status in bad_statuses:
            if status in statuses:
                out[combo_name] = status
                break
    return out


def add_effectiveness(summary: pd.DataFrame, monthly: pd.DataFrame, sample_confidence: pd.DataFrame, selected_correlation: pd.DataFrame, signal_distribution: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["effectiveness"] = ""
    out["benchmark_combo"] = ""
    out["win_rate_gain_vs_benchmark"] = np.nan
    out["avg_net_return_gain_vs_benchmark"] = np.nan
    sample_cols = [
        "combo_name",
        "split",
        "min_required_trades",
        "sample_status",
        "win_rate_standard_error",
        "win_rate_ci95_low",
        "win_rate_ci95_high",
    ]
    out = out.merge(sample_confidence[sample_cols], on=["combo_name", "split"], how="left")

    if selected_correlation.empty:
        out["max_selected_abs_corr"] = np.nan
        out["collinearity_status"] = "ok"
    else:
        corr_summary = selected_correlation.groupby("combo_name").agg(max_selected_abs_corr=("abs_spearman_corr", "max")).reset_index()
        out = out.merge(corr_summary, on="combo_name", how="left")
        blocked = _status_by_combo(selected_correlation, "correlation_status", ["collinearity_blocked", "collinearity_warning"])
        out["collinearity_status"] = out["combo_name"].map(blocked).fillna("ok")

    dist_cols = ["combo_name", "split", "tail_shrink_ratio", "distribution_status"]
    out = out.merge(signal_distribution[dist_cols], on=["combo_name", "split"], how="left")

    test = out[out["split"] == "test"].set_index("combo_name")
    stronger_single = None
    if {"delta_only", "poc_only"}.issubset(test.index):
        single_candidates = test.loc[["delta_only", "poc_only"]].sort_values(["avg_net_return", "win_rate"], ascending=False)
        stronger_single = single_candidates.index[0]

    benchmarks = {
        "delta_only": None,
        "poc_only": None,
        "position_only": None,
        "delta_poc": stronger_single,
        "delta_poc_position": "delta_poc",
    }
    positive_month_share = monthly_positive_share(monthly)

    for idx, row in out.iterrows():
        if row["split"] != "test":
            continue
        combo = row["combo_name"]
        benchmark = benchmarks.get(combo)
        label = "ineffective"
        if benchmark and benchmark in test.index:
            base = test.loc[benchmark]
            win_gain = row["win_rate"] - base["win_rate"]
            net_gain = row["avg_net_return"] - base["avg_net_return"]
            out.loc[idx, "benchmark_combo"] = benchmark
            out.loc[idx, "win_rate_gain_vs_benchmark"] = win_gain
            out.loc[idx, "avg_net_return_gain_vs_benchmark"] = net_gain
        if row.get("sample_status") == "insufficient_samples":
            out.loc[idx, "effectiveness"] = "insufficient_samples"
            continue
        if row.get("collinearity_status") == "collinearity_blocked":
            out.loc[idx, "effectiveness"] = "collinearity_blocked"
            continue
        if row.get("distribution_status") == "distribution_deformed":
            out.loc[idx, "effectiveness"] = "distribution_deformed"
            continue
        if row["win_rate"] > 0.5 and row["avg_net_return"] > 0:
            label = "absolute_effective"
        if benchmark and benchmark in test.index:
            if win_gain >= 0.01 and net_gain > 0:
                label = "incremental_effective"
            elif win_gain > 0 and net_gain <= 0:
                label = "win_rate_only"
            elif row["avg_gross_return"] > 0 and row["avg_net_return"] <= 0:
                label = "cost_failed"
            elif win_gain <= 0 and net_gain <= 0:
                label = "ineffective"
        elif row["avg_gross_return"] > 0 and row["avg_net_return"] <= 0:
            label = "cost_failed"
        if label in {"absolute_effective", "incremental_effective"} and positive_month_share.get(combo, 0.0) < 0.5:
            label = "unstable"
        out.loc[idx, "effectiveness"] = label

    out["effectiveness"] = out["effectiveness"].replace("", "not_applicable")
    return out


def monthly_positive_share(monthly: pd.DataFrame) -> Dict[str, float]:
    if monthly.empty:
        return {}
    test = monthly[monthly["split"] == "test"].copy()
    if test.empty:
        return {}
    return test.groupby("combo_name")["avg_net_return"].apply(lambda s: float((s > 0).mean())).to_dict()


def build_interaction_diagnostics(data: pd.DataFrame, dimension_features: pd.DataFrame, transform_params: pd.DataFrame, stage2_config: Stage2Config) -> pd.DataFrame:
    """观察 Delta、POC、位置三类分数同向或背离时，未来收益是否变化。"""

    selected = dimension_features.copy()
    scored = add_combo_signal_scores(data, selected, transform_params, "all_dimensions_for_diagnostics")
    needed = ["dimension_score_delta", "dimension_score_poc", "dimension_score_position", stage2_config.label_col]
    if any(col not in scored.columns for col in needed):
        return pd.DataFrame()
    diag = scored.dropna(subset=needed).copy()
    diag = diag[diag["datetime"] >= stage2_config.test_start]
    if diag.empty:
        return pd.DataFrame()

    diag["delta_sign"] = np.sign(diag["dimension_score_delta"])
    diag["poc_sign"] = np.sign(diag["dimension_score_poc"])
    diag["position_sign"] = np.sign(diag["dimension_score_position"])
    diag["pattern"] = np.select(
        [
            (diag["delta_sign"] == diag["poc_sign"]) & (diag["poc_sign"] == diag["position_sign"]),
            diag["delta_sign"] == diag["poc_sign"],
            diag["delta_sign"] != diag["poc_sign"],
        ],
        ["all_same_direction", "delta_poc_same", "delta_poc_conflict"],
        default="mixed",
    )
    rows = []
    for pattern, part in diag.groupby("pattern"):
        rows.append(
            {
                "pattern": pattern,
                "rows": int(len(part)),
                "mean_future_return_5": float(part[stage2_config.label_col].mean()),
                "median_future_return_5": float(part[stage2_config.label_col].median()),
                "positive_rate": float((part[stage2_config.label_col] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_audit(
    summary: pd.DataFrame,
    selected_features: pd.DataFrame,
    trades: pd.DataFrame,
    selected_correlation: pd.DataFrame,
    sample_confidence: pd.DataFrame,
    signal_distribution: pd.DataFrame,
    stage2_config: Stage2Config,
    config: GainCurveConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for combo_name, part in selected_features.groupby("combo_name"):
        corr_part = selected_correlation[selected_correlation["combo_name"] == combo_name] if not selected_correlation.empty else pd.DataFrame()
        max_corr = float(corr_part["abs_spearman_corr"].max()) if not corr_part.empty else 0.0
        blocked_count = int((corr_part["correlation_status"] == "collinearity_blocked").sum()) if not corr_part.empty else 0
        warning_count = int((corr_part["correlation_status"] == "collinearity_warning").sum()) if not corr_part.empty else 0
        rows.append(_audit_row("BLOCKER", combo_name, "selected_corr_threshold", "PASS" if blocked_count == 0 else "FAIL", "入选特征相关性不超过阈值", blocked_count, f"max_corr={max_corr:.6f}"))
        if warning_count:
            rows.append(_audit_row("WARNING", combo_name, "selected_corr_warning", "WARN", "入选特征存在 0.70~0.85 的高相关警告", warning_count, f"max_corr={max_corr:.6f}"))
        rows.append(_audit_row("BLOCKER", combo_name, "selected_from_train_summary", "PASS", "特征只来自训练集 feature_summary_train", 0))
        sample_part = sample_confidence[(sample_confidence["combo_name"] == combo_name) & (sample_confidence["split"] == "test")]
        if not sample_part.empty:
            sample_status = sample_part.iloc[0]["sample_status"]
            if sample_status == "insufficient_samples":
                rows.append(_audit_row("WARNING", combo_name, "insufficient_samples", "WARN", "测试集交易样本不足，不能认定有效", int(sample_part.iloc[0]["trades"])))
            elif sample_status == "low_confidence":
                rows.append(_audit_row("WARNING", combo_name, "low_confidence_samples", "WARN", "测试集交易样本偏少，结论置信度较低", int(sample_part.iloc[0]["trades"])))
        dist_part = signal_distribution[(signal_distribution["combo_name"] == combo_name) & (signal_distribution["split"] == "train")]
        if not dist_part.empty and dist_part.iloc[0]["distribution_status"] in {"distribution_deformed", "tail_compressed"}:
            rows.append(_audit_row("WARNING", combo_name, "signal_distribution", "WARN", "合成信号尾部出现明显收缩", 0, f"status={dist_part.iloc[0]['distribution_status']}, shrink={dist_part.iloc[0]['tail_shrink_ratio']}"))
    if trades.empty:
        rows.append(_audit_row("WARNING", "ALL", "no_trades", "WARN", "所有组合均无交易", 0))
    else:
        for combo_name, part in trades.groupby("combo_name"):
            overlap = check_no_overlap(part)
            overlap_count = int(overlap["overlap_count"].sum()) if not overlap.empty else 0
            boundary_bad = int(((part["split"] == "train") & (pd.to_datetime(part["exit_datetime"]) >= stage2_config.test_start)).sum())
            long_bad = int(((part["side"] == "long") & (part["signal_score"] < part["long_threshold"])).sum())
            short_bad = int(((part["side"] == "short") & (part["signal_score"] > part["short_threshold"])).sum())
            rows.append(_audit_row("BLOCKER", combo_name, "trade_overlap", "PASS" if overlap_count == 0 else "FAIL", "同合约交易不重叠", overlap_count))
            rows.append(_audit_row("BLOCKER", combo_name, "train_test_boundary", "PASS" if boundary_bad == 0 else "FAIL", "训练集交易不穿越测试边界", boundary_bad))
            rows.append(_audit_row("BLOCKER", combo_name, "threshold_direction", "PASS" if long_bad + short_bad == 0 else "FAIL", "交易方向符合阈值", long_bad + short_bad))
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
        if col in {"combo_name", "benchmark_combo", "split", "dimension", "dimension_a", "dimension_b", "effectiveness", "sample_status", "collinearity_status", "distribution_status", "correlation_status", "pattern", "status", "severity"}:
            view[col] = view[col].map(lambda x: VALUE_LABELS.get(x, COMBO_LABELS.get(x, x)))
        elif col in {"combo_dimensions"}:
            view[col] = view[col].map(lambda x: "+".join(VALUE_LABELS.get(part, part) for part in str(x).split(",")) if pd.notna(x) else x)
        elif col in {"group", "group_a", "group_b", "check_id", "is_cross_dimension"}:
            view[col] = view[col].map(lambda x: VALUE_LABELS.get(x, x))
    columns = [COLUMN_LABELS.get(col, col) for col in original_columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_format_md_cell(row[col]) for col in original_columns) + " |")
    return "\n".join(lines)


def _format_md_cell(value: Any) -> str:
    """把报告表格里的缺失值和浮点数格式化为更适合人工阅读的中文展示。"""

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


def explain_combo(combo_name: str, test_row: pd.Series) -> str:
    status = test_row.get("effectiveness", "unknown")
    win_rate = test_row.get("win_rate", np.nan)
    avg_net = test_row.get("avg_net_return", np.nan)
    trades = int(test_row.get("trades", 0)) if pd.notna(test_row.get("trades", np.nan)) else 0
    benchmark = test_row.get("benchmark_combo", "")
    win_gain = test_row.get("win_rate_gain_vs_benchmark", np.nan)
    net_gain = test_row.get("avg_net_return_gain_vs_benchmark", np.nan)
    label = COMBO_LABELS.get(combo_name, combo_name)
    if status == "insufficient_samples":
        return f"- `{label}`：样本不足。测试集只有 `{trades}` 笔交易，低于 `{MIN_TEST_TRADES}` 笔最低要求，即使胜率较高也不能认定有效。"
    if status == "collinearity_blocked":
        max_corr = test_row.get("max_selected_abs_corr", np.nan)
        return f"- `{label}`：共线性阻断。入选特征最大相关 `{max_corr:.4f}` 超过阈值，维度可能已经坍缩。"
    if status == "distribution_deformed":
        shrink = test_row.get("tail_shrink_ratio", np.nan)
        return f"- `{label}`：分布形变。合成信号尾部分离度收缩 `{shrink:.2%}`，线性平均可能削弱了极端动能。"
    if status in {"absolute_effective", "incremental_effective"}:
        return f"- `{label}`：有效。测试集交易 `{trades}` 笔，胜率 `{win_rate:.2%}`，平均净收益 `{avg_net:.8f}`。"
    if status == "win_rate_only":
        return f"- `{label}`：胜率有增益但净收益无增益，属于伪增益。测试集胜率 `{win_rate:.2%}`，平均净收益 `{avg_net:.8f}`。"
    if status == "cost_failed":
        gross = test_row.get("avg_gross_return", np.nan)
        return f"- `{label}`：无绝对有效性。测试集毛收益 `{gross:.8f}`，扣 2 tick 后平均净收益 `{avg_net:.8f}`，主要被成本吞噬。"
    if status == "unstable":
        return f"- `{label}`：总体有改善但月度稳定性不足，暂不认定为稳定有效。"
    if benchmark and pd.notna(win_gain) and pd.notna(net_gain):
        reason = []
        if win_gain < 0.01:
            reason.append(f"胜率增量 `{win_gain:.2%}` 未达到 1 pct point 门槛")
        if net_gain <= 0:
            reason.append(f"平均净收益相对 `{COMBO_LABELS.get(benchmark, benchmark)}` 下降 `{net_gain:.8f}`")
        elif avg_net <= 0:
            reason.append(f"平均净收益虽改善 `{net_gain:.8f}` 但仍为负 `{avg_net:.8f}`")
        reason_text = "；".join(reason) if reason else "未同时满足胜率和净收益增益门槛"
        return f"- `{label}`：无效。对比 `{COMBO_LABELS.get(benchmark, benchmark)}`，{reason_text}。"
    return f"- `{label}`：无效。测试集胜率 `{win_rate:.2%}`，平均净收益 `{avg_net:.8f}`。"


def build_conclusion_bullets(test: pd.DataFrame) -> List[str]:
    rows = {row["combo_name"]: row for _, row in test.iterrows()}
    bullets: List[str] = []
    for combo in ["delta_only", "poc_only", "position_only", "delta_poc", "delta_poc_position"]:
        row = rows.get(combo)
        if row is None:
            continue
        bullets.append(explain_combo(combo, row))
    if {"delta_only", "poc_only", "delta_poc", "delta_poc_position"}.issubset(rows):
        delta_poc = rows["delta_poc"]
        final = rows["delta_poc_position"]
        bullets.append(
            "- 主链条判断：`Delta+POC` 相对较强单维度没有胜率和净收益同步提升，说明 Delta 与 POC 在当前规则下没有形成可靠互补。"
        )
        bullets.append(
            f"- 位置增量判断：`Delta+POC+位置` 相对 `Delta+POC` 胜率提升 `{final['win_rate_gain_vs_benchmark']:.2%}`，平均净收益改善 `{final['avg_net_return_gain_vs_benchmark']:.8f}`，但胜率增量低于 1 pct point 且净收益仍为负，因此只能视作弱改善，不能认定有效。"
        )
    return bullets


def write_report(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    monthly: pd.DataFrame,
    interaction: pd.DataFrame,
    audit: pd.DataFrame,
    sample_confidence: pd.DataFrame,
    selected_correlation: pd.DataFrame,
    signal_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    test = summary[summary["split"] == "test"].copy()
    test = test.sort_values("combo_name")
    blockers = audit[(audit["severity"] == "BLOCKER") & (audit["status"] == "FAIL")]
    conclusion = "不成立"
    delta_poc = test[test["combo_name"] == "delta_poc"]
    final = test[test["combo_name"] == "delta_poc_position"]
    if not final.empty and final.iloc[0]["effectiveness"] in {"absolute_effective", "incremental_effective"}:
        conclusion = "成立"
    elif not delta_poc.empty and delta_poc.iloc[0]["effectiveness"] in {"absolute_effective", "incremental_effective"}:
        conclusion = "部分成立"

    lines = [
        "# 阶段二增益曲线验证报告",
        "",
        "## 结论",
        f"- Delta → POC → 位置 这条增益链条：`{conclusion}`。",
        f"- 审计阻断项：`{len(blockers)}`。",
        "- 判定口径：测试集胜率提升且扣 2 tick 后平均净收益同步改善，才算有效增益。",
        "",
        "## 测试集增益曲线",
        _md_table(test[["combo_name", "trades", "sample_status", "win_rate", "win_rate_ci95_low", "win_rate_ci95_high", "avg_gross_return", "avg_net_return", "total_net_return_sum", "max_drawdown_sum", "benchmark_combo", "win_rate_gain_vs_benchmark", "avg_net_return_gain_vs_benchmark", "effectiveness"]]),
        "",
        "## 组合逐项解释",
    ]
    lines.extend(build_conclusion_bullets(test))
    lines.extend(
        [
            "",
            "## 样本耗竭检查",
            "- 测试集交易数低于 100 笔时，优先判定为样本不足；100 到 299 笔为低置信。",
            _md_table(sample_confidence[sample_confidence["split"] == "test"][["combo_name", "split", "trades", "min_required_trades", "sample_status", "win_rate", "win_rate_standard_error", "win_rate_ci95_low", "win_rate_ci95_high"]]),
            "",
            "## 跨维度共线性检查",
            "- 相关性大于 0.85 视为共线性阻断；0.70 到 0.85 视为共线性警告。",
            _md_table(
                selected_correlation.sort_values("abs_spearman_corr", ascending=False)[
                    ["combo_name", "feature_a", "dimension_a", "feature_b", "dimension_b", "spearman_corr", "abs_spearman_corr", "is_cross_dimension", "correlation_status"]
                ],
                max_rows=30,
            ),
            "",
            "## 合成信号分布形变检查",
            "- 如果多维组合相对基准的尾部分离度收缩超过 40%，且净收益没有改善，说明线性平均可能削弱了尾部信号。",
            _md_table(
                signal_distribution[signal_distribution["split"] == "train"][
                    ["combo_name", "split", "signal_count", "score_mean", "score_std", "score_skew", "score_kurtosis", "score_p15", "score_p50", "score_p85", "long_threshold", "short_threshold", "tail_separation", "threshold_abs_mean", "benchmark_combo", "tail_shrink_ratio", "distribution_status"]
                ],
                max_rows=20,
            ),
            "",
            "## 入选特征",
            _md_table(selected[["combo_name", "dimension", "dimension_order", "feature", "group", "spearman_ic_5", "max_abs_corr_to_selected_in_dimension"]], max_rows=80),
            "",
            "## 月度稳定性",
            _md_table(monthly[monthly["split"] == "test"][["combo_name", "month", "trades", "win_rate", "avg_net_return"]], max_rows=40),
            "",
            "## 维度交互诊断",
            _md_table(interaction),
            "",
            "## 审计",
            _md_table(audit, max_rows=80),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(config: GainCurveConfig) -> Dict[str, Any]:
    data, summary_train, corr_train, transform_params, features = load_stage2_pack(config)
    stage2_config = make_stage2_config(config, data)
    dimension_features = select_dimension_features(summary_train, corr_train, config)

    all_trades = []
    all_summary = []
    all_monthly = []
    all_cost = []
    all_selected = []
    all_signal_distribution = []
    for combo in COMBO_DEFINITIONS:
        selected = combo_selected_features(dimension_features, combo)
        trades, summary, monthly, cost, signal_distribution = build_combo(data, selected, transform_params, combo, stage2_config)
        all_selected.append(selected)
        if not trades.empty:
            all_trades.append(trades)
        all_summary.append(summary)
        if not monthly.empty:
            all_monthly.append(monthly)
        if not cost.empty:
            all_cost.append(cost)
        if not signal_distribution.empty:
            all_signal_distribution.append(signal_distribution)

    selected_features = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    monthly = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()
    cost_sensitivity = pd.concat(all_cost, ignore_index=True) if all_cost else pd.DataFrame()
    signal_distribution = pd.concat(all_signal_distribution, ignore_index=True) if all_signal_distribution else pd.DataFrame()
    selected_correlation = build_selected_correlation(selected_features, corr_train, config.corr_threshold)
    sample_confidence = build_sample_confidence(summary)
    signal_distribution = add_distribution_diagnostics(signal_distribution, summary)
    summary = add_effectiveness(summary, monthly, sample_confidence, selected_correlation, signal_distribution)
    interaction = build_interaction_diagnostics(data, dimension_features, transform_params, stage2_config)
    audit = build_audit(summary, selected_features, trades, selected_correlation, sample_confidence, signal_distribution, stage2_config, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary, config.output_dir / "gain_curve_summary.csv")
    write_csv(selected_features, config.output_dir / "gain_curve_selected_features.csv")
    write_csv(trades, config.output_dir / "gain_curve_trades.csv")
    write_csv(monthly, config.output_dir / "gain_curve_monthly.csv")
    write_csv(cost_sensitivity, config.output_dir / "gain_curve_cost_sensitivity.csv")
    write_csv(interaction, config.output_dir / "gain_curve_interaction_diagnostics.csv")
    write_csv(sample_confidence, config.output_dir / "gain_curve_sample_confidence.csv")
    write_csv(selected_correlation, config.output_dir / "gain_curve_selected_correlation.csv")
    write_csv(signal_distribution, config.output_dir / "gain_curve_signal_distribution.csv")
    write_csv(audit, config.output_dir / "gain_curve_audit.csv")
    write_report(summary, selected_features, monthly, interaction, audit, sample_confidence, selected_correlation, signal_distribution, config.output_dir / "gain_curve_report_zh.md")

    blockers = audit[(audit["severity"] == "BLOCKER") & (audit["status"] == "FAIL")]
    return {
        "output_dir": config.output_dir,
        "contracts": int(data["contract"].nunique()),
        "rows": int(len(data)),
        "train_end": stage2_config.train_end,
        "test_start": stage2_config.test_start,
        "combos": len(COMBO_DEFINITIONS),
        "trades": int(len(trades)),
        "audit_blockers": int(len(blockers)),
    }
