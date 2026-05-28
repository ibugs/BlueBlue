#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 audit for the order-flow MVP research outputs.

This script does not change the research pipeline. It reads the generated CSVs,
checks data integrity, feature quality, label alignment, leakage risks, strategy
logic, extreme rows, and writes a repeatable audit pack.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_ORDERFLOW_DIR = Path("/Users/wangrendong/Projects/BlueBlue/orderflow_data/SHFE.cu")
DEFAULT_STAGE1_DIR = Path("/Users/wangrendong/Projects/BlueBlue/stage1_outputs/SHFE.cu")
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

REQUIRED_CORE_FILES = [
    "features.csv",
    "labels.csv",
    "feature_summary.csv",
    "feature_quintiles.csv",
    "monthly_ic.csv",
    "selected_features.csv",
    "trades_mvp.csv",
    "strategy_summary.csv",
    "equity_curve.csv",
    "trade_overlap_check.csv",
]

OPTIONAL_REPORT_FILES = [
    "stage1_mvp_report.md",
    "stage1_mvp_report_zh.md",
]

REQUIRED_STAGE1_FILES = REQUIRED_CORE_FILES + OPTIONAL_REPORT_FILES

FEATURE_GROUPS = {
    "bar_return": "price_structure",
    "range_ticks": "price_structure",
    "body_ratio": "price_structure",
    "close_position": "price_structure",
    "upper_shadow_ratio": "price_structure",
    "lower_shadow_ratio": "price_structure",
    "delta_strength": "orderflow",
    "delta_zscore_20": "orderflow",
    "cvd_change_10": "orderflow",
    "price_delta_agreement": "orderflow",
    "poc_distance_ticks": "poc",
    "poc_shift_ticks": "poc",
    "volume_zscore_20": "volume",
    "open_interest_change": "open_interest",
    "open_interest_zscore_20": "open_interest",
    "trend_return_12": "trend_volatility",
    "volatility_20": "trend_volatility",
}

FORBIDDEN_FEATURE_TOKENS = ["future_", "mfe", "mae", "entry_", "exit_", "trade_return"]
FLOAT_TOL = 1e-10


@dataclass
class AuditConfig:
    orderflow_dir: Path
    stage1_dir: Path
    output_dir: Path
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    tick_size: float
    fail_on_blocker: bool


def parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(description="Audit stage 1 order-flow MVP outputs")
    parser.add_argument("--orderflow_dir", type=str, default=str(DEFAULT_ORDERFLOW_DIR))
    parser.add_argument("--stage1_dir", type=str, default=str(DEFAULT_STAGE1_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--test_start", type=str, default=DEFAULT_TEST_START)
    parser.add_argument("--tick_size", type=float, default=10.0)
    parser.add_argument("--fail_on_blocker", action="store_true")
    args = parser.parse_args()
    stage1_dir = Path(args.stage1_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else stage1_dir / "audit"
    return AuditConfig(
        orderflow_dir=Path(args.orderflow_dir).expanduser(),
        stage1_dir=stage1_dir,
        output_dir=output_dir,
        train_end=pd.Timestamp(args.train_end),
        test_start=pd.Timestamp(args.test_start),
        tick_size=args.tick_size,
        fail_on_blocker=args.fail_on_blocker,
    )


def finding(severity: str, module: str, check_id: str, status: str, message: str, row_count: int = 0, detail: str = "") -> Dict[str, Any]:
    return {
        "severity": severity,
        "module": module,
        "check_id": check_id,
        "status": status,
        "message": message,
        "row_count": int(row_count),
        "detail": detail,
    }


def contract_from_orderflow_filename(path: Path) -> Optional[str]:
    match = re.search(r"_(SHFE\.cu\d+)_\d+\.csv$", path.name)
    return match.group(1) if match else None


def read_csv(path: Path, parse_datetime: bool = False) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    if parse_datetime and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def load_stage1(config: AuditConfig) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for file_name in REQUIRED_STAGE1_FILES:
        path = config.stage1_dir / file_name
        data[file_name] = path
        if path.exists() and path.suffix == ".csv":
            data[file_name.replace(".csv", "")] = read_csv(path, parse_datetime=True)
    return data


def load_orderflow(orderflow_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(orderflow_dir.glob("period_of_5_*_SHFE.cu*.csv")):
        contract = contract_from_orderflow_filename(path)
        if not contract:
            continue
        df = read_csv(path, parse_datetime=False)
        df["contract"] = contract
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M", errors="coerce")
        frames.append(df[[c for c in BASE_COLUMNS if c in df.columns]].copy())
    if not frames:
        return pd.DataFrame(columns=BASE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    for col in ["open", "high", "low", "close", "volume", "poc", "delta", "open_interest"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["contract", "datetime"]).reset_index(drop=True)


def compare_float_series(left: pd.Series, right: pd.Series, tolerance: float = FLOAT_TOL) -> Tuple[int, float]:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_missing = left_num.isna() & right_num.isna()
    both_present = left_num.notna() & right_num.notna()
    diff = (left_num - right_num).abs()
    ok = both_missing | (both_present & (diff <= tolerance))
    mismatch_count = int((~ok).sum())
    max_diff = float(diff[both_present].max()) if bool(both_present.any()) else 0.0
    return mismatch_count, max_diff


def forward_max(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def forward_min(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def add_check_row(rows: List[Dict[str, Any]], check_id: str, severity: str, status: str, message: str, value: Any = "", expected: Any = "", row_count: int = 0) -> None:
    rows.append(
        {
            "check_id": check_id,
            "severity": severity,
            "status": status,
            "message": message,
            "value": value,
            "expected": expected,
            "row_count": int(row_count),
        }
    )


def audit_data_integrity(config: AuditConfig, data: Dict[str, Any], orderflow: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []

    missing_files = [name for name in REQUIRED_CORE_FILES if not (config.stage1_dir / name).exists()]
    add_check_row(rows, "required_files", "BLOCKER", "FAIL" if missing_files else "PASS", "核心输出文件存在性", ",".join(missing_files), "no missing files", len(missing_files))
    if missing_files:
        findings.append(finding("BLOCKER", "data_integrity", "required_files", "FAIL", f"缺失核心文件: {missing_files}", len(missing_files)))

    features = data.get("features", pd.DataFrame())
    labels = data.get("labels", pd.DataFrame())
    if features.empty or labels.empty:
        findings.append(finding("BLOCKER", "data_integrity", "features_labels", "FAIL", "features.csv 或 labels.csv 为空或不存在"))
        return pd.DataFrame(rows), findings

    duplicate_count = int(features.duplicated(["contract", "datetime"]).sum())
    add_check_row(rows, "duplicate_contract_datetime", "BLOCKER", "FAIL" if duplicate_count else "PASS", "features中 contract+datetime 不应重复", duplicate_count, 0, duplicate_count)
    if duplicate_count:
        findings.append(finding("BLOCKER", "data_integrity", "duplicate_contract_datetime", "FAIL", "发现重复 contract+datetime", duplicate_count))

    label_duplicate_count = int(labels.duplicated(["contract", "datetime"]).sum())
    add_check_row(rows, "label_duplicate_contract_datetime", "BLOCKER", "FAIL" if label_duplicate_count else "PASS", "labels中 contract+datetime 不应重复", label_duplicate_count, 0, label_duplicate_count)
    if label_duplicate_count:
        findings.append(finding("BLOCKER", "data_integrity", "label_duplicate_contract_datetime", "FAIL", "labels发现重复 contract+datetime", label_duplicate_count))

    add_check_row(rows, "feature_label_row_count", "BLOCKER", "FAIL" if len(features) != len(labels) else "PASS", "features和labels行数一致", len(features), len(labels), abs(len(features) - len(labels)))
    if len(features) != len(labels):
        findings.append(finding("BLOCKER", "data_integrity", "feature_label_row_count", "FAIL", "features和labels行数不一致", abs(len(features) - len(labels))))

    base_mismatches = 0
    for col in BASE_COLUMNS:
        if col not in features.columns or col not in labels.columns:
            base_mismatches += 1
            continue
        if col == "datetime":
            base_mismatches += int((features[col] != labels[col]).sum())
        elif col == "contract":
            base_mismatches += int((features[col].astype(str) != labels[col].astype(str)).sum())
        else:
            mismatch, _ = compare_float_series(features[col], labels[col])
            base_mismatches += mismatch
    add_check_row(rows, "feature_label_base_columns", "BLOCKER", "FAIL" if base_mismatches else "PASS", "features和labels基础字段一致", base_mismatches, 0, base_mismatches)
    if base_mismatches:
        findings.append(finding("BLOCKER", "data_integrity", "feature_label_base_columns", "FAIL", "features和labels基础字段不一致", base_mismatches))

    ohlc_bad = features[
        (features["high"] < features[["open", "close", "low"]].max(axis=1))
        | (features["low"] > features[["open", "close", "high"]].min(axis=1))
    ]
    add_check_row(rows, "ohlc_structure", "BLOCKER", "FAIL" if len(ohlc_bad) else "PASS", "OHLC结构合法", len(ohlc_bad), 0, len(ohlc_bad))
    if len(ohlc_bad):
        findings.append(finding("BLOCKER", "data_integrity", "ohlc_structure", "FAIL", "OHLC结构异常", len(ohlc_bad)))

    bad_volume = features[features["volume"] < 0]
    bad_delta = features[features["delta"].abs() > features["volume"]]
    bad_poc = features[(features["poc"] < features["low"]) | (features["poc"] > features["high"])]
    for check_id, bad, msg in [
        ("volume_nonnegative", bad_volume, "volume >= 0"),
        ("delta_within_volume", bad_delta, "abs(delta) <= volume"),
        ("poc_in_bar_range", bad_poc, "poc位于[low, high]"),
    ]:
        add_check_row(rows, check_id, "BLOCKER", "FAIL" if len(bad) else "PASS", msg, len(bad), 0, len(bad))
        if len(bad):
            findings.append(finding("BLOCKER", "data_integrity", check_id, "FAIL", msg, len(bad)))

    if not orderflow.empty:
        orderflow_subset = orderflow.merge(features[["contract", "datetime"]], on=["contract", "datetime"], how="inner")
        add_check_row(rows, "orderflow_stage1_row_match", "BLOCKER", "FAIL" if len(orderflow_subset) != len(features) else "PASS", "stage1行可回溯到orderflow原始bar", len(orderflow_subset), len(features), abs(len(orderflow_subset) - len(features)))
        if len(orderflow_subset) != len(features):
            findings.append(finding("BLOCKER", "data_integrity", "orderflow_stage1_row_match", "FAIL", "stage1行无法全部回溯到orderflow", abs(len(orderflow_subset) - len(features))))

    add_check_row(rows, "data_scope", "OBSERVATION", "INFO", "数据范围", f"{features['datetime'].min()} -> {features['datetime'].max()}", f"contracts={features['contract'].nunique()}, rows={len(features)}", len(features))
    return pd.DataFrame(rows), findings


def audit_feature_quality(data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    features = data["features"]
    selected = data.get("selected_features", pd.DataFrame())
    rows: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []

    missing_features = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing_features:
        findings.append(finding("BLOCKER", "feature_quality", "missing_features", "FAIL", f"缺失特征: {missing_features}", len(missing_features)))

    for feature in FEATURE_COLUMNS:
        if feature not in features.columns:
            rows.append({"feature": feature, "group": FEATURE_GROUPS.get(feature, ""), "status": "MISSING"})
            continue
        s = pd.to_numeric(features[feature], errors="coerce")
        finite = s.replace([np.inf, -np.inf], np.nan)
        non_null = int(finite.notna().sum())
        unique_count = int(finite.nunique(dropna=True))
        missing_count = int(finite.isna().sum())
        inf_count = int(np.isinf(s).sum())
        std = float(finite.std(ddof=0)) if non_null else np.nan
        status = "PASS"
        if inf_count:
            status = "FAIL"
            findings.append(finding("BLOCKER", "feature_quality", f"{feature}_inf", "FAIL", f"{feature} 含 inf", inf_count))
        elif unique_count <= 1 or (pd.notna(std) and std == 0):
            status = "WARN"
            findings.append(finding("WARNING", "feature_quality", f"{feature}_low_variance", "WARN", f"{feature} 唯一值过少或零方差", unique_count))

        quantiles = finite.quantile([0.001, 0.01, 0.5, 0.99, 0.999]).to_dict() if non_null else {}
        rows.append(
            {
                "feature": feature,
                "group": FEATURE_GROUPS.get(feature, ""),
                "status": status,
                "non_null_count": non_null,
                "missing_count": missing_count,
                "missing_rate": float(missing_count / len(features)) if len(features) else np.nan,
                "coverage": float(non_null / len(features)) if len(features) else np.nan,
                "unique_count": unique_count,
                "inf_count": inf_count,
                "std": std,
                "min": float(finite.min()) if non_null else np.nan,
                "p001": float(quantiles.get(0.001, np.nan)),
                "p01": float(quantiles.get(0.01, np.nan)),
                "median": float(quantiles.get(0.5, np.nan)),
                "p99": float(quantiles.get(0.99, np.nan)),
                "p999": float(quantiles.get(0.999, np.nan)),
                "max": float(finite.max()) if non_null else np.nan,
            }
        )

    feature_quality = pd.DataFrame(rows)

    corr_rows = []
    valid_feature_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    corr_df = features[valid_feature_cols].apply(pd.to_numeric, errors="coerce")
    pearson = corr_df.corr(method="pearson")
    spearman = corr_df.corr(method="spearman")
    high_corr_count = 0
    for i, left in enumerate(valid_feature_cols):
        for right in valid_feature_cols[i + 1 :]:
            p = pearson.loc[left, right]
            sp = spearman.loc[left, right]
            high_corr = bool((pd.notna(p) and abs(p) >= 0.95) or (pd.notna(sp) and abs(sp) >= 0.95))
            high_corr_count += int(high_corr)
            corr_rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "group_a": FEATURE_GROUPS.get(left, ""),
                    "group_b": FEATURE_GROUPS.get(right, ""),
                    "pearson_corr": p,
                    "spearman_corr": sp,
                    "high_corr_abs_ge_095": high_corr,
                }
            )
    feature_correlation = pd.DataFrame(corr_rows)
    if high_corr_count:
        findings.append(finding("WARNING", "feature_quality", "high_feature_correlation", "WARN", "存在高度相关特征对 abs(corr)>=0.95", high_corr_count))

    selected_diversity_rows = []
    if not selected.empty and "feature" in selected.columns:
        selected_groups = selected["feature"].map(FEATURE_GROUPS).fillna("unknown")
        group_counts = selected_groups.value_counts()
        for group, count in group_counts.items():
            selected_diversity_rows.append({"group": group, "selected_feature_count": int(count), "share": float(count / len(selected))})
        if len(group_counts) and group_counts.iloc[0] / len(selected) >= 0.6:
            findings.append(finding("WARNING", "feature_quality", "selected_feature_concentration", "WARN", "选中特征过度集中在单一特征组", int(group_counts.iloc[0]), str(group_counts.to_dict())))
    feature_group_diversity = pd.DataFrame(selected_diversity_rows)

    expected_groups = set(FEATURE_GROUPS.values())
    actual_groups = set(feature_quality["group"].dropna())
    missing_groups = expected_groups - actual_groups
    if missing_groups:
        findings.append(finding("WARNING", "feature_quality", "missing_feature_groups", "WARN", f"缺失特征组: {sorted(missing_groups)}", len(missing_groups)))

    return feature_quality, feature_correlation, feature_group_diversity, findings


def recompute_labels_for_contract(group: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    g = group.sort_values("datetime").copy()
    for horizon in (1, 3, 5, 10):
        g[f"recalc_future_return_{horizon}"] = g["close"].shift(-horizon) / g["close"].replace(0, np.nan) - 1
    future_high_10 = forward_max(g["high"], 10)
    future_low_10 = forward_min(g["low"], 10)
    g["recalc_mfe_10_long"] = future_high_10 / g["close"].replace(0, np.nan) - 1
    g["recalc_mae_10_long"] = future_low_10 / g["close"].replace(0, np.nan) - 1
    g["recalc_entry_open_next"] = g["open"].shift(-1)
    g["recalc_exit_close_after_5"] = g["close"].shift(-5)
    g["recalc_trade_return_5_gross"] = g["recalc_exit_close_after_5"] / g["recalc_entry_open_next"].replace(0, np.nan) - 1
    g["recalc_trade_return_5_net"] = g["recalc_trade_return_5_gross"] - (2.0 * tick_size) / g["recalc_entry_open_next"].replace(0, np.nan)
    return g


def audit_label_alignment(config: AuditConfig, data: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    labels = data["labels"].copy()
    features = data["features"].copy()
    findings: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    recomputed = pd.concat([recompute_labels_for_contract(g, config.tick_size) for _, g in labels.groupby("contract", sort=True)], ignore_index=True)
    check_map = {
        "future_return_1": "recalc_future_return_1",
        "future_return_3": "recalc_future_return_3",
        "future_return_5": "recalc_future_return_5",
        "future_return_10": "recalc_future_return_10",
        "mfe_10_long": "recalc_mfe_10_long",
        "mae_10_long": "recalc_mae_10_long",
        "entry_open_next": "recalc_entry_open_next",
        "exit_close_after_5": "recalc_exit_close_after_5",
        "trade_return_5_gross": "recalc_trade_return_5_gross",
        "trade_return_5_net": "recalc_trade_return_5_net",
    }
    for label, recalculated in check_map.items():
        mismatch_count, max_abs_diff = compare_float_series(recomputed[label], recomputed[recalculated])
        status = "FAIL" if mismatch_count else "PASS"
        rows.append({"label": label, "status": status, "mismatch_count": mismatch_count, "max_abs_diff": max_abs_diff})
        if mismatch_count:
            findings.append(finding("BLOCKER", "label_alignment", label, "FAIL", f"{label} 与重新计算结果不一致", mismatch_count, f"max_abs_diff={max_abs_diff}"))

    tail_rules = {
        "future_return_1": 1,
        "future_return_3": 3,
        "future_return_5": 5,
        "future_return_10": 10,
        "mfe_10_long": 10,
        "mae_10_long": 10,
        "entry_open_next": 1,
        "exit_close_after_5": 5,
        "trade_return_5_gross": 5,
        "trade_return_5_net": 5,
    }
    for label, tail_n in tail_rules.items():
        bad_contracts = []
        for contract, group in labels.groupby("contract"):
            if group.tail(tail_n)[label].notna().any():
                bad_contracts.append(contract)
        rows.append({"label": f"{label}_tail_null", "status": "FAIL" if bad_contracts else "PASS", "mismatch_count": len(bad_contracts), "max_abs_diff": np.nan})
        if bad_contracts:
            findings.append(finding("BLOCKER", "label_alignment", f"{label}_tail_null", "FAIL", "合约尾部未来标签非空，疑似跨合约泄漏", len(bad_contracts), ",".join(bad_contracts[:10])))

    forbidden_columns = [col for col in features.columns if col not in BASE_COLUMNS and any(token in col for token in FORBIDDEN_FEATURE_TOKENS)]
    rows.append({"label": "forbidden_feature_tokens", "status": "FAIL" if forbidden_columns else "PASS", "mismatch_count": len(forbidden_columns), "max_abs_diff": np.nan})
    if forbidden_columns:
        findings.append(finding("BLOCKER", "label_alignment", "forbidden_feature_tokens", "FAIL", f"features.csv包含疑似未来/交易结果字段: {forbidden_columns}", len(forbidden_columns)))

    return pd.DataFrame(rows), findings


def calculate_signal_scores(features: pd.DataFrame, selected: pd.DataFrame, feature_summary: pd.DataFrame, config: AuditConfig) -> Tuple[pd.DataFrame, float, float]:
    working = features.copy()
    selected_features = selected["feature"].tolist() if "feature" in selected.columns else []
    fs = feature_summary.set_index("feature")
    train_mask = working["datetime"] <= config.train_end
    score_parts = []
    for feature in selected_features:
        direction = np.sign(fs.loc[feature, "spearman_ic_5"]) if feature in fs.index else np.nan
        train_values = pd.to_numeric(working.loc[train_mask, feature], errors="coerce").dropna()
        if train_values.empty or pd.isna(direction):
            continue
        mean = train_values.mean()
        std = train_values.std(ddof=0)
        if std == 0 or pd.isna(std):
            continue
        score_parts.append(direction * ((pd.to_numeric(working[feature], errors="coerce") - mean) / std))
    working["recalc_signal_score"] = pd.concat(score_parts, axis=1).mean(axis=1) if score_parts else np.nan
    train_scores = working.loc[train_mask, "recalc_signal_score"].dropna()
    long_threshold = float(train_scores.quantile(0.8)) if len(train_scores) else np.nan
    short_threshold = float(train_scores.quantile(0.2)) if len(train_scores) else np.nan
    return working, long_threshold, short_threshold


def audit_strategy(config: AuditConfig, data: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    features = data["features"].copy()
    selected = data.get("selected_features", pd.DataFrame()).copy()
    feature_summary = data.get("feature_summary", pd.DataFrame()).copy()
    trades = data.get("trades_mvp", pd.DataFrame()).copy()
    strategy_summary = data.get("strategy_summary", pd.DataFrame()).copy()

    expected = feature_summary.copy()
    expected = expected[expected["spearman_ic_5"].notna()]
    expected = expected[expected["coverage"] >= 0.5]
    expected = expected[expected["spearman_ic_5"].abs() > 0]
    expected = expected.reindex(expected["spearman_ic_5"].abs().sort_values(ascending=False).index).head(5)
    selected_list = selected["feature"].tolist() if "feature" in selected.columns else []
    expected_list = expected["feature"].tolist()
    selected_ok = selected_list == expected_list
    rows.append({"check_id": "selected_features_top5", "severity": "BLOCKER", "status": "PASS" if selected_ok else "FAIL", "message": "selected_features等于训练集abs(IC)前5", "value": ",".join(selected_list), "expected": ",".join(expected_list), "row_count": int(not selected_ok)})
    if not selected_ok:
        findings.append(finding("BLOCKER", "strategy", "selected_features_top5", "FAIL", "selected_features不等于训练集abs(IC)前5", 1, f"value={selected_list}, expected={expected_list}"))

    scored, long_threshold, short_threshold = calculate_signal_scores(features, selected, feature_summary, config)
    if trades.empty:
        findings.append(finding("OBSERVATION", "strategy", "no_trades", "INFO", "未生成交易"))
        return pd.DataFrame(rows), findings

    for col in ["signal_datetime", "entry_datetime", "exit_datetime"]:
        trades[col] = pd.to_datetime(trades[col], errors="coerce")
    scored_index = scored.sort_values(["contract", "datetime"]).reset_index(drop=True)
    scored_index["row_i"] = scored_index.groupby("contract").cumcount()
    lookup = scored_index.set_index(["contract", "datetime"])
    price_lookup = scored_index.set_index(["contract", "row_i"])

    threshold_bad = 0
    signal_score_bad = 0
    price_bad = 0
    return_bad = 0
    for _, trade in trades.iterrows():
        key = (trade["contract"], trade["signal_datetime"])
        if key not in lookup.index:
            signal_score_bad += 1
            continue
        signal_row = lookup.loc[key]
        score = signal_row["recalc_signal_score"]
        if pd.notna(score) and abs(score - trade["signal_score"]) > 1e-10:
            signal_score_bad += 1
        if trade["side"] == "long" and not (trade["signal_score"] >= long_threshold - 1e-12):
            threshold_bad += 1
        if trade["side"] == "short" and not (trade["signal_score"] <= short_threshold + 1e-12):
            threshold_bad += 1

        signal_i = int(signal_row["row_i"])
        entry_key = (trade["contract"], signal_i + 1)
        exit_key = (trade["contract"], signal_i + 5)
        if entry_key not in price_lookup.index or exit_key not in price_lookup.index:
            price_bad += 1
            continue
        entry_row = price_lookup.loc[entry_key]
        exit_row = price_lookup.loc[exit_key]
        if entry_row["datetime"] != trade["entry_datetime"] or exit_row["datetime"] != trade["exit_datetime"]:
            price_bad += 1
        if abs(entry_row["open"] - trade["entry_price"]) > 1e-10 or abs(exit_row["close"] - trade["exit_price"]) > 1e-10:
            price_bad += 1
        if trade["side"] == "long":
            gross = exit_row["close"] / entry_row["open"] - 1
        else:
            gross = entry_row["open"] / exit_row["close"] - 1
        net = gross - (2.0 * config.tick_size) / entry_row["open"]
        if abs(gross - trade["gross_return"]) > 1e-10 or abs(net - trade["net_return"]) > 1e-10:
            return_bad += 1

    for check_id, bad, msg in [
        ("signal_score_recalc", signal_score_bad, "交易signal_score可重算"),
        ("threshold_direction", threshold_bad, "交易方向符合80/20阈值"),
        ("entry_exit_price", price_bad, "入场/出场时间和价格可重算"),
        ("trade_return_recalc", return_bad, "每笔交易收益可重算"),
    ]:
        rows.append({"check_id": check_id, "severity": "BLOCKER", "status": "FAIL" if bad else "PASS", "message": msg, "value": bad, "expected": 0, "row_count": bad})
        if bad:
            findings.append(finding("BLOCKER", "strategy", check_id, "FAIL", msg, bad))

    overlap_count = 0
    for _, group in trades.groupby("contract"):
        g = group.sort_values("entry_datetime")
        overlap_count += int((g["entry_datetime"].shift(-1) < g["exit_datetime"]).sum())
    rows.append({"check_id": "trade_overlap", "severity": "BLOCKER", "status": "FAIL" if overlap_count else "PASS", "message": "同合约交易不重叠", "value": overlap_count, "expected": 0, "row_count": overlap_count})
    if overlap_count:
        findings.append(finding("BLOCKER", "strategy", "trade_overlap", "FAIL", "同合约交易重叠", overlap_count))

    boundary_count = int(((trades["split"] == "train") & (trades["exit_datetime"] >= config.test_start)).sum())
    rows.append({"check_id": "train_test_boundary", "severity": "BLOCKER", "status": "FAIL" if boundary_count else "PASS", "message": "训练集交易不穿越测试边界", "value": boundary_count, "expected": 0, "row_count": boundary_count})
    if boundary_count:
        findings.append(finding("BLOCKER", "strategy", "train_test_boundary", "FAIL", "训练集交易穿越测试边界", boundary_count))

    if not strategy_summary.empty and "avg_net_return" in strategy_summary.columns:
        test_row = strategy_summary[strategy_summary["split"] == "test"]
        if not test_row.empty and float(test_row.iloc[0]["avg_net_return"]) < 0:
            findings.append(finding("OBSERVATION", "strategy", "negative_test_return", "INFO", "测试集平均净收益为负，属于MVP反馈而非阻断", int(test_row.iloc[0]["trades"]), f"avg_net_return={test_row.iloc[0]['avg_net_return']}"))
    return pd.DataFrame(rows), findings


def audit_extreme_rows(data: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    features = data["features"]
    rows = []
    findings = []
    oi_threshold = float(pd.to_numeric(features["open_interest_change"], errors="coerce").abs().quantile(0.999))
    rules = [
        ("range_ticks", lambda s: s > 100, "> 100"),
        ("poc_distance_ticks", lambda s: s.abs() > 100, "abs > 100"),
        ("poc_shift_ticks", lambda s: s.abs() > 100, "abs > 100"),
        ("open_interest_change", lambda s: s.abs() > oi_threshold, f"abs > 99.9% quantile ({oi_threshold:.4f})"),
    ]
    for field, mask_fn, threshold in rules:
        s = pd.to_numeric(features[field], errors="coerce")
        mask = mask_fn(s)
        bad = features.loc[mask, ["contract", "datetime", "open", "high", "low", "close", "volume", "poc", "delta", "open_interest", field]].copy()
        bad["field"] = field
        bad["value"] = bad[field]
        bad["threshold"] = threshold
        rows.append(bad.drop(columns=[field]))
        if len(bad):
            findings.append(finding("WARNING", "extreme_rows", field, "WARN", f"{field} 出现极端值", len(bad), threshold))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, findings


def audit_report_consistency(config: AuditConfig, data: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    features = data["features"]
    strategy = data.get("strategy_summary", pd.DataFrame())
    rows = []
    findings = []
    report_paths = [config.stage1_dir / "stage1_mvp_report.md", config.stage1_dir / "stage1_mvp_report_zh.md"]
    for path in report_paths:
        exists = path.exists()
        text = path.read_text(encoding="utf-8") if exists else ""
        checks = {
            "rows": str(len(features)) in text,
            "contracts": str(features["contract"].nunique()) in text,
        }
        if not strategy.empty and "trades" in strategy.columns:
            all_row = strategy[strategy["split"] == "all"]
            if not all_row.empty:
                checks["all_trades"] = str(int(all_row.iloc[0]["trades"])) in text
        status = "PASS" if exists and all(checks.values()) else "WARN"
        rows.append({"report": path.name, "status": status, "checks": str(checks)})
        if status != "PASS":
            findings.append(finding("WARNING", "report_consistency", path.name, "WARN", "报告核心数字可能与CSV不一致或报告缺失", 1, str(checks)))
    findings.append(finding("WARNING", "report_consistency", "feature_quintiles_full_sample", "WARN", "feature_quintiles.csv 当前是全样本描述统计，不应作为训练期选特征依据"))
    return pd.DataFrame(rows), findings


def write_markdown_report(config: AuditConfig, findings_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    blocker_count = int((findings_df["severity"] == "BLOCKER").sum()) if not findings_df.empty else 0
    warning_count = int((findings_df["severity"] == "WARNING").sum()) if not findings_df.empty else 0
    observation_count = int((findings_df["severity"] == "OBSERVATION").sum()) if not findings_df.empty else 0
    conclusion = "未发现 BLOCKER，第一阶段未发现明显未来函数或跨合约泄漏。" if blocker_count == 0 else "发现 BLOCKER，需要先修复再进入下一阶段。"

    lines = [
        "# 第一阶段全面审计报告",
        "",
        "## 总结",
        "",
        f"- 结论：{conclusion}",
        f"- BLOCKER：`{blocker_count}`",
        f"- WARNING：`{warning_count}`",
        f"- OBSERVATION：`{observation_count}`",
        "",
        "## 审计摘要",
        "",
        summary_df.to_markdown(index=False) if not summary_df.empty else "无摘要。",
        "",
        "## 主要发现",
        "",
        findings_df.to_markdown(index=False) if not findings_df.empty else "没有发现问题。",
        "",
        "## 解释",
        "",
        "- BLOCKER 代表必须修复的问题，例如未来函数、标签错位、跨合约泄漏、策略重叠持仓。",
        "- WARNING 代表第一阶段需要重点关注的问题，例如极端值、特征高相关、报告解释风险。",
        "- OBSERVATION 代表研究反馈，例如策略亏损或IC偏弱，不等于代码错误。",
        "",
        "## 下一步建议",
        "",
        "- 优先查看 `extreme_rows.csv`，确认极端Bar是否来自换月、流动性枯竭或异常行情。",
        "- 查看 `feature_correlation.csv`，决定第二阶段是否合并高度相关特征。",
        "- 将 `feature_quintiles.csv` 明确标注为全样本描述统计；若用于选特征，应改成训练集统计。",
        "- 第二阶段引入市场状态分层、walk-forward、真实成本和角色化Alpha组合。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(findings_df: pd.DataFrame) -> pd.DataFrame:
    if findings_df.empty:
        return pd.DataFrame([{"severity": "ALL", "count": 0}])
    summary = findings_df.groupby(["severity", "module"], dropna=False).size().reset_index(name="count")
    overall = findings_df.groupby("severity").size().reset_index(name="count")
    overall["module"] = "ALL"
    return pd.concat([overall[["severity", "module", "count"]], summary], ignore_index=True)


def run(config: AuditConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_stage1(config)
    orderflow = load_orderflow(config.orderflow_dir)

    all_findings: List[Dict[str, Any]] = []
    data_integrity, findings = audit_data_integrity(config, data, orderflow)
    all_findings.extend(findings)

    if "features" in data and "labels" in data:
        feature_quality, feature_correlation, feature_group_diversity, findings = audit_feature_quality(data)
        all_findings.extend(findings)
        label_alignment, findings = audit_label_alignment(config, data)
        all_findings.extend(findings)
        strategy_audit, findings = audit_strategy(config, data)
        all_findings.extend(findings)
        extreme_rows, findings = audit_extreme_rows(data)
        all_findings.extend(findings)
        report_consistency, findings = audit_report_consistency(config, data)
        all_findings.extend(findings)
    else:
        feature_quality = pd.DataFrame()
        feature_correlation = pd.DataFrame()
        feature_group_diversity = pd.DataFrame()
        label_alignment = pd.DataFrame()
        strategy_audit = pd.DataFrame()
        extreme_rows = pd.DataFrame()
        report_consistency = pd.DataFrame()

    findings_df = pd.DataFrame(all_findings, columns=["severity", "module", "check_id", "status", "message", "row_count", "detail"])
    summary_df = build_summary(findings_df)

    data_integrity.to_csv(config.output_dir / "data_integrity.csv", index=False, encoding="utf-8-sig")
    feature_quality.to_csv(config.output_dir / "feature_quality.csv", index=False, encoding="utf-8-sig")
    feature_correlation.to_csv(config.output_dir / "feature_correlation.csv", index=False, encoding="utf-8-sig")
    feature_group_diversity.to_csv(config.output_dir / "feature_group_diversity.csv", index=False, encoding="utf-8-sig")
    label_alignment.to_csv(config.output_dir / "label_alignment.csv", index=False, encoding="utf-8-sig")
    strategy_audit.to_csv(config.output_dir / "strategy_audit.csv", index=False, encoding="utf-8-sig")
    extreme_rows.to_csv(config.output_dir / "extreme_rows.csv", index=False, encoding="utf-8-sig")
    report_consistency.to_csv(config.output_dir / "report_consistency.csv", index=False, encoding="utf-8-sig")
    findings_df.to_csv(config.output_dir / "audit_findings.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(config.output_dir / "audit_summary.csv", index=False, encoding="utf-8-sig")
    write_markdown_report(config, findings_df, summary_df, config.output_dir / "stage1_audit_report_zh.md")

    blocker_count = int((findings_df["severity"] == "BLOCKER").sum()) if not findings_df.empty else 0
    warning_count = int((findings_df["severity"] == "WARNING").sum()) if not findings_df.empty else 0
    observation_count = int((findings_df["severity"] == "OBSERVATION").sum()) if not findings_df.empty else 0
    print(f"output_dir={config.output_dir}")
    print(f"blockers={blocker_count}, warnings={warning_count}, observations={observation_count}")
    if config.fail_on_blocker and blocker_count:
        return 1
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
