"""阶段二 P0 阻断审计。

本模块只做进入阶段三前的红线检查，不修改阶段二研究逻辑。审计范围覆盖：
标签对齐、训练/测试隔离、交易收益复算、Triple Barrier 几何与路径抽查、
持仓重叠、边界穿越、底层 Bar 完整性和极端值定位。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .io import read_csv, write_csv


DEFAULT_STAGE2_DIR = Path("/Users/wangrendong/Projects/BlueBlue/stage2_outputs/SHFE.cu")
DEFAULT_OUTPUT_SUBDIR = "blocker_audit"
DEFAULT_TRAIN_END = "2024-03-23 00:55:00"
DEFAULT_TEST_START = "2024-03-25 09:00:00"

BASE_COLUMNS = ["contract", "datetime", "open", "high", "low", "close", "volume", "poc", "delta", "open_interest"]
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
SHIFT_FEATURES = [
    "close_to_prev_close_return",
    "gap_return",
    "delta_change_1",
    "poc_shift_ticks",
    "open_interest_change",
    "open_interest_change_pct",
]
LEAKAGE_PATTERNS = ("future_return", "mfe_", "mae_", "entry_", "exit_", "trade_return", "net_return")
CORE_TRADE_FILES = [
    ("stage2_strategy", Path("trades_stage2.csv")),
    ("gain_curve", Path("gain_curve/gain_curve_trades.csv")),
    ("trend_pullback_gain", Path("trend_pullback_gain/trend_pullback_gain_trades.csv")),
    ("trend_pullback_sweep_long", Path("trend_pullback_sweep_ls/long/trend_pullback_sweep_trades.csv")),
    ("trend_pullback_sweep_short", Path("trend_pullback_sweep_ls/short/trend_pullback_sweep_trades.csv")),
]


@dataclass(frozen=True)
class BlockerAuditConfig:
    stage2_dir: Path
    output_dir: Path
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    holding_bars: int
    cost_ticks_roundtrip: float
    tick_size: float
    barrier_sample_per_file: int
    random_seed: int
    fail_on_blocker: bool

    @property
    def roundtrip_cost_price(self) -> float:
        return self.cost_ticks_roundtrip * self.tick_size


def parse_args(argv: Optional[Sequence[str]] = None) -> BlockerAuditConfig:
    parser = argparse.ArgumentParser(description="Stage 2 P0 blocker audit before walk-forward")
    parser.add_argument("--stage2_dir", type=str, default=str(DEFAULT_STAGE2_DIR))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--test_start", type=str, default=DEFAULT_TEST_START)
    parser.add_argument("--holding_bars", type=int, default=5)
    parser.add_argument("--cost_ticks_roundtrip", type=float, default=2.0)
    parser.add_argument("--tick_size", type=float, default=10.0)
    parser.add_argument("--barrier_sample_per_file", type=int, default=20000)
    parser.add_argument("--random_seed", type=int, default=20260528)
    parser.add_argument("--fail_on_blocker", action="store_true")
    args = parser.parse_args(argv)

    stage2_dir = Path(args.stage2_dir).expanduser()
    return BlockerAuditConfig(
        stage2_dir=stage2_dir,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else stage2_dir / DEFAULT_OUTPUT_SUBDIR,
        train_end=pd.Timestamp(args.train_end),
        test_start=pd.Timestamp(args.test_start),
        holding_bars=int(args.holding_bars),
        cost_ticks_roundtrip=float(args.cost_ticks_roundtrip),
        tick_size=float(args.tick_size),
        barrier_sample_per_file=int(args.barrier_sample_per_file),
        random_seed=int(args.random_seed),
        fail_on_blocker=bool(args.fail_on_blocker),
    )


def _finding(
    category: str,
    check_id: str,
    severity: str,
    status: str,
    affected_rows: int,
    message: str,
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "category": category,
        "check_id": check_id,
        "severity": severity,
        "status": status,
        "affected_rows": int(affected_rows),
        "message": message,
        "detail": detail,
    }


def _parse_datetime_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _numeric_equal(left: pd.Series, right: pd.Series, tol: float = 1e-10) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_nan = left_num.isna() & right_num.isna()
    return both_nan | ((left_num - right_num).abs() <= tol)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _safe_read(path: Path, parse_dates: Iterable[str] = ()) -> pd.DataFrame:
    df = read_csv(path)
    return _parse_datetime_columns(df, parse_dates)


def _forward_max(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def _forward_min(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def _recompute_labels(features: pd.DataFrame, config: BlockerAuditConfig) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for _, group in features.groupby("contract", sort=True):
        g = group.sort_values("datetime").copy()
        close = g["close"].replace(0, np.nan)
        for horizon in (1, 3, 5, 10):
            g[f"future_return_{horizon}"] = g["close"].shift(-horizon) / close - 1.0
        g["mfe_10_long"] = _forward_max(g["high"], 10) / close - 1.0
        g["mae_10_long"] = _forward_min(g["low"], 10) / close - 1.0
        g["entry_open_next"] = g["open"].shift(-1)
        g["exit_close_after_5"] = g["close"].shift(-config.holding_bars)
        entry = g["entry_open_next"].replace(0, np.nan)
        g["trade_return_5_gross"] = g["exit_close_after_5"] / entry - 1.0
        g["trade_return_5_net"] = g["trade_return_5_gross"] - config.roundtrip_cost_price / entry
        frames.append(g[["contract", "datetime"] + LABEL_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def audit_label_alignment(features: pd.DataFrame, labels: pd.DataFrame, config: BlockerAuditConfig) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    expected = _recompute_labels(features, config)
    actual = labels[["contract", "datetime"] + [col for col in LABEL_COLUMNS if col in labels.columns]].copy()
    merged = expected.merge(actual, on=["contract", "datetime"], how="outer", suffixes=("_expected", "_actual"), indicator=True)
    key_mismatch = int((merged["_merge"] != "both").sum())
    rows.append(
        {
            "check": "label_key_alignment",
            "field": "contract_datetime",
            "status": "PASS" if key_mismatch == 0 else "FAIL",
            "mismatch_count": key_mismatch,
            "first_contract": "",
            "first_datetime": "",
            "expected_value": "",
            "actual_value": "",
        }
    )
    findings.append(
        _finding(
            "label_alignment",
            "label_key_alignment",
            "BLOCKER",
            "PASS" if key_mismatch == 0 else "FAIL",
            key_mismatch,
            "stage2_features 与 stage2_labels 的 contract+datetime 键必须完全一致",
        )
    )

    both = merged[merged["_merge"] == "both"].copy()
    total_mismatch = 0
    for col in LABEL_COLUMNS:
        exp_col = f"{col}_expected"
        act_col = f"{col}_actual"
        if exp_col not in both.columns or act_col not in both.columns:
            mismatch_count = len(both)
            first = both.head(1)
        else:
            equal = _numeric_equal(both[exp_col], both[act_col])
            mismatch_count = int((~equal).sum())
            first = both.loc[~equal].head(1)
        total_mismatch += mismatch_count
        rows.append(
            {
                "check": "label_value_alignment",
                "field": col,
                "status": "PASS" if mismatch_count == 0 else "FAIL",
                "mismatch_count": mismatch_count,
                "first_contract": first["contract"].iloc[0] if not first.empty else "",
                "first_datetime": first["datetime"].iloc[0] if not first.empty else "",
                "expected_value": _format_value(first[exp_col].iloc[0]) if not first.empty and exp_col in first else "",
                "actual_value": _format_value(first[act_col].iloc[0]) if not first.empty and act_col in first else "",
            }
        )
    findings.append(
        _finding(
            "label_alignment",
            "label_value_alignment",
            "BLOCKER",
            "PASS" if total_mismatch == 0 else "FAIL",
            total_mismatch,
            "所有 future_return、MFE/MAE、entry/exit、固定持有收益必须可逐行复算",
        )
    )

    tail_rows: List[Dict[str, Any]] = []
    tail_rules = {
        "future_return_1": 1,
        "future_return_3": 3,
        "future_return_5": 5,
        "future_return_10": 10,
        "mfe_10_long": 10,
        "mae_10_long": 10,
        "entry_open_next": 1,
        "exit_close_after_5": config.holding_bars,
        "trade_return_5_gross": config.holding_bars,
        "trade_return_5_net": config.holding_bars,
    }
    sorted_labels = labels.sort_values(["contract", "datetime"]).copy()
    for field, n_tail in tail_rules.items():
        if field not in sorted_labels.columns:
            bad = len(sorted_labels)
            first_contract = ""
            first_datetime = ""
        else:
            tail = sorted_labels.groupby("contract", sort=True).tail(n_tail)
            bad_mask = tail[field].notna()
            bad = int(bad_mask.sum())
            first = tail.loc[bad_mask].head(1)
            first_contract = first["contract"].iloc[0] if not first.empty else ""
            first_datetime = first["datetime"].iloc[0] if not first.empty else ""
        tail_rows.append(
            {
                "check": "contract_tail_future_labels_empty",
                "field": field,
                "status": "PASS" if bad == 0 else "FAIL",
                "mismatch_count": bad,
                "first_contract": first_contract,
                "first_datetime": first_datetime,
                "expected_value": "NaN",
                "actual_value": "not_null" if bad else "",
            }
        )
    rows.extend(tail_rows)
    tail_bad = int(sum(row["mismatch_count"] for row in tail_rows))
    findings.append(
        _finding(
            "label_alignment",
            "contract_tail_future_labels_empty",
            "BLOCKER",
            "PASS" if tail_bad == 0 else "FAIL",
            tail_bad,
            "每个合约尾部未来标签必须为空，防止跨合约泄漏",
        )
    )

    leak_cols = [col for col in features.columns if any(pattern in col.lower() for pattern in LEAKAGE_PATTERNS)]
    rows.append(
        {
            "check": "feature_file_no_future_or_trade_columns",
            "field": ",".join(leak_cols),
            "status": "PASS" if not leak_cols else "FAIL",
            "mismatch_count": len(leak_cols),
            "first_contract": "",
            "first_datetime": "",
            "expected_value": "no leakage columns",
            "actual_value": ",".join(leak_cols),
        }
    )
    findings.append(
        _finding(
            "label_alignment",
            "feature_file_no_future_or_trade_columns",
            "BLOCKER",
            "PASS" if not leak_cols else "FAIL",
            len(leak_cols),
            "stage2_features 不允许包含未来收益、MFE/MAE、entry/exit 或交易结果字段",
        )
    )

    shift_bad = 0
    shift_details = []
    first_rows = features.sort_values(["contract", "datetime"]).groupby("contract", sort=True).head(1)
    for col in SHIFT_FEATURES:
        if col in first_rows.columns:
            bad = int(first_rows[col].notna().sum())
            if bad:
                shift_details.append(f"{col}:{bad}")
            shift_bad += bad
    rows.append(
        {
            "check": "contract_first_shift_features_empty",
            "field": ",".join(SHIFT_FEATURES),
            "status": "PASS" if shift_bad == 0 else "FAIL",
            "mismatch_count": shift_bad,
            "first_contract": "",
            "first_datetime": "",
            "expected_value": "NaN at contract first row",
            "actual_value": ";".join(shift_details),
        }
    )
    findings.append(
        _finding(
            "label_alignment",
            "contract_first_shift_features_empty",
            "BLOCKER",
            "PASS" if shift_bad == 0 else "FAIL",
            shift_bad,
            "跨合约 rolling/shift 的哨兵检查：每个合约首行的 shift 特征应为空",
        )
    )

    return pd.DataFrame(rows), findings


def audit_data_integrity(features: pd.DataFrame, config: BlockerAuditConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    findings: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    def add(check_id: str, severity: str, bad_count: int, message: str, detail: str = "") -> None:
        rows.append({"check_id": check_id, "severity": severity, "status": "PASS" if bad_count == 0 else "FAIL", "affected_rows": int(bad_count), "message": message, "detail": detail})
        findings.append(_finding("data_integrity", check_id, severity, "PASS" if bad_count == 0 else "FAIL", bad_count, message, detail))

    missing_core = [col for col in BASE_COLUMNS if col not in features.columns]
    add("required_base_columns", "BLOCKER", len(missing_core), "核心 Bar 字段必须存在", ",".join(missing_core))

    invalid_dt = int(features["datetime"].isna().sum()) if "datetime" in features.columns else len(features)
    add("datetime_parse", "BLOCKER", invalid_dt, "datetime 必须可解析")

    dup_count = int(features.duplicated(["contract", "datetime"]).sum()) if {"contract", "datetime"}.issubset(features.columns) else len(features)
    add("duplicate_contract_datetime", "BLOCKER", dup_count, "contract+datetime 不允许重复")

    monotonic_bad = 0
    if {"contract", "datetime"}.issubset(features.columns):
        for _, group in features.sort_values(["contract", "datetime"]).groupby("contract", sort=True):
            diffs = group["datetime"].diff().dropna()
            if (diffs <= pd.Timedelta(0)).any():
                monotonic_bad += 1
    add("contract_datetime_monotonic", "BLOCKER", monotonic_bad, "单合约 datetime 必须递增")

    if {"open", "high", "low", "close"}.issubset(features.columns):
        ohlc_bad = int(
            (
                (features["high"] < features["open"])
                | (features["high"] < features["close"])
                | (features["high"] < features["low"])
                | (features["low"] > features["open"])
                | (features["low"] > features["close"])
            ).sum()
        )
    else:
        ohlc_bad = len(features)
    add("ohlc_structure", "BLOCKER", ohlc_bad, "OHLC 结构必须满足 high>=open/close/low 且 low<=open/close")

    volume_bad = int((features["volume"] < 0).sum()) if "volume" in features.columns else len(features)
    add("volume_non_negative", "BLOCKER", volume_bad, "volume 不允许为负")

    if {"delta", "volume"}.issubset(features.columns):
        delta_bad = int((features["delta"].abs() > features["volume"]).sum())
    else:
        delta_bad = len(features)
    add("delta_abs_lte_volume", "BLOCKER", delta_bad, "abs(delta) 必须小于等于 volume")

    if {"poc", "low", "high"}.issubset(features.columns):
        poc_mask = features["poc"].notna()
        poc_bad = int(((features.loc[poc_mask, "poc"] < features.loc[poc_mask, "low"]) | (features.loc[poc_mask, "poc"] > features.loc[poc_mask, "high"])).sum())
    else:
        poc_bad = len(features)
    add("poc_inside_bar_range", "BLOCKER", poc_bad, "poc 必须落在 [low, high] 内")

    oi_bad = int((features["open_interest"] < 0).sum()) if "open_interest" in features.columns else 0
    add("open_interest_non_negative", "BLOCKER", oi_bad, "open_interest 不允许为负")

    feature_rows: List[Dict[str, Any]] = []
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        series = pd.to_numeric(features[col], errors="coerce")
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        inf_count = int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum())
        null_count = int(series.isna().sum())
        unique_count = int(finite.nunique()) if len(finite) else 0
        zero_variance = bool(unique_count <= 1 and len(finite) > 0)
        feature_rows.append(
            {
                "field": col,
                "null_count": null_count,
                "null_rate": null_count / len(features) if len(features) else np.nan,
                "inf_count": inf_count,
                "unique_count": unique_count,
                "zero_variance": zero_variance,
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
            }
        )
    feature_nan_inf = pd.DataFrame(feature_rows)
    total_inf = int(feature_nan_inf["inf_count"].sum()) if not feature_nan_inf.empty else 0
    findings.append(_finding("data_integrity", "feature_inf_count", "BLOCKER", "PASS" if total_inf == 0 else "FAIL", total_inf, "特征和基础字段不允许出现 inf"))
    core_null = int(features[[col for col in BASE_COLUMNS if col in features.columns]].isna().sum().sum())
    findings.append(_finding("data_integrity", "core_field_null_count", "BLOCKER", "PASS" if core_null == 0 else "FAIL", core_null, "核心字段不应缺失"))

    extreme_specs = [
        ("true_range_ticks", "high"),
        ("delta_strength", "abs"),
        ("poc_shift_ticks", "abs"),
        ("open_interest_change", "abs"),
        ("volume_zscore_20", "abs"),
    ]
    extreme_rows: List[Dict[str, Any]] = []
    for field, mode in extreme_specs:
        if field not in features.columns:
            continue
        series = pd.to_numeric(features[field], errors="coerce")
        metric = series.abs() if mode == "abs" else series
        threshold = float(metric.quantile(0.999)) if metric.notna().any() else np.nan
        mask = metric > threshold if pd.notna(threshold) else pd.Series(False, index=features.index)
        part = features.loc[mask, ["contract", "datetime", "open", "high", "low", "close", field]].copy()
        part["field"] = field
        part["threshold_p999"] = threshold
        part["metric_value"] = metric.loc[mask].values
        extreme_rows.extend(part.sort_values("metric_value", ascending=False).head(500).to_dict("records"))
    extreme_df = pd.DataFrame(extreme_rows)
    findings.append(_finding("data_integrity", "extreme_bar_locator", "WARNING", "WARN" if not extreme_df.empty else "PASS", len(extreme_df), "极端 Bar 已定位；默认不作为阻断项"))

    return pd.DataFrame(rows), feature_nan_inf, extreme_df, findings


def audit_train_test_isolation(config: BlockerAuditConfig) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []

    def add(check_id: str, severity: str, bad_count: int, message: str, detail: str = "") -> None:
        rows.append({"check_id": check_id, "severity": severity, "status": "PASS" if bad_count == 0 else "FAIL", "affected_rows": int(bad_count), "message": message, "detail": detail})
        findings.append(_finding("train_test_isolation", check_id, severity, "PASS" if bad_count == 0 else "FAIL", bad_count, message, detail))

    stage2_dir = config.stage2_dir
    train_summary_path = stage2_dir / "feature_summary_train.csv"
    selected_path = stage2_dir / "selected_features_stage2.csv"
    if train_summary_path.exists() and selected_path.exists():
        train_summary = read_csv(train_summary_path)
        selected = read_csv(selected_path)
        merged = selected.merge(train_summary[["feature", "spearman_ic_5", "coverage", "unique_count"]], on="feature", how="left", suffixes=("_selected", "_train"))
        bad = int(
            (
                merged["spearman_ic_5_train"].isna()
                | ((merged["spearman_ic_5_selected"] - merged["spearman_ic_5_train"]).abs() > 1e-12)
                | ((merged["coverage_selected"] - merged["coverage_train"]).abs() > 1e-12)
                | (merged["unique_count_selected"].astype(float) != merged["unique_count_train"].astype(float))
            ).sum()
        )
        add("selected_features_from_train_summary", "BLOCKER", bad, "selected_features_stage2 必须与训练集 feature_summary_train 一致")
    else:
        add("selected_features_from_train_summary", "BLOCKER", 1, "缺少 selected_features_stage2.csv 或 feature_summary_train.csv")

    transform_path = stage2_dir / "feature_transform_params.csv"
    if transform_path.exists():
        transform = read_csv(transform_path)
        source_text = transform.get("param_source", pd.Series(dtype=str)).astype(str).str.lower()
        bad = int(source_text.str.contains("test").sum())
        missing_train_source = int((~source_text.str.contains("train")).sum())
        add("transform_params_no_test_source", "BLOCKER", bad, "winsorize/标准化参数不允许来自测试集")
        add("transform_params_train_source_declared", "BLOCKER", missing_train_source, "标准化参数必须声明训练集来源")
    else:
        add("transform_params_exists", "BLOCKER", 1, "缺少 feature_transform_params.csv")

    signal_path = stage2_dir / "signal_scores.csv"
    trades_path = stage2_dir / "trades_stage2.csv"
    if signal_path.exists() and trades_path.exists():
        signal = _safe_read(signal_path, ["datetime"])
        trades = _safe_read(trades_path, ["signal_datetime", "entry_datetime", "exit_datetime"])
        train_scores = signal.loc[(signal["datetime"] <= config.train_end) & signal["risk_filter_pass"].fillna(False), "signal_score"].dropna()
        if len(train_scores) and not trades.empty:
            expected_long = float(train_scores.quantile(0.85))
            expected_short = float(train_scores.quantile(0.15))
            long_bad = int((pd.to_numeric(trades["long_threshold"], errors="coerce") - expected_long).abs().gt(1e-12).sum()) if "long_threshold" in trades.columns else len(trades)
            short_bad = int((pd.to_numeric(trades["short_threshold"], errors="coerce") - expected_short).abs().gt(1e-12).sum()) if "short_threshold" in trades.columns else len(trades)
            add("stage2_thresholds_from_train_scores", "BLOCKER", long_bad + short_bad, "主策略 long/short 阈值必须由训练集 signal_score 分位数确定", f"expected_long={expected_long}, expected_short={expected_short}")
        else:
            add("stage2_thresholds_from_train_scores", "BLOCKER", 1, "无法从训练集 signal_scores 复算阈值")
    else:
        add("stage2_thresholds_from_train_scores", "BLOCKER", 1, "缺少 signal_scores.csv 或 trades_stage2.csv")

    gain_selected_path = stage2_dir / "gain_curve/gain_curve_selected_features.csv"
    if train_summary_path.exists() and gain_selected_path.exists():
        train_summary = read_csv(train_summary_path)
        gain_selected = read_csv(gain_selected_path)
        merged = gain_selected.merge(train_summary[["feature", "spearman_ic_5", "coverage", "unique_count"]], on="feature", how="left", suffixes=("_selected", "_train"))
        bad = int(
            (
                merged["spearman_ic_5_train"].isna()
                | ((merged["spearman_ic_5_selected"] - merged["spearman_ic_5_train"]).abs() > 1e-12)
                | ((merged["coverage_selected"] - merged["coverage_train"]).abs() > 1e-12)
                | (merged["unique_count_selected"].astype(float) != merged["unique_count_train"].astype(float))
            ).sum()
        )
        add("gain_curve_selected_features_from_train", "BLOCKER", bad, "增益曲线选特征必须使用训练集 feature_summary")
    elif gain_selected_path.exists():
        add("gain_curve_selected_features_from_train", "BLOCKER", 1, "无法校验增益曲线选特征训练来源")

    gain_summary_path = stage2_dir / "gain_curve/gain_curve_summary.csv"
    if gain_summary_path.exists():
        gain_summary = read_csv(gain_summary_path)
        bad = 0
        for _, part in gain_summary.groupby("combo_name"):
            train_part = part[part["split"] == "train"]
            if train_part.empty:
                bad += 1
                continue
            long_threshold = train_part["long_threshold"].iloc[0]
            short_threshold = train_part["short_threshold"].iloc[0]
            bad += int((part["long_threshold"] != long_threshold).sum())
            bad += int((part["short_threshold"] != short_threshold).sum())
        add("gain_curve_thresholds_reused_across_splits", "BLOCKER", bad, "增益曲线训练/测试必须复用同一训练阈值")

    for side in ("long", "short"):
        ranking_path = stage2_dir / f"trend_pullback_sweep_ls/{side}/trend_pullback_parameter_ranking_train.csv"
        verification_path = stage2_dir / f"trend_pullback_sweep_ls/{side}/trend_pullback_parameter_verification_test.csv"
        if ranking_path.exists():
            ranking = read_csv(ranking_path)
            bad = int((ranking.get("split", pd.Series(dtype=str)).astype(str) != "train").sum())
            add(f"sweep_{side}_ranking_train_only", "BLOCKER", bad, f"{side} 参数排序表必须只包含训练集")
        else:
            add(f"sweep_{side}_ranking_train_only", "BLOCKER", 1, f"缺少 {side} 参数排序表")
        if verification_path.exists():
            verification = read_csv(verification_path)
            if "train_rank" in verification.columns:
                recommendation = verification.get("recommendation_status", pd.Series("", index=verification.index)).astype(str)
                bad = int((verification["train_rank"].isna() & (recommendation != "test_only")).sum())
            else:
                bad = len(verification)
            add(f"sweep_{side}_verification_has_train_rank", "BLOCKER", bad, f"{side} 非 test_only 的测试验证行必须带训练排名")
            test_only = int((verification.get("recommendation_status", pd.Series(dtype=str)).astype(str) == "test_only").sum())
            rows.append(
                {
                    "check_id": f"sweep_{side}_test_only_marked",
                    "severity": "WARNING",
                    "status": "WARN" if test_only else "PASS",
                    "affected_rows": test_only,
                    "message": f"{side} 如果出现测试集偶然最优，必须标记为 test_only",
                    "detail": "",
                }
            )
        else:
            add(f"sweep_{side}_verification_has_train_rank", "BLOCKER", 1, f"缺少 {side} 参数验证表")

    asym_path = stage2_dir / "trend_pullback_sweep_ls/summary/long_short_asymmetry.csv"
    if asym_path.exists():
        asym = read_csv(asym_path)
        test_source = int(asym.astype(str).apply(lambda col: col.str.contains("测试集观察最优", regex=False)).any(axis=1).sum())
        rows.append(
            {
                "check_id": "long_short_summary_test_observed_source",
                "severity": "WARNING",
                "status": "WARN" if test_source else "PASS",
                "affected_rows": test_source,
                "message": "汇总报告若展示测试集观察最优，只能作为观察，不能作为正式参数建议",
                "detail": "",
            }
        )
        findings.append(
            _finding(
                "train_test_isolation",
                "long_short_summary_test_observed_source",
                "WARNING",
                "WARN" if test_source else "PASS",
                test_source,
                "汇总报告出现测试集观察最优时需警惕参数建议泄漏；本项为警告，不直接阻断",
            )
        )

    return pd.DataFrame(rows), findings


def _expected_gross(entry: pd.Series, exit_: pd.Series, side: pd.Series) -> pd.Series:
    long_ret = exit_ / entry - 1.0
    short_ret = 1.0 - exit_ / entry
    return pd.Series(np.where(side.astype(str).str.lower() == "short", short_ret, long_ret), index=entry.index)


def _trade_key_columns(df: pd.DataFrame) -> List[str]:
    keys = [col for col in ["case_id", "combo_name", "contract", "side"] if col in df.columns]
    if "contract" not in keys:
        keys.append("contract")
    return keys


def _build_bar_lookup(features: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for contract, group in features.sort_values(["contract", "datetime"]).groupby("contract", sort=True):
        g = group.reset_index(drop=True)
        dt_index = pd.Index(g["datetime"])
        lookup[contract] = {
            "index": {value: i for i, value in enumerate(dt_index)},
            "high": g["high"].to_numpy(dtype=float),
            "low": g["low"].to_numpy(dtype=float),
            "close": g["close"].to_numpy(dtype=float),
        }
    return lookup


def _validate_barrier_path_sample(
    trades: pd.DataFrame,
    features_lookup: Dict[str, Dict[str, Any]],
    sample_size: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, int]:
    if trades.empty or not {"stop_price_barrier", "take_profit_price_barrier", "exit_reason"}.issubset(trades.columns):
        return pd.DataFrame(), 0
    usable = trades.dropna(subset=["entry_datetime", "exit_datetime", "entry_price", "exit_price", "stop_price_barrier", "take_profit_price_barrier"]).copy()
    if usable.empty:
        return pd.DataFrame(), 0
    if len(usable) > sample_size:
        usable = usable.sample(sample_size, random_state=random_seed)

    rows: List[Dict[str, Any]] = []
    bad = 0
    for row in usable.itertuples(index=False):
        contract = getattr(row, "contract")
        side = getattr(row, "side")
        data = features_lookup.get(contract)
        status = "PASS"
        message = ""
        expected_reason = ""
        expected_exit_price = np.nan
        if data is None:
            status = "FAIL"
            message = "contract not found in features"
        else:
            entry_i = data["index"].get(getattr(row, "entry_datetime"))
            exit_i = data["index"].get(getattr(row, "exit_datetime"))
            if entry_i is None or exit_i is None or exit_i < entry_i:
                status = "FAIL"
                message = "entry/exit datetime not found or reversed"
            else:
                stop_price = float(getattr(row, "stop_price_barrier"))
                take_price = float(getattr(row, "take_profit_price_barrier"))
                for pos in range(entry_i, exit_i + 1):
                    if side == "short":
                        if data["high"][pos] >= stop_price:
                            expected_reason = "stop_loss"
                            expected_exit_price = stop_price
                            break
                        if data["low"][pos] <= take_price:
                            expected_reason = "take_profit"
                            expected_exit_price = take_price
                            break
                    else:
                        if data["low"][pos] <= stop_price:
                            expected_reason = "stop_loss"
                            expected_exit_price = stop_price
                            break
                        if data["high"][pos] >= take_price:
                            expected_reason = "take_profit"
                            expected_exit_price = take_price
                            break
                if not expected_reason:
                    expected_reason = "time_barrier" if getattr(row, "exit_reason") in {"time_barrier", "fixed_time"} else getattr(row, "exit_reason")
                    expected_exit_price = data["close"][exit_i]
                reason_ok = str(getattr(row, "exit_reason")) == expected_reason or (str(getattr(row, "exit_reason")) == "fixed_time" and expected_reason == "time_barrier")
                price_ok = abs(float(getattr(row, "exit_price")) - expected_exit_price) <= 1e-8
                if not reason_ok or not price_ok:
                    status = "FAIL"
                    message = f"expected_reason={expected_reason}, expected_exit_price={expected_exit_price}"
        if status != "PASS":
            bad += 1
            if len(rows) < 200:
                rows.append(
                    {
                        "contract": contract,
                        "side": side,
                        "entry_datetime": getattr(row, "entry_datetime"),
                        "exit_datetime": getattr(row, "exit_datetime"),
                        "reported_exit_reason": getattr(row, "exit_reason"),
                        "expected_exit_reason": expected_reason,
                        "reported_exit_price": getattr(row, "exit_price"),
                        "expected_exit_price": expected_exit_price,
                        "status": status,
                        "message": message,
                    }
                )
    return pd.DataFrame(rows), bad


def audit_trade_files(features: pd.DataFrame, config: BlockerAuditConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    repricing_rows: List[Dict[str, Any]] = []
    barrier_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    features_lookup = _build_bar_lookup(features)

    for file_id, rel_path in CORE_TRADE_FILES:
        path = config.stage2_dir / rel_path
        if not path.exists():
            for target, rows in [
                ("trade_repricing", repricing_rows),
                ("triple_barrier", barrier_rows),
                ("trade_overlap", overlap_rows),
                ("boundary_crossing", boundary_rows),
            ]:
                rows.append({"file_id": file_id, "path": str(rel_path), "status": "FAIL", "affected_rows": 1, "message": "file missing"})
                findings.append(_finding(target, f"{file_id}_exists", "BLOCKER", "FAIL", 1, f"缺少交易文件 {rel_path}"))
            continue

        trades = _safe_read(path, ["signal_datetime", "entry_datetime", "exit_datetime"])
        for col in ["entry_price", "exit_price", "gross_return", "net_return"]:
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce")

        required = {"contract", "side", "entry_datetime", "exit_datetime", "entry_price", "exit_price", "gross_return", "net_return"}
        missing = sorted(required - set(trades.columns))
        if missing:
            repricing_rows.append({"file_id": file_id, "path": str(rel_path), "status": "FAIL", "affected_rows": len(trades), "message": f"missing columns: {missing}"})
            findings.append(_finding("trade_repricing", f"{file_id}_required_columns", "BLOCKER", "FAIL", len(trades), f"交易表缺少必要字段: {missing}"))
            continue

        entry = trades["entry_price"].replace(0, np.nan)
        expected_gross = _expected_gross(entry, trades["exit_price"], trades["side"])
        expected_net = expected_gross - config.roundtrip_cost_price / entry
        gross_bad_mask = (trades["gross_return"] - expected_gross).abs() > 1e-10
        net_bad_mask = (trades["net_return"] - expected_net).abs() > 1e-10
        gross_bad = int(gross_bad_mask.fillna(True).sum())
        net_bad = int(net_bad_mask.fillna(True).sum())
        first_bad = trades.loc[gross_bad_mask | net_bad_mask].head(1)
        repricing_rows.append(
            {
                "file_id": file_id,
                "path": str(rel_path),
                "status": "PASS" if gross_bad + net_bad == 0 else "FAIL",
                "rows": len(trades),
                "gross_mismatch_count": gross_bad,
                "net_mismatch_count": net_bad,
                "first_contract": first_bad["contract"].iloc[0] if not first_bad.empty else "",
                "first_side": first_bad["side"].iloc[0] if not first_bad.empty else "",
                "first_entry_price": first_bad["entry_price"].iloc[0] if not first_bad.empty else "",
                "first_exit_price": first_bad["exit_price"].iloc[0] if not first_bad.empty else "",
                "reported_gross_return": first_bad["gross_return"].iloc[0] if not first_bad.empty else "",
                "expected_gross_return": expected_gross.loc[first_bad.index[0]] if not first_bad.empty else "",
                "reported_net_return": first_bad["net_return"].iloc[0] if not first_bad.empty else "",
                "expected_net_return": expected_net.loc[first_bad.index[0]] if not first_bad.empty else "",
            }
        )
        findings.append(
            _finding(
                "trade_repricing",
                f"{file_id}_linear_futures_return_formula",
                "BLOCKER",
                "PASS" if gross_bad + net_bad == 0 else "FAIL",
                gross_bad + net_bad,
                "多头用 exit/entry-1，空头用 1-exit/entry；净收益扣 roundtrip cost",
            )
        )

        if {"stop_price_barrier", "take_profit_price_barrier"}.issubset(trades.columns):
            stop = pd.to_numeric(trades["stop_price_barrier"], errors="coerce")
            take = pd.to_numeric(trades["take_profit_price_barrier"], errors="coerce")
            is_short = trades["side"].astype(str).str.lower() == "short"
            geom_bad_mask = np.where(is_short, (take > trades["entry_price"]) | (stop < trades["entry_price"]), (stop > trades["entry_price"]) | (take < trades["entry_price"]))
            geom_bad = int(pd.Series(geom_bad_mask).fillna(True).sum())
            path_examples, path_bad = _validate_barrier_path_sample(trades, features_lookup, config.barrier_sample_per_file, config.random_seed)
            barrier_rows.append(
                {
                    "file_id": file_id,
                    "path": str(rel_path),
                    "status": "PASS" if geom_bad + path_bad == 0 else "FAIL",
                    "rows": len(trades),
                    "geometry_mismatch_count": geom_bad,
                    "path_sample_size": min(len(trades), config.barrier_sample_per_file),
                    "path_sample_mismatch_count": path_bad,
                    "path_example_count": len(path_examples),
                }
            )
            if not path_examples.empty:
                example_path = config.output_dir / f"triple_barrier_path_examples_{file_id}.csv"
                write_csv(path_examples, example_path)
            findings.append(
                _finding(
                    "triple_barrier",
                    f"{file_id}_barrier_geometry_and_path",
                    "BLOCKER",
                    "PASS" if geom_bad + path_bad == 0 else "FAIL",
                    geom_bad + path_bad,
                    "Triple Barrier 几何必须符合多空方向，抽样路径必须与退出原因一致",
                    f"path_sample={min(len(trades), config.barrier_sample_per_file)}",
                )
            )
        else:
            barrier_rows.append({"file_id": file_id, "path": str(rel_path), "status": "PASS", "rows": len(trades), "geometry_mismatch_count": 0, "path_sample_size": 0, "path_sample_mismatch_count": 0, "path_example_count": 0})

        # Triple Barrier 可以在入场所在 5 分钟 Bar 内触发止损/止盈，因此同一时间退出不是倒挂；
        # 真正的时间错误是 exit_datetime 早于 entry_datetime。
        order_bad = int((trades["entry_datetime"] > trades["exit_datetime"]).sum())
        same_bar_exit = int((trades["entry_datetime"] == trades["exit_datetime"]).sum())
        train_cross = int(((trades["split"] == "train") & (trades["exit_datetime"] > config.train_end)).sum()) if "split" in trades.columns else 0
        test_cross = int(((trades["split"] == "test") & (trades["entry_datetime"] < config.test_start)).sum()) if "split" in trades.columns else 0
        boundary_rows.append(
            {
                "file_id": file_id,
                "path": str(rel_path),
                "status": "PASS" if order_bad + train_cross + test_cross == 0 else "FAIL",
                "rows": len(trades),
                "entry_exit_order_bad": order_bad,
                "same_bar_exit_count": same_bar_exit,
                "train_exit_after_train_end": train_cross,
                "test_entry_before_test_start": test_cross,
            }
        )
        findings.append(
            _finding(
                "boundary_crossing",
                f"{file_id}_entry_exit_and_split_boundary",
                "BLOCKER",
                "PASS" if order_bad + train_cross + test_cross == 0 else "FAIL",
                order_bad + train_cross + test_cross,
                "entry/exit 时间顺序必须正确，训练交易不得穿越测试边界",
            )
        )

        overlap_bad = 0
        overlap_groups = 0
        key_cols = _trade_key_columns(trades)
        for keys, part in trades.sort_values(key_cols + ["entry_datetime"]).groupby(key_cols, sort=False):
            g = part.sort_values("entry_datetime")
            bad_mask = g["entry_datetime"].shift(-1) < g["exit_datetime"]
            bad = int(bad_mask.sum())
            if bad:
                overlap_groups += 1
                overlap_bad += bad
        overlap_rows.append(
            {
                "file_id": file_id,
                "path": str(rel_path),
                "status": "PASS" if overlap_bad == 0 else "FAIL",
                "rows": len(trades),
                "overlap_count": overlap_bad,
                "overlap_group_count": overlap_groups,
                "group_keys": ",".join(key_cols),
            }
        )
        findings.append(
            _finding(
                "trade_overlap",
                f"{file_id}_no_same_contract_side_overlap",
                "BLOCKER",
                "PASS" if overlap_bad == 0 else "FAIL",
                overlap_bad,
                "同一策略、同一合约、同一方向不允许重叠持仓",
                ",".join(key_cols),
            )
        )

    return pd.DataFrame(repricing_rows), pd.DataFrame(barrier_rows), pd.DataFrame(overlap_rows), pd.DataFrame(boundary_rows), findings


def _summary_from_findings(findings: pd.DataFrame) -> pd.DataFrame:
    if findings.empty:
        return pd.DataFrame()
    grouped = findings.groupby("category", dropna=False)
    rows: List[Dict[str, Any]] = []
    for category, part in grouped:
        blocker_fail = int(((part["severity"] == "BLOCKER") & (part["status"] == "FAIL")).sum())
        warning_count = int((part["status"] == "WARN").sum())
        rows.append(
            {
                "category": category,
                "checks": len(part),
                "blocker_fail_count": blocker_fail,
                "warning_count": warning_count,
                "pass_count": int((part["status"] == "PASS").sum()),
                "overall_status": "P0_BLOCKED" if blocker_fail else "PASS",
            }
        )
    total_blockers = int(((findings["severity"] == "BLOCKER") & (findings["status"] == "FAIL")).sum())
    rows.append(
        {
            "category": "__OVERALL__",
            "checks": len(findings),
            "blocker_fail_count": total_blockers,
            "warning_count": int((findings["status"] == "WARN").sum()),
            "pass_count": int((findings["status"] == "PASS").sum()),
            "overall_status": "P0_BLOCKED" if total_blockers else "P0_PASS",
        }
    )
    return pd.DataFrame(rows)


def _zh_status(status: str) -> str:
    return {"PASS": "通过", "FAIL": "失败", "WARN": "警告", "P0_PASS": "可以进入阶段三", "P0_BLOCKED": "不建议进入阶段三"}.get(status, status)


def _render_report(config: BlockerAuditConfig, findings: pd.DataFrame, summary: pd.DataFrame) -> str:
    overall = summary.loc[summary["category"] == "__OVERALL__", "overall_status"].iloc[0] if not summary.empty else "P0_BLOCKED"
    blocker_failures = findings[(findings["severity"] == "BLOCKER") & (findings["status"] == "FAIL")].copy()
    warnings = findings[findings["status"] == "WARN"].copy()

    lines = [
        "# 阶段二 P0 阻断审计报告",
        "",
        f"- 审计目录：`{config.stage2_dir}`",
        f"- 输出目录：`{config.output_dir}`",
        f"- 训练结束：`{config.train_end}`",
        f"- 测试开始：`{config.test_start}`",
        f"- 总判定：**{_zh_status(overall)}**",
        "",
        "## 总览",
        "",
        "| 类别 | 检查数 | 阻断失败 | 警告 | 通过 | 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        if row.category == "__OVERALL__":
            continue
        lines.append(
            f"| {row.category} | {row.checks} | {row.blocker_fail_count} | {row.warning_count} | {row.pass_count} | {_zh_status(row.overall_status)} |"
        )
    overall_row = summary[summary["category"] == "__OVERALL__"].iloc[0]
    lines.extend(
        [
            f"| **总计** | **{overall_row['checks']}** | **{overall_row['blocker_fail_count']}** | **{overall_row['warning_count']}** | **{overall_row['pass_count']}** | **{_zh_status(overall_row['overall_status'])}** |",
            "",
            "## 阻断项",
            "",
        ]
    )
    if blocker_failures.empty:
        lines.append("没有发现 P0 阻断项。")
    else:
        lines.extend(["| 类别 | 检查项 | 影响行数 | 说明 | 细节 |", "|---|---|---:|---|---|"])
        for row in blocker_failures.sort_values(["category", "check_id"]).itertuples(index=False):
            lines.append(f"| {row.category} | {row.check_id} | {row.affected_rows} | {row.message} | {row.detail} |")

    lines.extend(["", "## 警告项", ""])
    if warnings.empty:
        lines.append("没有警告项。")
    else:
        lines.extend(["| 类别 | 检查项 | 影响行数 | 说明 |", "|---|---|---:|---|"])
        for row in warnings.sort_values(["category", "check_id"]).itertuples(index=False):
            lines.append(f"| {row.category} | {row.check_id} | {row.affected_rows} | {row.message} |")

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `stage2_blocker_findings.csv`",
            "- `stage2_blocker_summary.csv`",
            "- `label_alignment_audit.csv`",
            "- `train_test_isolation_audit.csv`",
            "- `trade_repricing_audit.csv`",
            "- `triple_barrier_audit.csv`",
            "- `trade_overlap_audit.csv`",
            "- `boundary_crossing_audit.csv`",
            "- `data_integrity_audit.csv`",
            "- `extreme_bar_audit.csv`",
            "- `feature_nan_inf_audit.csv`",
            "",
            "## 结论",
            "",
        ]
    )
    if overall == "P0_PASS":
        lines.append("P0 红线检查通过，可以进入阶段三 walk-forward 的设计与实现。")
    else:
        lines.append("P0 红线检查未通过，不建议进入阶段三。应先修复上表中的阻断项，再重新运行本审计。")
    return "\n".join(lines) + "\n"


def run(config: BlockerAuditConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir = config.stage2_dir

    features = _safe_read(stage2_dir / "stage2_features.csv", ["datetime"])
    labels = _safe_read(stage2_dir / "stage2_labels.csv", ["datetime"])
    features = features.sort_values(["contract", "datetime"]).reset_index(drop=True)
    labels = labels.sort_values(["contract", "datetime"]).reset_index(drop=True)

    all_findings: List[Dict[str, Any]] = []

    data_integrity, feature_nan_inf, extreme_bars, findings = audit_data_integrity(features, config)
    all_findings.extend(findings)
    label_alignment, findings = audit_label_alignment(features, labels, config)
    all_findings.extend(findings)
    train_test_isolation, findings = audit_train_test_isolation(config)
    all_findings.extend(findings)
    repricing, triple_barrier, overlap, boundary, findings = audit_trade_files(features, config)
    all_findings.extend(findings)

    findings_df = pd.DataFrame(all_findings)
    summary = _summary_from_findings(findings_df)

    write_csv(label_alignment, config.output_dir / "label_alignment_audit.csv")
    write_csv(train_test_isolation, config.output_dir / "train_test_isolation_audit.csv")
    write_csv(repricing, config.output_dir / "trade_repricing_audit.csv")
    write_csv(triple_barrier, config.output_dir / "triple_barrier_audit.csv")
    write_csv(overlap, config.output_dir / "trade_overlap_audit.csv")
    write_csv(boundary, config.output_dir / "boundary_crossing_audit.csv")
    write_csv(data_integrity, config.output_dir / "data_integrity_audit.csv")
    write_csv(extreme_bars, config.output_dir / "extreme_bar_audit.csv")
    write_csv(feature_nan_inf, config.output_dir / "feature_nan_inf_audit.csv")
    write_csv(findings_df, config.output_dir / "stage2_blocker_findings.csv")
    write_csv(summary, config.output_dir / "stage2_blocker_summary.csv")
    report = _render_report(config, findings_df, summary)
    (config.output_dir / "stage2_blocker_audit_report_zh.md").write_text(report, encoding="utf-8")

    blocker_fail_count = int(summary.loc[summary["category"] == "__OVERALL__", "blocker_fail_count"].iloc[0]) if not summary.empty else 1
    if config.fail_on_blocker and blocker_fail_count:
        raise SystemExit(2)
    return {
        "output_dir": config.output_dir,
        "features_rows": len(features),
        "labels_rows": len(labels),
        "findings": len(findings_df),
        "blocker_fail_count": blocker_fail_count,
        "overall_status": "P0_BLOCKED" if blocker_fail_count else "P0_PASS",
    }
