"""阶段二审计。

审计目标是把“研究结果不好”和“代码/数据有硬伤”分开。策略亏损只
记录为 OBSERVATION；未来函数、标签错位、跨合约污染、策略重叠等才是 BLOCKER。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import BASE_COLUMNS, LABEL_COLUMNS, Stage2Config
from .features import FEATURE_COLUMNS
from .labels import build_labels
from .strategy import check_no_overlap

FLOAT_TOL = 1e-10
FORBIDDEN_FEATURE_TOKENS = ["future_", "mfe", "mae", "entry_", "exit_", "trade_return"]


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


def _md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    """不依赖 tabulate 的简易 Markdown 表格。"""

    if df.empty:
        return "无数据。"
    view = df.head(max_rows).copy()
    columns = list(view.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _compare_float(left: pd.Series, right: pd.Series, tolerance: float = FLOAT_TOL) -> int:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_missing = left_num.isna() & right_num.isna()
    both_present = left_num.notna() & right_num.notna()
    ok = both_missing | (both_present & ((left_num - right_num).abs() <= tolerance))
    return int((~ok).sum())


def _audit_label_alignment(labels: pd.DataFrame, config: Stage2Config) -> List[Dict[str, Any]]:
    recomputed = build_labels(labels[BASE_COLUMNS].copy(), config)
    rows: List[Dict[str, Any]] = []
    for label in LABEL_COLUMNS:
        mismatch = _compare_float(labels[label], recomputed[label])
        rows.append(
            finding(
                "BLOCKER",
                "label_alignment",
                label,
                "PASS" if mismatch == 0 else "FAIL",
                f"{label} 可按单合约重新计算",
                mismatch,
            )
        )

        horizon = 10
        if label.startswith("future_return_"):
            horizon = int(label.rsplit("_", 1)[-1])
        elif label == "entry_open_next":
            horizon = 1
        elif label in {"exit_close_after_5", "trade_return_5_gross", "trade_return_5_net"}:
            horizon = config.holding_bars
        tail_bad = 0
        for _, group in labels.groupby("contract"):
            tail = group.tail(horizon)
            tail_bad += int(tail[label].notna().sum())
        rows.append(
            finding(
                "BLOCKER",
                "label_alignment",
                f"{label}_tail_null",
                "PASS" if tail_bad == 0 else "FAIL",
                f"{label} 合约尾部应为空，防止跨合约泄漏",
                tail_bad,
            )
        )
    return rows


def _selected_corr(selected_features: pd.DataFrame, corr_pairs: pd.DataFrame) -> float:
    if selected_features.empty or corr_pairs.empty:
        return 0.0
    selected = set(selected_features["feature"])
    part = corr_pairs[corr_pairs["feature_a"].isin(selected) & corr_pairs["feature_b"].isin(selected)]
    if part.empty:
        return 0.0
    return float(part["spearman_corr"].abs().max())


def run_stage2_audit(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    selected_features: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    config: Stage2Config,
) -> Tuple[pd.DataFrame, str]:
    rows: List[Dict[str, Any]] = []

    duplicate_features = int(features.duplicated(["contract", "datetime"]).sum())
    duplicate_labels = int(labels.duplicated(["contract", "datetime"]).sum())
    rows.append(finding("BLOCKER", "data_integrity", "duplicate_features", "PASS" if duplicate_features == 0 else "FAIL", "features 不应有重复 contract+datetime", duplicate_features))
    rows.append(finding("BLOCKER", "data_integrity", "duplicate_labels", "PASS" if duplicate_labels == 0 else "FAIL", "labels 不应有重复 contract+datetime", duplicate_labels))
    rows.append(finding("BLOCKER", "data_integrity", "row_count_match", "PASS" if len(features) == len(labels) else "FAIL", "features 与 labels 行数一致", abs(len(features) - len(labels)), f"features={len(features)}, labels={len(labels)}"))

    ohlc_bad = int(((features["high"] < features[["open", "close", "low"]].max(axis=1)) | (features["low"] > features[["open", "close", "high"]].min(axis=1))).sum())
    delta_bad = int((features["delta"].abs() > features["volume"]).sum())
    poc_bad = int(((features["poc"] < features["low"]) | (features["poc"] > features["high"])).sum())
    rows.append(finding("BLOCKER", "data_integrity", "ohlc_structure", "PASS" if ohlc_bad == 0 else "FAIL", "OHLC 结构合法", ohlc_bad))
    rows.append(finding("BLOCKER", "data_integrity", "delta_within_volume", "PASS" if delta_bad == 0 else "FAIL", "abs(delta) <= volume", delta_bad))
    rows.append(finding("BLOCKER", "data_integrity", "poc_in_bar", "PASS" if poc_bad == 0 else "FAIL", "poc 位于 [low, high]", poc_bad))

    rows.extend(_audit_label_alignment(labels, config))

    forbidden = [feature for feature in FEATURE_COLUMNS if any(token in feature for token in FORBIDDEN_FEATURE_TOKENS)]
    rows.append(finding("BLOCKER", "feature_leakage", "forbidden_feature_names", "PASS" if not forbidden else "FAIL", "特征名不应包含未来标签或交易结果字段", len(forbidden), ",".join(forbidden)))

    selected_count = len(selected_features)
    selected_groups = int(selected_features["group"].nunique()) if not selected_features.empty else 0
    max_group_share = float(selected_features["group"].value_counts(normalize=True).max()) if not selected_features.empty else np.nan
    max_corr = _selected_corr(selected_features, corr_pairs)
    rows.append(finding("BLOCKER", "feature_selection", "selected_count", "PASS" if 0 < selected_count <= config.max_selected_features else "FAIL", "入选特征数量在限制内", selected_count, f"limit={config.max_selected_features}"))
    rows.append(finding("BLOCKER", "feature_selection", "selected_corr_threshold", "PASS" if max_corr <= config.corr_threshold else "FAIL", "入选特征训练集最大相关性不超过阈值", int(max_corr > config.corr_threshold), f"max_corr={max_corr:.6f}, threshold={config.corr_threshold}"))
    rows.append(finding("BLOCKER", "feature_selection", "selected_group_count", "PASS" if selected_groups >= config.min_selected_groups else "FAIL", "入选特征覆盖足够多维度", selected_groups, f"min_groups={config.min_selected_groups}"))
    rows.append(finding("BLOCKER", "feature_selection", "selected_group_share", "PASS" if pd.notna(max_group_share) and max_group_share <= config.max_group_share else "FAIL", "单一特征组占比不超过上限", int(pd.isna(max_group_share) or max_group_share > config.max_group_share), f"max_group_share={max_group_share}, limit={config.max_group_share}"))

    overlap = check_no_overlap(trades)
    overlap_count = int(overlap["overlap_count"].sum()) if not overlap.empty else 0
    rows.append(finding("BLOCKER", "strategy", "trade_overlap", "PASS" if overlap_count == 0 else "FAIL", "同合约交易不重叠", overlap_count))
    if trades.empty:
        rows.append(finding("WARNING", "strategy", "no_trades", "WARN", "阶段二策略没有生成交易", 0))
    else:
        boundary_bad = int(((trades["split"] == "train") & (pd.to_datetime(trades["exit_datetime"]) >= config.test_start)).sum())
        rows.append(finding("BLOCKER", "strategy", "train_test_boundary", "PASS" if boundary_bad == 0 else "FAIL", "训练集交易不穿越测试边界", boundary_bad))
        long_bad = int(((trades["side"] == "long") & (trades["signal_score"] < trades["long_threshold"])).sum())
        short_bad = int(((trades["side"] == "short") & (trades["signal_score"] > trades["short_threshold"])).sum())
        rows.append(finding("BLOCKER", "strategy", "threshold_direction", "PASS" if long_bad + short_bad == 0 else "FAIL", "交易方向符合训练集阈值", long_bad + short_bad))

    test_row = strategy_summary[strategy_summary["split"] == "test"]
    if not test_row.empty:
        avg_test = test_row.iloc[0]["avg_net_return"]
        if pd.notna(avg_test) and avg_test < 0:
            rows.append(finding("OBSERVATION", "strategy", "negative_test_return", "INFO", "测试集平均净收益为负，属于研究反馈", int(test_row.iloc[0]["trades"]), f"avg_net_return={avg_test}"))

    findings = pd.DataFrame(rows)
    blockers = findings[(findings["severity"] == "BLOCKER") & (findings["status"] == "FAIL")]
    warnings = findings[(findings["severity"] == "WARNING") & (findings["status"] != "PASS")]
    observations = findings[findings["severity"] == "OBSERVATION"]
    conclusion = "未发现 BLOCKER" if blockers.empty else "发现 BLOCKER，需要先修复"
    report = "\n".join(
        [
            "# 阶段二审计报告",
            "",
            f"- 结论：{conclusion}",
            f"- BLOCKER：`{len(blockers)}`",
            f"- WARNING：`{len(warnings)}`",
            f"- OBSERVATION：`{len(observations)}`",
            f"- 训练集截止：`{config.train_end}`",
            f"- 测试集开始：`{config.test_start}`",
            "",
            "## 关键回答",
            "- 未来函数：通过标签复算和合约尾部空值检查。",
            "- 跨合约污染：通过合约尾部标签空值检查。",
            "- 特征相关性：按训练集 Spearman 相关性检查。",
            "- 特征组覆盖：检查入选特征组数量和单组占比。",
            "- 策略逻辑：检查阈值方向、同合约不重叠和训练/测试边界。",
            "",
            "## 审计明细",
            _md_table(findings),
            "",
        ]
    )
    return findings, report
