"""阶段二中文报告输出。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Stage2Config


def _md_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "无数据。"
    view = df.head(max_rows).copy()
    columns = list(view.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_stage2_report_zh(
    config: Stage2Config,
    features: pd.DataFrame,
    feature_summary_train: pd.DataFrame,
    feature_summary_test: pd.DataFrame,
    selected_features: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    audit_findings: pd.DataFrame,
    output_path: Path,
) -> None:
    top_train_cols = ["feature", "group", "coverage", "spearman_ic_5", "quintile_spread_mean_return"]
    selected_cols = ["selection_order", "feature", "group", "spearman_ic_5", "mrmr_score", "max_abs_corr_to_selected"]
    failed = audit_findings[(audit_findings["severity"] == "BLOCKER") & (audit_findings["status"] == "FAIL")]
    warnings = audit_findings[(audit_findings["severity"] == "WARNING") & (audit_findings["status"] != "PASS")]

    group_counts = selected_features["group"].value_counts().reset_index() if not selected_features.empty else pd.DataFrame()
    if not group_counts.empty:
        group_counts.columns = ["group", "selected_count"]

    lines = [
        "# 阶段二研究报告：低冗余多维特征框架",
        "",
        "## 1. 数据范围与切分",
        f"- 输入目录：`{config.input_dir}`",
        f"- 输出目录：`{config.output_dir}`",
        f"- 数据起点：`{config.start_date}`，默认读取全部可用历史。",
        f"- 实际时间范围：`{features['datetime'].min()}` 至 `{features['datetime'].max()}`",
        f"- 合约数量：`{features['contract'].nunique()}`",
        f"- 样本行数：`{len(features)}`",
        f"- 训练集截止：`{config.train_end}`",
        f"- 测试集开始：`{config.test_start}`",
        "",
        "## 2. 阶段二做了什么",
        "- 把阶段一单脚本拆成配置、读取、特征、标签、检验、选择、策略、审计、报告模块。",
        "- 特征从 17 个扩展到多组候选特征，并用训练集参数做 winsorize 与稳健标准化。",
        "- 特征选择采用低冗余、多维度约束：训练集相关性阈值、组别覆盖、单组占比限制。",
        "- 策略仍使用单次训练/测试切分，没有使用 walk-forward。",
        "",
        "## 3. 训练集 Top 特征",
        _md_table(feature_summary_train[top_train_cols], max_rows=15),
        "",
        "## 4. 测试集对应表现",
        _md_table(feature_summary_test[top_train_cols], max_rows=15),
        "",
        "## 5. 入选特征",
        _md_table(selected_features[selected_cols], max_rows=25) if not selected_features.empty else "没有入选特征。",
        "",
        "## 6. 入选特征组分布",
        _md_table(group_counts, max_rows=10),
        "",
        "## 7. 阶段二策略表现",
        _md_table(strategy_summary, max_rows=10),
        "",
        "## 8. 成本敏感性",
        _md_table(cost_sensitivity, max_rows=20),
        "",
        "## 9. 审计结论",
        f"- BLOCKER 数量：`{len(failed)}`",
        f"- WARNING 数量：`{len(warnings)}`",
        "- 如果策略仍然亏损，结论是当前特征和组合规则仍不足以覆盖交易成本，而不是数据管线失败。",
        "",
        "## 10. 留给阶段三",
        "- 引入 walk-forward，观察特征选择是否随时间稳定。",
        "- 加入真实手续费、滑点、盘口可成交性和成交延迟。",
        "- 从单品种事件研究升级到跨品种组合和资金管理。",
        "- 对阶段二入选特征做状态分层，例如趋势/震荡、高低波动、高低流动性。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
