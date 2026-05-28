#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段二入口：低冗余多维特征、单次切分研究框架。

默认读取全部可用 5 分钟订单流 Bar，并按全样本时间跨度 80/20 做单次
训练/测试切分。阶段三再引入 walk-forward。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from blueblue_research.audit import run_stage2_audit
from blueblue_research.config import BASE_COLUMNS, LABEL_COLUMNS, parse_args, resolve_time_split
from blueblue_research.feature_tests import (
    build_feature_correlation,
    build_feature_quality,
    build_monthly_ic,
    summarize_features,
)
from blueblue_research.features import FEATURE_COLUMNS, build_raw_features, feature_catalog, fit_transform_features
from blueblue_research.io import load_orderflow_bars, write_csv
from blueblue_research.labels import build_labels
from blueblue_research.reporting import write_stage2_report_zh
from blueblue_research.selection import select_features
from blueblue_research.strategy import build_strategy, check_no_overlap


def run() -> None:
    raw_config = parse_args()
    bars = load_orderflow_bars(raw_config)
    config = resolve_time_split(raw_config, bars)

    raw_features = build_raw_features(bars, config.tick_size)
    features, transform_params = fit_transform_features(raw_features, config)
    labeled = build_labels(features, config)

    train_mask = labeled["datetime"] <= config.train_end
    test_mask = labeled["datetime"] >= config.test_start
    train_df = labeled.loc[train_mask].copy()
    test_df = labeled.loc[test_mask].copy()

    feature_quality = build_feature_quality(labeled)
    feature_summary_train, feature_quintiles_train = summarize_features(train_df, config.label_col, "train")
    feature_summary_test, feature_quintiles_test = summarize_features(test_df, config.label_col, "test")
    monthly_ic = build_monthly_ic(labeled, config.label_col)
    feature_correlation_train = build_feature_correlation(train_df)
    selected_features = select_features(feature_summary_train, feature_correlation_train, config)

    scored, trades, strategy_summary, equity_curve, cost_sensitivity, risk_thresholds = build_strategy(
        labeled, selected_features, transform_params, config
    )
    overlap_check = check_no_overlap(trades)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(labeled[BASE_COLUMNS + FEATURE_COLUMNS], output_dir / "stage2_features.csv")
    write_csv(labeled[BASE_COLUMNS + LABEL_COLUMNS], output_dir / "stage2_labels.csv")
    write_csv(feature_catalog(), output_dir / "feature_catalog.csv")
    write_csv(feature_quality, output_dir / "feature_quality.csv")
    write_csv(transform_params, output_dir / "feature_transform_params.csv")
    write_csv(feature_summary_train, output_dir / "feature_summary_train.csv")
    write_csv(feature_summary_test, output_dir / "feature_summary_test.csv")
    write_csv(feature_quintiles_train, output_dir / "feature_quintiles_train.csv")
    write_csv(feature_quintiles_test, output_dir / "feature_quintiles_test.csv")
    write_csv(monthly_ic, output_dir / "monthly_ic_stage2.csv")
    write_csv(feature_correlation_train, output_dir / "feature_correlation_train.csv")
    write_csv(selected_features, output_dir / "selected_features_stage2.csv")

    score_columns = [col for col in scored.columns if col.startswith("group_score_")]
    write_csv(
        scored[BASE_COLUMNS + ["signal_score", "risk_filter_pass"] + score_columns],
        output_dir / "signal_scores.csv",
    )
    write_csv(trades, output_dir / "trades_stage2.csv")
    write_csv(strategy_summary, output_dir / "strategy_summary_stage2.csv")
    write_csv(equity_curve, output_dir / "equity_curve_stage2.csv")
    write_csv(cost_sensitivity, output_dir / "cost_sensitivity.csv")
    write_csv(pd.DataFrame([risk_thresholds]), output_dir / "risk_thresholds.csv")
    write_csv(overlap_check, output_dir / "trade_overlap_check_stage2.csv")

    audit_findings, audit_report = run_stage2_audit(
        features=labeled[BASE_COLUMNS + FEATURE_COLUMNS],
        labels=labeled[BASE_COLUMNS + LABEL_COLUMNS],
        selected_features=selected_features,
        corr_pairs=feature_correlation_train,
        trades=trades,
        strategy_summary=strategy_summary,
        config=config,
    )
    write_csv(audit_findings, output_dir / "stage2_audit_findings.csv")
    (output_dir / "stage2_audit_report_zh.md").write_text(audit_report, encoding="utf-8")
    write_stage2_report_zh(
        config=config,
        features=labeled,
        feature_summary_train=feature_summary_train,
        feature_summary_test=feature_summary_test,
        selected_features=selected_features,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        audit_findings=audit_findings,
        output_path=output_dir / "stage2_report_zh.md",
    )

    blockers = audit_findings[(audit_findings["severity"] == "BLOCKER") & (audit_findings["status"] == "FAIL")]
    print(f"output_dir={output_dir}")
    print(f"contracts={labeled['contract'].nunique()}, rows={len(labeled)}, features={len(FEATURE_COLUMNS)}")
    print(f"train_end={config.train_end}, test_start={config.test_start}")
    print(f"selected_features={len(selected_features)}, trades={len(trades)}, audit_blockers={len(blockers)}")


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
