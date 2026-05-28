# 阶段二 P0 阻断审计报告

- 审计目录：`/Users/wangrendong/Projects/BlueBlue/stage2_outputs/SHFE.cu`
- 输出目录：`/Users/wangrendong/Projects/BlueBlue/stage2_outputs/SHFE.cu/blocker_audit`
- 训练结束：`2024-03-23 00:55:00`
- 测试开始：`2024-03-25 09:00:00`
- 总判定：**不建议进入阶段三**

## 总览

| 类别 | 检查数 | 阻断失败 | 警告 | 通过 | 状态 |
|---|---:|---:|---:|---:|---|
| boundary_crossing | 5 | 0 | 0 | 5 | 通过 |
| data_integrity | 12 | 0 | 1 | 11 | 通过 |
| label_alignment | 5 | 0 | 0 | 5 | 通过 |
| trade_overlap | 5 | 0 | 0 | 5 | 通过 |
| trade_repricing | 5 | 2 | 0 | 3 | 不建议进入阶段三 |
| train_test_isolation | 11 | 0 | 1 | 10 | 通过 |
| triple_barrier | 3 | 0 | 0 | 3 | 通过 |
| **总计** | **46** | **2** | **2** | **42** | **不建议进入阶段三** |

## 阻断项

| 类别 | 检查项 | 影响行数 | 说明 | 细节 |
|---|---|---:|---|---|
| trade_repricing | gain_curve_linear_futures_return_formula | 159546 | 多头用 exit/entry-1，空头用 1-exit/entry；净收益扣 roundtrip cost |  |
| trade_repricing | stage2_strategy_linear_futures_return_formula | 31716 | 多头用 exit/entry-1，空头用 1-exit/entry；净收益扣 roundtrip cost |  |

## 警告项

| 类别 | 检查项 | 影响行数 | 说明 |
|---|---|---:|---|
| data_integrity | extreme_bar_locator | 1265 | 极端 Bar 已定位；默认不作为阻断项 |
| train_test_isolation | long_short_summary_test_observed_source | 1 | 汇总报告出现测试集观察最优时需警惕参数建议泄漏；本项为警告，不直接阻断 |

## 输出文件

- `stage2_blocker_findings.csv`
- `stage2_blocker_summary.csv`
- `label_alignment_audit.csv`
- `train_test_isolation_audit.csv`
- `trade_repricing_audit.csv`
- `triple_barrier_audit.csv`
- `trade_overlap_audit.csv`
- `boundary_crossing_audit.csv`
- `data_integrity_audit.csv`
- `extreme_bar_audit.csv`
- `feature_nan_inf_audit.csv`

## 结论

P0 红线检查未通过，不建议进入阶段三。应先修复上表中的阻断项，再重新运行本审计。
