# 阶段二审计报告

- 结论：未发现 BLOCKER
- BLOCKER：`0`
- WARNING：`0`
- OBSERVATION：`1`
- 训练集截止：`2024-03-23 00:55:00`
- 测试集开始：`2024-03-25 09:00:00`

## 关键回答
- 未来函数：通过标签复算和合约尾部空值检查。
- 跨合约污染：通过合约尾部标签空值检查。
- 特征相关性：按训练集 Spearman 相关性检查。
- 特征组覆盖：检查入选特征组数量和单组占比。
- 策略逻辑：检查阈值方向、同合约不重叠和训练/测试边界。

## 审计明细
| severity | module | check_id | status | message | row_count | detail |
| --- | --- | --- | --- | --- | --- | --- |
| BLOCKER | data_integrity | duplicate_features | PASS | features 不应有重复 contract+datetime | 0 |  |
| BLOCKER | data_integrity | duplicate_labels | PASS | labels 不应有重复 contract+datetime | 0 |  |
| BLOCKER | data_integrity | row_count_match | PASS | features 与 labels 行数一致 | 0 | features=259076, labels=259076 |
| BLOCKER | data_integrity | ohlc_structure | PASS | OHLC 结构合法 | 0 |  |
| BLOCKER | data_integrity | delta_within_volume | PASS | abs(delta) <= volume | 0 |  |
| BLOCKER | data_integrity | poc_in_bar | PASS | poc 位于 [low, high] | 0 |  |
| BLOCKER | label_alignment | future_return_1 | PASS | future_return_1 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | future_return_1_tail_null | PASS | future_return_1 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | future_return_3 | PASS | future_return_3 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | future_return_3_tail_null | PASS | future_return_3 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | future_return_5 | PASS | future_return_5 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | future_return_5_tail_null | PASS | future_return_5 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | future_return_10 | PASS | future_return_10 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | future_return_10_tail_null | PASS | future_return_10 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | mfe_10_long | PASS | mfe_10_long 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | mfe_10_long_tail_null | PASS | mfe_10_long 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | mae_10_long | PASS | mae_10_long 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | mae_10_long_tail_null | PASS | mae_10_long 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | entry_open_next | PASS | entry_open_next 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | entry_open_next_tail_null | PASS | entry_open_next 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | exit_close_after_5 | PASS | exit_close_after_5 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | exit_close_after_5_tail_null | PASS | exit_close_after_5 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | trade_return_5_gross | PASS | trade_return_5_gross 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | trade_return_5_gross_tail_null | PASS | trade_return_5_gross 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | label_alignment | trade_return_5_net | PASS | trade_return_5_net 可按单合约重新计算 | 0 |  |
| BLOCKER | label_alignment | trade_return_5_net_tail_null | PASS | trade_return_5_net 合约尾部应为空，防止跨合约泄漏 | 0 |  |
| BLOCKER | feature_leakage | forbidden_feature_names | PASS | 特征名不应包含未来标签或交易结果字段 | 0 |  |
| BLOCKER | feature_selection | selected_count | PASS | 入选特征数量在限制内 | 18 | limit=18 |
| BLOCKER | feature_selection | selected_corr_threshold | PASS | 入选特征训练集最大相关性不超过阈值 | 0 | max_corr=0.752445, threshold=0.85 |
| BLOCKER | feature_selection | selected_group_count | PASS | 入选特征覆盖足够多维度 | 7 | min_groups=5 |
| BLOCKER | feature_selection | selected_group_share | PASS | 单一特征组占比不超过上限 | 0 | max_group_share=0.2777777777777778, limit=0.45 |
| BLOCKER | strategy | trade_overlap | PASS | 同合约交易不重叠 | 0 |  |
| BLOCKER | strategy | train_test_boundary | PASS | 训练集交易不穿越测试边界 | 0 |  |
| BLOCKER | strategy | threshold_direction | PASS | 交易方向符合训练集阈值 | 0 |  |
| OBSERVATION | strategy | negative_test_return | INFO | 测试集平均净收益为负，属于研究反馈 | 7486 | avg_net_return=-0.00024314821556704877 |
