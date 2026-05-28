# 第一阶段全面审计报告

## 总结

- 结论：未发现 BLOCKER，第一阶段未发现明显未来函数或跨合约泄漏。
- BLOCKER：`0`
- WARNING：`6`
- OBSERVATION：`1`

## 审计摘要

| severity    | module             |   count |
|:------------|:-------------------|--------:|
| OBSERVATION | ALL                |       1 |
| WARNING     | ALL                |       6 |
| OBSERVATION | strategy           |       1 |
| WARNING     | extreme_rows       |       4 |
| WARNING     | feature_quality    |       1 |
| WARNING     | report_consistency |       1 |

## 主要发现

| severity    | module             | check_id                       | status   | message                                                              |   row_count | detail                                                  |
|:------------|:-------------------|:-------------------------------|:---------|:---------------------------------------------------------------------|------------:|:--------------------------------------------------------|
| WARNING     | feature_quality    | selected_feature_concentration | WARN     | 选中特征过度集中在单一特征组                                         |           3 | {'price_structure': 3, 'trend_volatility': 1, 'poc': 1} |
| OBSERVATION | strategy           | negative_test_return           | INFO     | 测试集平均净收益为负，属于MVP反馈而非阻断                            |        3518 | avg_net_return=-0.0002594191855853                      |
| WARNING     | extreme_rows       | range_ticks                    | WARN     | range_ticks 出现极端值                                               |          55 | > 100                                                   |
| WARNING     | extreme_rows       | poc_distance_ticks             | WARN     | poc_distance_ticks 出现极端值                                        |           6 | abs > 100                                               |
| WARNING     | extreme_rows       | poc_shift_ticks                | WARN     | poc_shift_ticks 出现极端值                                           |          58 | abs > 100                                               |
| WARNING     | extreme_rows       | open_interest_change           | WARN     | open_interest_change 出现极端值                                      |          73 | abs > 99.9% quantile (2326.6840)                        |
| WARNING     | report_consistency | feature_quintiles_full_sample  | WARN     | feature_quintiles.csv 当前是全样本描述统计，不应作为训练期选特征依据 |           0 |                                                         |

## 解释

- BLOCKER 代表必须修复的问题，例如未来函数、标签错位、跨合约泄漏、策略重叠持仓。
- WARNING 代表第一阶段需要重点关注的问题，例如极端值、特征高相关、报告解释风险。
- OBSERVATION 代表研究反馈，例如策略亏损或IC偏弱，不等于代码错误。

## 下一步建议

- 优先查看 `extreme_rows.csv`，确认极端Bar是否来自换月、流动性枯竭或异常行情。
- 查看 `feature_correlation.csv`，决定第二阶段是否合并高度相关特征。
- 将 `feature_quintiles.csv` 明确标注为全样本描述统计；若用于选特征，应改成训练集统计。
- 第二阶段引入市场状态分层、walk-forward、真实成本和角色化Alpha组合。
