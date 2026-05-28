# 阶段二三项核心优化说明

本次优化只修改代码与文档，不提交任何新生成的 CSV。若需要重新生成研究结果，应在本地运行流水线，并让 CSV 继续受 `.gitignore` 管理。

## 1. 三重屏障跳空止损成交价

原逻辑在触发止损时直接使用 `stop_price` 作为成交价。这会高估跳空行情中的止损执行质量：

- 多头隔夜或 Bar 内跳空低开并跌破止损时，真实成交价应不高于止损价。
- 空头跳空高开并穿越止损时，真实成交价应不低于止损价。

新逻辑：

- 多头止损：`exit_price = min(open_at_stop_bar, stop_price)`
- 空头止损：`exit_price = max(open_at_stop_bar, stop_price)`
- 止盈仍按 `take_profit_price` 结算。
- 同一根 Bar 同时触及止损和止盈时，继续采用保守的止损优先规则。

影响范围：

- `trend_pullback_sweep.py` 的多空参数扫描。
- `trend_pullback.py` 的 long-only 趋势回调专项实验。
- `blocker_audit.py` 的 Triple Barrier 路径复算口径。

## 2. mRMR 去冗余打分

原 mRMR 打分为乘法惩罚：

```text
score = abs_ic * (1 - 0.5 * redundancy + diversity_bonus)
```

该公式对高 IC、高相关的候选特征惩罚偏弱。新公式改为尺度归一的减法惩罚：

```text
score = abs_ic - 0.8 * redundancy * ic_scale + 0.15 * group_diversity * ic_scale
```

其中：

- `ic_scale` 使用训练池 `abs_spearman_ic_5` 的中位数。
- 若中位数不可用，则回退到均值。
- 若均值仍不可用，则使用 `1e-6`。

保留项：

- `corr_threshold=0.85` 硬约束仍然生效。
- 组别上限仍然生效。
- 每个特征组先给一个代表名额的逻辑仍然保留。

## 3. 离散/布尔特征标准化

原逻辑对所有特征统一做 winsorize 和 robust zscore。对 `volume_burst_flag` 这类 0/1 稀疏特征，如果训练期全为 0，测试期出现的 1 可能被 `scale=0` 分支抹成 NaN。

新逻辑给特征增加 `feature_type`：

- `continuous`：继续使用训练集 winsorize + MAD/std 标准化。
- `binary_flag`：保留原始 0/1。
- `signed_flag`：保留原始 -1/0/1。
- `cyclical`：保留原始 sin/cos。

`feature_transform_params.csv` 中这些特征会记录为：

```text
scale_type = passthrough
```

这样既保留物理含义，也方便审计。

## 验证建议

推荐先使用 `/private/tmp` 做小样本验证：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/blueblue_pycache \
/opt/homebrew/Caskroom/miniforge/base/envs/tianqin/bin/python -m py_compile \
stage2_pipeline.py stage2_trend_pullback_gain.py stage2_trend_pullback_sweep.py blueblue_research/*.py
```

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/tianqin/bin/python stage2_pipeline.py \
  --contracts SHFE.cu2604 \
  --output_dir /private/tmp/blueblue_stage2_opt_smoke
```

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/tianqin/bin/python stage2_trend_pullback_sweep.py \
  --contracts SHFE.cu2604 \
  --sides long,short \
  --report_tag opt_smoke \
  --output_dir /private/tmp/blueblue_sweep_opt_smoke
```

提交前必须确认：

```bash
git status --short
```

本轮提交范围只应包含 `.py` 和 `.md`，不要提交任何 `.csv`。
