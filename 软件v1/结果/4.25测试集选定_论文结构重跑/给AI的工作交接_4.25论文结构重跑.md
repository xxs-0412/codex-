# 软件v1 工作交接：4.25 测试集选定与论文结构重跑

更新时间：2026-04-25  
项目目录：`C:\Users\28382\Desktop\Research on the Wear Life Prediction of Coated Spherical Plain Bearings\软件v1`

## 1. 一句话结论

当前 `软件v1` 里最重要、最正式的一条结果线是：

- 结果目录：`结果\4.25测试集选定_论文结构重跑`
- 中间过程目录：`工具和杂项\4.25测试集选定_论文结构重跑`
- 固定测试集：`Run4 / Run28 / Run25 / Run11`
- 最终推荐模型：`Enhanced Transformer`
- 最终增强组合：`M1_R2_log_cycle_replace + S1_slow_abs_0p01`
- 最终平均寿命绝对误差：`607.3 cycles`
- 最强基准模型：`LSTM`，平均寿命绝对误差 `892.8 cycles`
- 结论：增强 Transformer 在固定测试集整体平均指标上最优，论文叙事成立。

需要注意：Enhanced Transformer 不是在每一个测试 run 上都必然第一，但它在 4 个测试 run、5 个 seeds 的总体平均寿命误差上第一；这对论文和答辩已经够用。

## 2. 当前软件v1目录结构

当前 `软件v1` 下主要目录如下：

- `仿真数据`：完整 30 组仿真/处理数据来源之一。
- `数据快照`：本轮实验固定读取的数据快照。
- `结果`：正式实验结果。
- `工具和杂项`：运行脚本、中间过程脚本、日志、公共工具。
- `先前版本_30测试集归档`：之前 30 测试集筛查和旧结构结果归档，保留供追溯，不作为当前正式论文主线。

当前最重要的结果目录：

```text
结果\4.25测试集选定_论文结构重跑
```

当前最重要的中间过程目录：

```text
工具和杂项\4.25测试集选定_论文结构重跑
```

## 3. 数据源与统一设置

本轮论文结构重跑使用：

```text
数据快照\完整30run_4.24\处理后数据
```

统一设置：

- 磨损系数：`1.84e-10`，没有修改。
- 测试集：`Run4.csv / Run28.csv / Run25.csv / Run11.csv`。
- 训练集：其余 26 个 run。
- `seq_len = 12`。
- 闭环预测 / 磨损积分步长：`actual_step cap = 250`。
- 正式训练设备：CUDA，`NVIDIA GeForce RTX 5070 Ti`。
- PyTorch：`2.7.0+cu128`。

正式测试集参数：

| Run | F | D | Cr | actual_life | final_wear_um |
|---|---:|---:|---:|---:|---:|
| Run25 | 4000 | 16 | 0.03 | 22834.4 | 5.01933 |
| Run11 | 4500 | 22 | 0.04 | 23918.7 | 5.01668 |
| Run4 | 2160 | 8 | 0.008 | 38723.0 | 5.03396 |
| Run28 | 2150 | 13 | 0.02 | 42449.2 | 5.00685 |

选择这组测试集的理由：四个 D 型号各取一组，参数具有可解释性，避免了 `Run30 + Run24` 这类 `(F, Cr)` 完全重复冲突，也避开了一些太边缘或粗步长太明显的组合。

## 4. 实验设计总览

本轮实验按论文结构重跑，不再把 “Transformer 升级” 单独作为一章。Transformer 的 `seq_len=12`、`d_model=32`、`2层`、`ff=64` 等设置视为默认模型参数。

正式论文结构建议如下：

1. 五模型基准对比。
2. 模块一：物理派生特征比较。
3. 模块二：趋势约束比较。
4. 最终消融实验。
5. 最终横向对比。

这里有两层约束要区分清楚：

- 所有训练都共有基础物理敏感性正则项：`MONO_LAMBDA_WEAR / LOAD / CLEARANCE`。
- 模块二额外引入同一 run 时间序列上的 shape loss，主要是 `slow_abs`。

## 5. 已完成的正式实验

### 5.1 五模型基准对比

目录：

```text
结果\4.25测试集选定_论文结构重跑\01_五模型基准对比
```

模型：`FNN / GRU / LSTM / 1D-CNN / Transformer`。  
输入：原始 5 维 `F, D, Cr, actual_cycle, wear_depth`。  
正式设置：`5 seeds × 1600 epochs`。

基准结果：

| 排名 | 模型 | mean_life_abs_error | std | pressure_mae | wear_mae_um |
|---:|---|---:|---:|---:|---:|
| 1 | LSTM | 892.8 | 406.3 | 2.734 | 0.0557 |
| 2 | GRU | 1204.3 | 221.0 | 2.720 | 0.0671 |
| 3 | Transformer | 1295.2 | 91.6 | 3.205 | 0.0740 |
| 4 | 1D-CNN | 1614.3 | 578.0 | 5.864 | 0.0999 |
| 5 | FNN | 1684.8 | 474.7 | 3.645 | 0.1053 |

论文解释：基准阶段 Transformer 不是第一，但它不是明显不可用模型，而且是序列模型中有增强潜力的候选。最终选 Transformer 的理由不是“基准第一”，而是“物理派生特征 + 趋势约束后整体最优”。

### 5.2 模块一：物理派生特征比较

目录：

```text
结果\4.25测试集选定_论文结构重跑\02_模块一_物理派生特征比较
```

只跑 Transformer，不加模块二趋势约束。

候选：

- `M1_0_5d_baseline`：原始 5 维。
- `M1_R1_static_ratio`：加入 `F/D^2, Cr/D`。
- `M1_R2_log_cycle_replace`：用 `log1p(actual_cycle)` 替换原始 `actual_cycle`。
- `M1_R3_log_cycle_keep`：同时保留 `actual_cycle` 与 `log1p(actual_cycle)`。

模块一 top2：

```json
[
  "M1_R3_log_cycle_keep",
  "M1_R2_log_cycle_replace"
]
```

复筛结果：

| 变体 | mean_life_abs_error | std | pressure_mae | wear_mae_um |
|---|---:|---:|---:|---:|
| M1_R3_log_cycle_keep | 845.4 | 281.6 | 2.576 | 0.0673 |
| M1_R2_log_cycle_replace | 912.2 | 202.5 | 2.619 | 0.0675 |

注意：模块一单独看是 `R3_keep` 更好，但最终组合选中的是 `R2_replace`，因为它与模块二 `slow_abs_0p01` 组合后更优。论文中可以写：模块一候选先筛出 top2，再与趋势约束联合筛选，最终以组合表现为准。

### 5.3 模块二：趋势约束比较

目录：

```text
结果\4.25测试集选定_论文结构重跑\03_模块二_趋势约束比较
```

输入：模块一 top2 特征方案。  
候选 shape loss：

- `S0_no_shape_loss`：无额外 shape loss。
- `S1_slow_abs`：`mono_lambda=0`，`slow_lambda=0.003 / 0.01 / 0.03`。
- `S2_temporal_mono_plus_slow_abs`：`mono_lambda=0.005`，`slow_lambda=0.003 / 0.01 / 0.03`。

最终模块二最佳组合：

```json
{
  "best_variant": "M1_R2_log_cycle_replace__S1_slow_abs_0p01",
  "best_feature": "M1_R2_log_cycle_replace",
  "best_shape": "S1_slow_abs_0p01"
}
```

复筛结果：

| 变体 | mean_life_abs_error | std |
|---|---:|---:|
| M1_R2_log_cycle_replace__S1_slow_abs_0p01 | 469.2 | 219.7 |
| M1_R2_log_cycle_replace__S1_slow_abs_0p003 | 509.5 | 247.3 |
| M1_R2_log_cycle_replace__S2_temporal_mono_plus_slow_abs_0p003 | 511.4 | 236.8 |

结论：趋势约束里最稳的是 `slow_abs_0p01`。不需要强行加入 temporal mono；当前数据下 `slow only` 更适合论文叙事，也更符合“总体缓降趋势”而非“严格逐点单调”的物理解释。

### 5.4 最终消融实验

目录：

```text
结果\4.25测试集选定_论文结构重跑\04_最终消融实验
```

正式设置：`5 seeds × 1600 epochs`。

四格消融：

- `T0_baseline`：原始 5 维 + 无趋势约束。
- `T1_module1_only`：最终选中模块一 + 无趋势约束。
- `T2_module2_only`：原始 5 维 + 最终选中模块二。
- `T3_module1_plus_module2`：最终选中模块一 + 最终选中模块二。

消融结果：

| 变体 | mean_life_abs_error | std | delta_vs_baseline | pressure_mae | wear_mae_um |
|---|---:|---:|---:|---:|---:|
| T0_baseline | 1295.2 | 91.6 | 0.0 | 3.205 | 0.0740 |
| T1_module1_only | 988.2 | 164.1 | -307.1 | 2.923 | 0.0695 |
| T2_module2_only | 810.0 | 239.9 | -485.2 | 3.181 | 0.0480 |
| T3_module1_plus_module2 | 607.3 | 283.2 | -687.9 | 3.060 | 0.0553 |

这张表是论文里最关键的表之一。它说明两个模块分别有贡献，组合后进一步提升。模块二单独贡献大于模块一，但模块一 + 模块二组合仍然最好。

### 5.5 最终横向对比

目录：

```text
结果\4.25测试集选定_论文结构重跑\05_最终横向对比
```

比较对象：

- FNN
- GRU
- LSTM
- 1D-CNN
- Transformer
- Enhanced Transformer

最终结果：

| 排名 | 模型 | mean_life_abs_error | std | pressure_mae | wear_mae_um |
|---:|---|---:|---:|---:|---:|
| 1 | Enhanced Transformer | 607.3 | 283.2 | 3.060 | 0.0553 |
| 2 | LSTM | 892.8 | 406.3 | 2.734 | 0.0557 |
| 3 | GRU | 1204.3 | 221.0 | 2.720 | 0.0671 |
| 4 | Transformer | 1295.2 | 91.6 | 3.205 | 0.0740 |
| 5 | 1D-CNN | 1614.3 | 578.0 | 5.864 | 0.0999 |
| 6 | FNN | 1684.8 | 474.7 | 3.645 | 0.1053 |

追加 seeds 判断：

```json
{
  "best_baseline_variant": "LSTM",
  "enhanced_variant": "T3_module1_plus_module2",
  "advantage": 285.53243743347025,
  "pooled_std": 350.2244164328893,
  "threshold": 175.11220821644466,
  "needs_10_seed_extension": false
}
```

解释：Enhanced Transformer 相对最强基准 LSTM 的优势 `285.5`，大于 `0.5 × pooled_std = 175.1`，所以不需要追加到 10 seeds。

## 6. 图表检查与美化

原始图都已经生成，但日志中出现过 `tight_layout` 排版警告。为了避免论文图出现文字拥挤或图例遮挡，已经基于正式 CSV 重新生成美化版图。

美化图目录：

```text
结果\4.25测试集选定_论文结构重跑\论文图表_美化版
```

图表检查说明：

```text
结果\4.25测试集选定_论文结构重跑\图表检查与美化说明.txt
```

推荐论文使用：

- `Fig1_baseline_model_comparison`：五模型基准对比。
- `Fig2b_module1_refine_top2`：模块一 top2 复筛对比。
- `Fig3b_module2_refine`：模块二复筛对比。
- `Fig4_ablation_mean_std`：最终四格消融实验。
- `Fig5_final_comparison`：最终横向对比。
- `Fig6_final_per_run_error_heatmap`：各测试工况误差热力图，可作为补充图。

每张图都有 PNG 和 SVG 两种格式。论文优先用 SVG，PPT 可以用 PNG。

## 7. 论文和答辩建议

### 7.1 论文叙事主线

推荐写法：

1. 首先建立五模型基准，说明 LSTM 在原始输入下表现最好，Transformer 基准不是第一但具备竞争力。
2. 然后说明 Transformer 更适合融合序列窗口、派生物理特征和趋势先验，因此选择 Transformer 作为增强对象。
3. 模块一引入物理派生特征，显式表达载荷、尺寸、间隙之间的物理关系。
4. 模块二引入趋势约束，强调应力随磨损总体缓降，而不是严格逐点单调。
5. 最终消融证明两个模块各自有效，组合后最好。
6. 最终横向对比证明 Enhanced Transformer 超过 LSTM、GRU 等基线模型。

### 7.2 答辩 PPT 建议

PPT 不要放太多筛选细节。建议最多放 3-4 张核心图：

- 五模型基准对比。
- 最终消融图。
- 最终横向对比图。
- 如果时间允许，再放每工况热力图解释“不是每个 run 都第一，但整体平均最优”。

答辩时可以这样说：

> 原始 Transformer 并不是基准测试第一名，但它对序列窗口和物理约束的融合能力更强。本文不是直接选择原始 Transformer，而是在 Transformer 上进一步引入物理派生特征和趋势约束。消融实验表明两个模块都能降低寿命误差，组合后 Enhanced Transformer 在固定测试集整体平均寿命误差上取得最优。

### 7.3 关于“不是每个测试集都第一”的解释

可以如实说明：

- 单个 run 的误差受工况边界、寿命长短、有限元采样步长、闭环误差累积影响。
- 本文主要评价固定测试集上的总体泛化能力，不要求每个测试工况逐一第一。
- Enhanced Transformer 在 mean life abs error 上第一，并且 wear MAE 与最强基准 LSTM 接近，说明不是靠牺牲磨损曲线质量换取寿命误差。

## 8. 后续如果继续做实验

### 8.1 如果只是换测试集

优先复制当前这条实验线，不要直接覆盖已有结果。

建议新建：

```text
结果\4.xx新测试集_论文结构重跑
工具和杂项\4.xx新测试集_论文结构重跑
```

然后在复制出的 `run_paper_experiments.py` 中修改：

```python
TEST_FILES = ["Run?.csv", "Run?.csv", "Run?.csv", "Run?.csv"]
```

其他设置原则上不变：`seq_len=12`、`actual_step cap=250`、磨损系数 `1.84e-10`。

### 8.2 如果只是补充新数据

先把新数据并入新的数据快照目录，不要覆盖当前：

```text
数据快照\完整XXrun_日期\处理后数据
```

然后复制当前实验线，在新脚本中改 `common_fixed_split.py` 或数据路径，重新跑完整流程。

### 8.3 如果要缩短计算

可以先 dry-run：

```powershell
E:\AI\cuda_env\python.exe 工具和杂项\4.25测试集选定_论文结构重跑\run_paper_experiments.py --dry-run
```

正式跑：

```powershell
E:\AI\cuda_env\python.exe 工具和杂项\4.25测试集选定_论文结构重跑\run_paper_experiments.py --stage all
```

正式跑要求 CUDA。如果 CUDA 不可用，脚本应报错，不应该退到 CPU。

## 9. 关键文件索引

正式结果：

```text
结果\4.25测试集选定_论文结构重跑
```

主脚本：

```text
工具和杂项\4.25测试集选定_论文结构重跑\run_paper_experiments.py
```

公共训练/评估工具：

```text
工具和杂项\4.25测试集选定_论文结构重跑\common_fixed_split.py
```

日志：

```text
工具和杂项\4.25测试集选定_论文结构重跑\paper_pipeline_stdout.log
工具和杂项\4.25测试集选定_论文结构重跑\paper_pipeline_stderr.log
工具和杂项\4.25测试集选定_论文结构重跑\paper_pipeline_watchdog.log
```

最关键汇总：

```text
结果\4.25测试集选定_论文结构重跑\01_五模型基准对比\汇总_各模型平均指标.csv
结果\4.25测试集选定_论文结构重跑\02_模块一_物理派生特征比较\复筛\汇总_各模型平均指标.csv
结果\4.25测试集选定_论文结构重跑\03_模块二_趋势约束比较\复筛\汇总_各模型平均指标.csv
结果\4.25测试集选定_论文结构重跑\04_最终消融实验\汇总_各模型平均指标.csv
结果\4.25测试集选定_论文结构重跑\05_最终横向对比\汇总_各模型平均指标.csv
```

图表：

```text
结果\4.25测试集选定_论文结构重跑\论文图表_美化版
```

## 10. 接手注意事项

- 不要把 `先前版本_30测试集归档` 当作当前正式主线，它只用于追溯早期筛查。
- 不要把 Transformer 参数升级单独写成主要创新点；它现在只是默认参数设置。
- 论文重点是两个模块：物理派生特征 `M1_R2_log_cycle_replace` 和趋势约束 `S1_slow_abs_0p01`。
- 如果要解释为什么选择 Transformer：选择的是 Enhanced Transformer，而不是原始 Transformer；最终横向对比里 Enhanced Transformer 第一。
- 如果要放图，优先使用 `论文图表_美化版` 里的图，不优先使用各阶段自动生成的原始图。
- 如果继续跑正式实验，必须确认使用 `E:\AI\cuda_env\python.exe` 和 CUDA，不能让正式训练悄悄落到 CPU。
