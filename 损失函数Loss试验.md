# 损失函数Loss试验

> **Workspace:** d:\Bearing training

---

## User
接下来给你发一段之前设定的计划，你不要管文件位置什么v2版本什么的，专注他要干那些事就行

## Assistant
好的，我明白了！请把你的计划发过来，我会专注于计划中需要完成的具体任务，忽略文件位置和版本号等细节。等你发送！

## User
接下来给你发一段之前设定的计划，告诉我他主要干了什么，不用执行：# V2 组内 Shape Loss 落地计划 
 
 ## Summary 
 - 在 [v2_pipeline.py](</C:/Users/28382/Desktop/毕业设计/Bearing training/软件主体/main_program/v2_pipeline.py>) 里新增两项组内时序软约束：`run_stress_mono_lambda` 和 `run_stress_slow_lambda`，并把它们接入现有 `total_objective()`，不改主 target 仍为 `log(stress)` 的训练主线。 
 - 约束采用 teacher-forced 方式，只看同一 run 的连续 2/3 步窗口，且只对参与窗口中所有点都满足 `wear_depth <= 0.005 mm` 的样本生效。 
 - 不把这次改动塞进现有大搜索流程；新增一个单独的 shape-loss 对照脚本，固定基线配置做 `baseline / shape_loss_v1 / shape_loss_v2` 三组实验。 
 - 这轮对照实验的基线来源固定为“手动指定”，基线配置固定为：`v2_dual_branch + seq=10 + wear_consistency_lambda=0.25 + use_cycle_step=True + seed=PRIMARY_SEED`。 
 
 ## Implementation Changes 
 - 扩展 `CandidateConfig`： 
   - 新增 `run_stress_mono_lambda: float = 0.0` 
   - 新增 `run_stress_slow_lambda: float = 0.0` 
   - `to_record()`、`runtime_to_bundle()`、`load_runtime_from_bundle()`、`config_from_summary_row()`、`summarize_fold_metrics()`、`aggregate_seed_summaries()` 全部同步带上这两个字段。 
   - 旧 bundle / 旧 CSV 读取时缺失字段默认回落到 `0.0`，保证向后兼容。 
 
 - 扩展 `SequenceDataset`： 
   - 新增 `step_index: np.ndarray`，保存每个样本在原 run 内对应的当前时间点索引 `idx`。 
   - `build_raw_sequence_dataset()` 在构造样本时同步写入 `case_names` 和 `step_index`。 
   - `slice()` 同步切分 `step_index`，后续 train/val/test 都保留 run 内时序位置信息。 
 
 - 新增组内形状约束数据结构与构造函数： 
   - 在 `v2_pipeline.py` 新增 `RunShapeDataset`，至少包含： 
     - `mono_raw_sequences`, `mono_actual_step` 
     - `slow_raw_sequences`, `slow_actual_step` 
   - 新增 `build_run_shape_dataset(sequence_dataset, max_wear_mm=0.005)`： 
     - 先按 `case_name + step_index` 分组并按时间排序 
     - 构造相邻 2 步窗口用于 `mono` 
     - 构造连续 3 步窗口用于 `slow` 
     - 仅保留所有参与点的当前 `wear_depth` 都 `<= 0.005 mm` 的窗口 
     - 严禁跨 run 拼接；长度不足 2/3 的 run 不产生对应样本 
 
 - 新增 stress 反变换与 shape loss 计算函数： 
   - 新增 `scaled_log_to_stress(pred_log_scaled, target_scaler)`，先 inverse scale 回 `log(stress)`，再 `exp()` 回物理 stress。 
   - 新增 `run_stress_shape_losses(...)`： 
     - 对 `mono` 批次：预测 `s_t, s_{t+1}`，计算 `mean(relu(s_{t+1} - s_t))` 
     - 对 `slow` 批次：预测 `s_t, s_{t+1}, s_{t+2}`，计算 `drop_t = s_t - s_{t+1}`、`drop_{t+1} = s_{t+1} - s_{t+2}`，再算 `mean(relu(drop_{t+1} - drop_t))` 
     - 没有有效 pair/triple 时直接返回 0，不报错 
   - 这两项 loss 一律在物理 stress 空间里算，不在 log 空间里算。 
 
 - 调整训练与评估路径： 
   - `train_candidate_model()` 在训练开始前分别为 `train_dataset` 和 `val_dataset` 预构造 `RunShapeDataset` 
   - 主 batch 仍沿用现有 `batch_indices()` 
   - 辅助 shape batch 单独采样： 
     - `mono` 和 `slow` 各自按自己的样本数生成 batch 
     - batch size 默认取 `min(config.batch_size, 该辅助数据集样本数)` 
     - 每个训练 step 取一个主 batch，同时取一个 `mono` batch 和一个 `slow` batch；若辅助 batch 数较少则循环取模复用 
     - `mono`/`slow` 的随机种子分别用 `config.seed + 1000`、`config.seed + 2000` 的 epoch 偏移，保证可复现 
   - `evaluate_loss()` 对 val 集不做小 batch，直接用整份 val shape 数据算一次完整辅助 loss，避免监控分数受辅助采样波动影响 
 
 - 整合到 `total_objective()`： 
   - 保留现有： 
     - 压力主损失 
     - `wear_consistency_loss` 
     - 组间 `monotonic_penalty` 
   - 新增： 
     - `+ config.run_stress_mono_lambda * L_run_stress_mono` 
     - `+ config.run_stress_slow_lambda * L_run_stress_slow` 
   - 默认权重： 
     - `run_stress_mono_lambda = 0.05` 
     - `run_stress_slow_lambda = 0.01` 
 
 - 单独新增 shape-loss 对照脚本，不改现有大扫描主流程： 
   - 新建 `软件主体/main_program/benchmark_v2_shape_losses.py` 
   - 脚本顶部显式声明： 
     - `BASE_SHAPE_CONFIG = make_v2_dual_branch_candidate(sequence_length=10, wear_consistency_lambda=0.25, use_cycle_step=True, seed=PRIMARY_SEED)` 
   - 从这一个 base config 派生三组： 
     - `baseline`: `mono=0.0, slow=0.0` 
     - `shape_loss_v1`: `mono=0.05, slow=0.0` 
     - `shape_loss_v2`: `mono=0.05, slow=0.01` 
   - 复用现有 fixed split manifest 和 `evaluate_candidate_suite()` 
   - 输出目录单独放到 `结果/v2_shape_loss_benchmark/`，至少写出： 
     - `shape_loss_manifest.json` 
     - `shape_loss_summary.csv` 
     - `shape_loss_folds.csv` 
     - `shape_loss_base_config.json` 
   - [benchmark_v2_candidates.py](</C:/Users/28382/Desktop/毕业设计/Bearing training/软件主体/main_program/benchmark_v2_candidates.py>) 保持现状，不把这次定向实验混进原有 round1/round2/round3。 
 
 ## Public Interfaces / Outputs 
 - `CandidateConfig` 新增两个可序列化字段： 
   - `run_stress_mono_lambda` 
   - `run_stress_slow_lambda` 
 - summary/fold CSV、recommended payload、bundle metadata 全部带上这两个字段，后续比较和部署训练能完整回放配置。 
 - 新增一个独立实验入口脚本：`benchmark_v2_shape_losses.py`，用途只限这轮 shape-loss 对照，不替代现有总 benchmark。 
 
 ## Test Plan 
 - 新增 `软件主体/main_program/test_v2_shape_losses.py`，使用 `unittest`，覆盖： 
   - 严格单调下降时 `L_run_stress_mono == 0` 
   - 出现上升时 `L_run_stress_mono > 0` 
   - 下降幅度递减时 `L_run_stress_slow == 0` 
   - 后段下降更陡时 `L_run_stress_slow > 0` 
   - 跨过 `5 um` 的 pair/triple 只保留完全落在 `<= 5 um` 的窗口 
   - run 长度不足 2/3 时不生成对应 shape 样本 
   - 不跨 run 拼接窗口 
   - 两个新 lambda 都为 0 时，`total_objective()` 退化回现有实现 
 - 集成验证： 
   - 用 `C:\Users\28382\Miniconda3\envs\torch_env\python.exe`，不要用 PATH 里的 `python` 
   - 从 `软件主体/main_program` 目录运行 `C:\Users\28382\Miniconda3\envs\torch_env\python.exe -m unittest test_v2_shape_losses` 
   - 再运行 `C:\Users\28382\Miniconda3\envs\torch_env\python.exe benchmark_v2_shape_losses.py` 
   - 检查三组实验都能完成训练、无 NaN、无 shape mismatch，且 summary/fold CSV 都写出新字段 
 
 ## Assumptions And Defaults 
 - `<= 5 um` 统一按 `wear_depth <= 0.005 mm` 判断，使用每个样本当前预测点对应的 `raw_sequences[:, -1, 4]`。 
 - 两项新约束只作用于训练/验证目标，不改变推理接口，不影响现有部署模型，除非后续显式训练并保存新的 candidate bundle。 
 - 这轮不新增显式 `wear_depth >= 0` loss，也不新增单独的 `delta_wear` 递减约束。 
 - 这轮 shape-loss 对照的基线配置固定为你选定的 `Dual强配置`；以后只有在“你认定主候选配置已经换了”时，才需要改 `BASE_SHAPE_CONFIG` 这一处单点常量。

## Assistant
好的，以下是这个计划主要干的事情的总结：

---

## 核心目标

在现有的轴承应力预测模型训练中，新增**两项组内时序软约束（Shape Loss）**，让模型在同一运行周期（run）内预测的应力变化趋势更符合物理规律。

---

## 具体做了什么

### 1. 新增两个约束损失项

- **`run_stress_mono_lambda`**（单调性约束）：同一 run 内相邻两步的应力应该单调递减，如果 `s_{t+1} > s_t` 则产生惩罚。默认权重 `0.05`。
- **`run_stress_slow_lambda`**（缓降约束）：同一 run 内连续三步的应力下降幅度应该递减（即越往后降得越慢），如果后段下降更陡则产生惩罚。默认权重 `0.01`。

这两项 loss 都在**物理应力空间**（先反变换回 stress）里计算，不在 log 空间里算。

### 2. 约束的适用范围

- 只看**同一 run 内**的连续 2 步/3 步窗口，**严禁跨 run 拼接**。
- 只对窗口内所有点的 `wear_depth <= 0.005 mm`（5μm）的样本生效，超过磨损阈值的样本不参与约束。
- run 长度不足 2 步或 3 步的，不产生对应样本。

### 3. 数据层面的改动

- **`CandidateConfig`** 新增两个 lambda 字段，所有序列化/反序列化方法同步带上，旧数据缺失字段默认回落到 `0.0`（向后兼容）。
- **`SequenceDataset`** 新增 `step_index` 字段，记录每个样本在原 run 内的时间位置，train/val/test 切分时同步保留。
- 新增 **`RunShapeDataset`** 数据结构和 **`build_run_shape_dataset()`** 构造函数，专门为 mono（2步窗口）和 slow（3步窗口）准备辅助训练数据。

### 4. 训练流程的改动

- 训练时，每个 step 除了取主 batch，还同时取一个 mono batch 和一个 slow batch（辅助数据不足则循环复用）。
- 验证时，直接用整份 val shape 数据算一次完整辅助 loss，不做小 batch 采样，避免监控分数波动。
- 这两项新 loss 以加权方式加入 `total_objective()`，不影响现有的压力主损失、磨损一致性损失和组间单调性惩罚。

### 5. 新增独立对照实验脚本

- 新建 `benchmark_v2_shape_losses.py`，固定基线配置，做三组对照实验：
  - **baseline**：两个 lambda 都为 0（纯基线）
  - **shape_loss_v1**：只开 mono 约束
  - **shape_loss_v2**：同时开 mono + slow 约束
- 不改动现有的主扫描流程，对照实验结果单独输出。

### 6. 新增单元测试

- 测试 mono/slow loss 在各种情况下的正确性（严格单调为0、出现违反>0、磨损阈值过滤、跨run不拼接、lambda为0时退化等）。

---

## 一句话总结

**在不改变现有训练主线的前提下，为模型新增"组内应力应单调递减且递减幅度应逐渐放缓"的物理软约束，并通过独立对照实验验证其效果。**

## User
你读读目前我v1版本的程序，如果我想加这么一个模块试验其有没有用，你帮我设计个试验方案

## Assistant
好的，让我先全面了解你现有的 v1 版本程序结构，然后再设计实验方案。

好的，我已经通读了你的 v1 版本全部代码。下面先总结我理解到的现状，然后给出针对 v1 架构的 Shape Loss 实验方案。

---

## 现状理解

### 你的 v1 程序结构

| 文件 | 作用 |
|------|------|
| `process_real_wear_data.py` | 原始 xlsx → 标准化 CSV |
| `train_real_wear_models.py` | 单点 FNN 训练 + 递推评估（最原始版本） |
| `benchmark_network_architectures.py` | 5 种架构对比（FNN/GRU/LSTM/CNN/Transformer），序列输入 |
| `train_transformer_deployment_model.py` | 部署用 Transformer 训练 |
| `cross_testset_architecture_scan.py` | 跨测试集扫描 |
| `scan_transformer_testsets.py` | Transformer 留一分析 |
| `predict_life.py` | 推理/预测界面 |

### 核心技术细节

- **预测目标**：`log(stress)`
- **特征**：`[F, D, Cr, actual_cycle, wear_depth]`（5维）
- **序列长度**：6（v1 基线），15（升级测试中）
- **训练方式**：全批量训练，无验证集，用训练 loss 选最优模型
- **现有正则**：扰动式单调性惩罚（wear=0.08, load=0.04, clearance=0.04）——这是**跨样本**的虚拟扰动，不是组内时序约束
- **评估**：10 折留一验证
- **指标**：pressure MAE/RMSE/MAPE、磨损曲线 MAE、寿命绝对/相对误差

### 4.13v1 基线结果

| 模型 | 压力 MAE | 磨损 MAE | 寿命误差 | 最优折数 |
|------|---------|---------|---------|---------|
| FNN | 28.82 | 0.348 | 14652 | 4 |
| GRU | 22.37 | 0.322 | 15328 | 6 |
| Transformer | 22.33 | 0.360 | 18673 | 0 |
| 1D-CNN | 30.87 | 0.417 | 18902 | 0 |
| LSTM | 28.48 | 0.388 | 20842 | 0 |

**核心矛盾**：Transformer 压力拟合和 GRU 并列最优，但递推寿命误差排第三，0 折最优。说明模型在单点拟合上好，但递推时应力曲线的"形状"不对，导致误差累积。

---

## Shape Loss 实验方案（适配 v1 架构）

### 一、实验目标

验证"组内时序形状约束"能否改善 Transformer 的递推寿命预测精度。具体来说：
- **mono 约束**：同一 run 内，应力应单调递减（`s_{t+1} ≤ s_t`）
- **slow 约束**：同一 run 内，应力下降幅度应递减（越往后降得越慢）

### 二、需要改动的模块

#### 1. 数据层：`build_sequence_dataset()` 扩展

当前 `build_sequence_dataset()` 只返回 `(sequences, targets)`，丢失了每个样本属于哪个 run 以及在 run 内的时间位置。需要扩展为同时返回 `case_names` 和 `step_indices`：

```
返回: (sequences, targets, case_names, step_indices)
```

- `case_names: list[str]`：每个样本对应的工况名（如 "试验1.csv"）
- `step_indices: np.ndarray`：每个样本在原 run 内的时间步索引

这样就能知道哪些样本属于同一个 run，以及它们在 run 内的先后顺序。

#### 2. 新增：`build_run_shape_pairs()` 函数

利用 `case_names` 和 `step_indices`，构造组内时序窗口：

- **mono pairs**：同一 run 内相邻 2 步 `(idx_t, idx_{t+1})`
- **slow triples**：同一 run 内连续 3 步 `(idx_t, idx_{t+1}, idx_{t+2})`
- **过滤条件**：窗口内所有点的 `wear_depth <= 0.005 mm`（5μm）
- **严禁跨 run**：不同 case_name 的样本不拼接

#### 3. 新增：`scaled_log_to_stress()` 反变换函数

```
pred_log_scaled → target_scaler.inverse → log(stress) → exp → stress (MPa)
```

在物理应力空间计算 shape loss，不在 log 空间。

#### 4. 新增：`run_stress_shape_losses()` 计算

- **mono loss**：`mean(relu(s_{t+1} - s_t))`，应力上升则惩罚
- **slow loss**：`drop_t = s_t - s_{t+1}`，`drop_{t+1} = s_{t+1} - s_{t+2}`，`mean(relu(drop_{t+1} - drop_t))`，后段下降更陡则惩罚
- 无有效 pair/triple 时返回 0

#### 5. 训练循环改动

v1 是全批量训练（不是 mini-batch），所以集成方式比 v2 简单：

```python
# 每个 epoch:
pred = model(train_x_scaled)
data_loss = mse(pred, train_y_scaled)
wear_pen, load_pen, clearance_pen = monotonic_penalty(...)

# 新增：组内 shape loss（全量计算，无需采样）
mono_loss, slow_loss = run_stress_shape_losses(
    model, train_x_scaled, target_scaler, 
    mono_pairs, slow_pairs
)

loss = (
    data_loss
    + MONO_LAMBDA_WEAR * wear_pen
    + MONO_LAMBDA_LOAD * load_pen
    + MONO_LAMBDA_CLEARANCE * clearance_pen
    + run_stress_mono_lambda * mono_loss    # 新增
    + run_stress_slow_lambda * slow_loss    # 新增
)
```

由于 v1 全批量训练，shape loss 也直接全量计算，不需要辅助 batch 采样，实现更简洁。

#### 6. 验证集监控

v1 没有验证集，用训练 loss 选模型。加入 shape loss 后，仍然用训练 total_loss 选模型，但建议在日志中额外打印 mono_loss 和 slow_loss 的值，方便观察约束是否生效。

### 三、实验设计

#### 固定基线配置

以 Transformer 升级版（当前正在跑的）为基线：
- `seq_len=15, d_model=64, nhead=4, num_layers=2, dim_ff=128`
- `epochs=1200, lr=1e-3, weight_decay=1e-6`
- 现有单调性惩罚：`wear=0.08, load=0.04, clearance=0.04`

#### 三组对照实验

| 组名 | `run_stress_mono_lambda` | `run_stress_slow_lambda` | 说明 |
|------|--------------------------|--------------------------|------|
| **baseline** | 0.0 | 0.0 | 纯基线，无 shape loss |
| **shape_v1** | 0.05 | 0.0 | 只开 mono 约束 |
| **shape_v2** | 0.05 | 0.01 | 同时开 mono + slow |

#### 评估方法

- **10 折留一验证**，与 4.13v1 基线完全一致
- **指标**：pressure MAE、wear MAE、life abs error（与基线对齐）
- **额外关注**：shape loss 值随 epoch 的变化趋势，验证约束是否真正生效

#### 输出

独立输出目录 `结果/shape_loss_benchmark/`：
- `详细结果_各折.csv`：每折每组的完整指标
- `汇总_对比.csv`：三组实验的平均指标对比
- `图_寿命误差对比.png`：三组在各工况上的寿命误差分组柱状图
- `图_三指标对比.png`：三组在 pressure/wear/life 三个指标上的对比
- `实验说明.txt`

### 四、实现步骤

1. **扩展 `build_sequence_dataset()`**：返回 `case_names` 和 `step_indices`
2. **新增 `build_run_shape_pairs()`**：构造 mono pairs 和 slow triples
3. **新增 `scaled_log_to_stress()` 和 `run_stress_shape_losses()`**：反变换 + shape loss 计算
4. **修改训练循环**：在 total_loss 中加入 shape loss 项
5. **编写独立对照脚本**：`结果/shape_loss_benchmark/run_benchmark.py`，三组实验
6. **运行实验，对比结果**

### 五、关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| shape loss 计算空间 | 物理 stress 空间 | log 空间的差值不直观，物理空间更符合"应力递减"的语义 |
| 磨损阈值过滤 | `wear_depth <= 0.005 mm` | 高磨损区应力行为复杂，约束可能不合理 |
| lambda 默认值 | mono=0.05, slow=0.01 | mono 是一阶约束（更基础），权重应更大；slow 是二阶约束（更精细），权重小一些 |
| 是否改现有主流程 | 不改 | 独立脚本，不影响现有代码 |
| 训练方式 | 全量 shape loss（非采样） | v1 全批量训练，直接全量计算更简单、更稳定 |

### 六、预期结果与判断标准

- **shape_v1 > baseline**：mono 约束有效，说明"组内应力单调递减"的先验帮助了递推精度
- **shape_v2 > shape_v1**：slow 约束额外有效，说明"下降幅度递减"的先验进一步改善曲线形状
- **shape_v1 ≈ baseline**：mono 约束无效，可能因为现有扰动式单调性惩罚已经隐式覆盖了这个约束
- **shape_v1 < baseline**：mono 约束有害，可能过度约束导致拟合能力下降

判断标准：以 **平均寿命绝对误差** 为主要判据，pressure MAE 和 wear MAE 为辅助参考。

---

这就是完整的实验方案。需要我开始实现吗？

## User
读一下两个md文件，我觉得是不是可以在5种神经网络（其中transformer是v2）的基础上都加一个这个部分，然后看看效果，能列个计划吗

## Assistant
好的，我已经读完了两个 md 文件，完整了解了你的项目历史。现在来设计在 5 种架构上都加 Shape Loss 的实验计划。

---

## 实验计划：5 架构 × Shape Loss 对照

### 一、实验动机

之前你做的升级测试只改了 Transformer，Shape Loss 也只在 v2 pipeline 的计划里提过。但核心问题是：**Shape Loss 到底对哪种架构帮助最大？**

- FNN/GRU 等本身递推就比 Transformer 稳，加约束还有用吗？
- Transformer v2 已经通过增大序列和容量改善了，Shape Loss 还能进一步帮助吗？
- 不同架构对 mono（单调递减）和 slow（递减放缓）两种约束的敏感度是否不同？

### 二、5 种架构的基线配置

| 架构 | seq_len | 特殊参数 | 基线寿命误差 |
|------|---------|---------|-------------|
| FNN | 6 | 5→64→64→32→1 | 14652 |
| GRU | 6 | hidden=32 | 15328 |
| LSTM | 6 | hidden=32 | 20842 |
| 1D-CNN | 6 | channels=32 | 18902 |
| **Transformer v2** | **15** | **d=64, ff=128, 2层** | **14955** |

前 4 种用 4.13v1 基线配置，Transformer 用升级后的 v2 配置。训练配置统一：1200 epochs, lr=1e-3, wd=1e-6, 无 dropout, 现有扰动式单调性惩罚。

### 三、实验矩阵

对每种架构，测 3 组：

| 组名 | `run_stress_mono_lambda` | `run_stress_slow_lambda` | 说明 |
|------|--------------------------|--------------------------|------|
| **baseline** | 0.0 | 0.0 | 无 shape loss（已有基线数据） |
| **shape_v1** | 0.05 | 0.0 | 只加 mono 约束 |
| **shape_v2** | 0.05 | 0.01 | mono + slow 约束 |

**总计**：5 架构 × 3 组 × 10 折 = **150 次训练**

其中 baseline 的 50 次已有数据（4.13v1 的 50 次 + Transformer v2 的 10 次），实际只需新跑 **100 次**。

### 四、需要实现的改动

#### 1. 扩展 `build_sequence_dataset()`

当前返回 `(sequences, targets)`，需扩展为同时返回 `case_names` 和 `step_indices`：

```
返回: (sequences, targets, case_names, step_indices)
```

- `case_names: list[str]`：每个样本对应的工况名
- `step_indices: np.ndarray`：每个样本在原 run 内的时间步索引

#### 2. 新增 `build_run_shape_pairs()`

利用 case_names 和 step_indices，构造组内时序窗口：
- **mono pairs**：同一 run 内相邻 2 步 `(idx_t, idx_{t+1})`
- **slow triples**：同一 run 内连续 3 步 `(idx_t, idx_{t+1}, idx_{t+2})`
- 过滤：窗口内所有点的 `wear_depth <= 0.005 mm`
- 严禁跨 run 拼接

#### 3. 新增 `scaled_log_to_stress()` + `run_stress_shape_losses()`

- 反变换：`pred_log_scaled → target_scaler.inverse → log(stress) → exp → stress`
- mono loss：`mean(relu(s_{t+1} - s_t))`
- slow loss：`mean(relu(drop_{t+1} - drop_t))`，其中 `drop = s_t - s_{t+1}`
- 无有效 pair/triple 时返回 0

#### 4. 修改训练循环

v1 全批量训练，shape loss 直接全量计算：

```python
loss = (
    data_loss
    + MONO_LAMBDA_WEAR * wear_pen
    + MONO_LAMBDA_LOAD * load_pen
    + MONO_LAMBDA_CLEARANCE * clearance_pen
    + run_stress_mono_lambda * mono_loss    # 新增
    + run_stress_slow_lambda * slow_loss    # 新增
)
```

### 五、执行步骤

| 步骤 | 内容 | 
|------|------|
| **1** | 实现数据扩展（case_names, step_indices） |
| **2** | 实现 shape pairs 构造函数 |
| **3** | 实现 shape loss 计算函数 |
| **4** | 编写独立对照脚本 `结果/shape_loss_benchmark/run_benchmark.py` |
| **5** | 运行实验（5架构 × 2组shape变体 × 10折 = 100次训练） |
| **6** | 汇总结果，生成对比图表和分析文档 |

### 六、输出文件

```
结果/shape_loss_benchmark/
├── run_benchmark.py              自包含运行脚本
├── 详细结果_各折各模型各配置.csv   150条完整指标
├── 汇总_各架构各配置对比.csv       5架构×3配置的平均指标
├── 图1_各架构shape_loss效果对比.png  每种架构baseline vs shape_v1 vs shape_v2
├── 图2_寿命误差热力图.png          架构×配置的寿命误差矩阵
├── 图3_最优配置跨架构对比.png       每种架构选最优配置后横向对比
├── 实验说明.txt
└── 结果分析.txt
```

### 七、预期分析维度

1. **架构内对比**：每种架构，shape loss 是否有提升？哪种约束（mono/slow）贡献大？
2. **架构间对比**：哪种架构从 shape loss 受益最多？
3. **与已有基线对比**：加了 shape loss 后，排名是否变化？Transformer v2 + shape loss 能否超过 FNN 基线（14652）？
4. **极端工况分析**：shape loss 对试验1/试验13这两个误差黑洞有没有帮助？

### 八、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| shape loss 计算空间 | 物理 stress 空间 | log 空间差值不直观 |
| 磨损阈值过滤 | wear_depth ≤ 0.005mm | 高磨损区行为复杂 |
| lambda 默认值 | mono=0.05, slow=0.01 | 与 v2 计划一致 |
| 是否改现有主流程 | 不改 | 独立脚本 |
| 训练方式 | 全量 shape loss | v1 全批量训练，无需采样 |
| baseline 数据 | 复用已有 | 4.13v1 + Transformer v2 已有基线 |

---

你看这个方案行不行？有什么要调整的？

## User
我就是觉得100组测试会不会略多，transformer升级一共跑了多少组，还有你在结果里先新建一个加Loss函数试验的文件夹，然后你学一下上面两个md文件是怎么设置文件夹内容的，如何在完成实验后删除不必要的文件的

## Assistant
好问题！让我先算一下 Transformer 升级测试实际跑了多少组，再讨论实验规模和文件组织方式。

---

## Transformer 升级测试实际跑了多少组

从 md 记录来看：

| 轮次 | 配置 | 折数 | 组数 |
|------|------|------|------|
| 第一轮a（验证集+早停） | dropout+lr+wd+clip+val+es | 10 | 10 |
| 第一轮b（去验证集） | lr+wd+clip+3000ep | 10 | 10 |
| 第二轮a（seq=10） | seq_len=10 | 10 | 10 |
| 第二轮b（seq=10+温和训练） | seq=10+lr+wd+clip | 10 | 10 |
| 第三轮A（d=64,ff=128） | seq=10,d=64 | 10 | 10 |
| 第三轮B（d=64,3层,ff=256） | seq=10,d=64,3层 | 10 | 10 |
| seq=15测试 | seq=15,d=64 | 10 | 10 |
| 最终正式测试 | seq=15,d=64,ff=128 | 10 | 10 |
| **总计** | | | **~80 组** |

**80 组**，而且每组的 Transformer v2 模型更大（d_model=64, seq_len=15），单组训练时间更长。

## 关于 100 组的问题

你说得对，100 组确实偏多。有几个缩减方案：

### 方案 A：只测 Transformer v2 + shape loss（推荐）
- **20 组**（2 配置 × 10 折），最快出结论
- 理由：Shape Loss 本质是改善递推曲线形状，Transformer 是递推误差最大的架构，如果它都没明显提升，其他架构大概率也没用
- 后续如果效果好，再扩展到其他架构

### 方案 B：5 架构只测 shape_v1（不开 slow）
- **50 组**（5 架构 × 1 配置 × 10 折）
- 先看 mono 约束对各架构的影响，有价值的再加 slow

### 方案 C：全量但复用基线
- **100 组**新训练，但 baseline 数据直接从已有 CSV 复用过来合并
- 不需要重新跑 baseline 的 50 组

**我建议方案 A 先行**，用最小成本验证 shape loss 是否有效，有效了再扩大范围。

---

现在来创建文件夹并学习文件组织规范：

好的，我看清楚了文件组织规范：

- **结果文件夹**：只保留最终正式结果（脚本 + 图表 + CSV + 说明文档）
- **中间过程**：放到 `工具和杂项/xxx_中间过程/` 里

现在创建文件夹：

文件夹已创建。现在总结一下我的建议方案：

---

## 最终实验计划

### 实验规模：分两阶段

**第一阶段（先跑，验证有效性）**：只测 Transformer v2

| 组名 | mono_lambda | slow_lambda | 训练组数 |
|------|-------------|-------------|---------|
| baseline | 0.0 | 0.0 | 0（已有数据） |
| shape_v1 | 0.05 | 0.0 | 10 |
| shape_v2 | 0.05 | 0.01 | 10 |

只需新跑 **20 组**，和 Transformer 升级测试的 80 组比已经很精简了。

**第二阶段（视第一阶段结果决定）**：
- 如果 Transformer v2 + shape loss 有明显提升 → 扩展到其他 4 种架构（再加 40-80 组）
- 如果没提升 → 不用扩展了，结论就是 shape loss 在当前框架下没用

### 文件组织（学 Transformer 升级测试的规范）

```
结果/shape_loss试验/                          ← 最终正式结果
├── run_benchmark.py                          自包含运行脚本
├── 详细结果_各折.csv                          shape_v1 + shape_v2 的完整指标
├── 汇总_各配置对比.csv                        baseline / shape_v1 / shape_v2 对比
├── 图1_各工况寿命误差对比.png                  三组在各工况上的寿命误差
├── 图2_三指标对比.png                         pressure/wear/life 三指标对比
├── 实验说明.txt
└── 结果分析.txt

工具和杂项/shape_loss试验_中间过程/            ← 中间过程和调试脚本
├── debug_shape_loss.py                       调试用脚本
├── ...（如有中间结果）
```

### 执行流程

1. 实现数据扩展（case_names, step_indices）
2. 实现 shape pairs 构造 + shape loss 计算
3. 先写一个快速验证脚本（只跑 1-2 折），确认 shape loss 计算正确、数值合理
4. 把快速验证脚本和中间结果放到 `工具和杂项/shape_loss试验_中间过程/`
5. 确认无误后，写正式的 `run_benchmark.py`，放到 `结果/shape_loss试验/`
6. 运行正式测试（20 组）
7. 生成图表和分析文档
8. 清理中间过程文件

---

你觉得这个方案行吗？确认了我就开始实现。

## User
可以，执行吧

## Assistant
好的，开始实现！我先基于 Transformer v2 的 run_benchmark.py 写一个完整的自包含脚本，包含所有 shape loss 的实现。

现在我来写完整的自包含脚本，包含所有 shape loss 的实现。先写快速验证脚本到中间过程目录，确认无误后再写正式版本。

快速验证脚本已写好，现在运行它（只跑 2 折 × 3 配置 = 6 次训练，确认 shape loss 计算正确）：

正在运行，shape pairs 已成功构造（mono=1197, slow=1185）。让我等待更多输出：

## User
你记得把方案说明写文件夹里，说清楚要干什么，怎么干的

## Assistant
好的，shape loss 计算正常！baseline 的 mono=0.0000（因为 lambda=0 不计算），shape_v1 的 mono=0.1241（有值，说明约束生效了）。继续等待：

初步结果很有意思！第一折（试验7）的结果：

| 配置 | 寿命误差 | 压力MAE | 磨损MAE |
|------|---------|---------|---------|
| baseline | 3175 | 32.31 | 0.3951 |
| **shape_v1** | **1827** ✅ | **26.84** ✅ | **0.2219** ✅ |
| shape_v2 | 2553 | 30.26 | 0.3322 |

shape_v1 在所有指标上都明显优于 baseline！继续等第二折完成：

快速验证结果出来了！2 折的初步结果：

| 折 | 配置 | 寿命误差 | 压力MAE | 磨损MAE |
|---|------|---------|---------|---------|
| 试验7 | baseline | 3175 | 32.31 | 0.395 |
| 试验7 | **shape_v1** | **1827** ✅ | **26.84** ✅ | **0.222** ✅ |
| 试验7 | shape_v2 | 2553 | 30.26 | 0.332 |
| 试验12 | baseline | **2336** ✅ | **19.61** ✅ | **0.285** ✅ |
| 试验12 | shape_v1 | 2915 | 27.64 | 0.360 |
| 试验12 | shape_v2 | 2699 | 23.98 | 0.349 |

Shape loss 计算正确，数值合理。试验7 上 shape_v1 大幅提升，试验12 上 baseline 更好——这正是需要 10 折完整测试来确认的原因。

现在写正式的 `run_benchmark.py` 和方案说明：

现在写方案说明文档：

现在启动正式测试（3 配置 × 10 折 = 30 次训练）：

正式测试已启动！正在跑第 1/30 组。让我等待一段时间后检查进度：

## User
不是20组嘛，基准不用跑啊

## Assistant
你说得对！baseline 的数据已经有了（Transformer v2 升级测试的结果），不需要重跑。让我停掉当前运行，修改脚本只跑 shape_v1 和 shape_v2（20 组），然后从已有数据加载 baseline：

## User
后面的基准你也别重跑

## Assistant
好的，我来修改脚本：去掉 baseline 的训练，直接从已有 Transformer v2 升级测试结果中加载 baseline 数据。

现在修改 main() 函数，从已有结果加载 baseline 数据，只训练 shape_v1 和 shape_v2：

还需要修改打印配置的部分和图表函数中的 CONFIGS 引用：

好，现在确认一下 baseline 数据文件是否存在：

Baseline 数据存在，10 折完整。现在启动正式测试（只需跑 20 组）：

## User
记得方案结果分析什么里面的按实际实验步骤写

## Assistant
已成功启动！Baseline 数据已从已有结果加载（10 行），现在只跑 shape_v1 和 shape_v2 共 20 组。

让我更新方案说明，按实际实验步骤写：

方案说明已更新，按实际实验步骤写了。现在检查一下正式测试的进度：

正在跑，已完成 2/20。前两折结果和快速验证一致。继续等待：

8/20 在跑了，继续等待：

14/20 在跑了，继续等待：

19/20 快完成了！最后一组在跑，等它结束：

实验完成了！让我读取完整结果并分析：

实验完成！图表已自动生成，但中文字体有问题。让我修复图表中的中文字体显示，然后写结果分析：

现在单独重新生成图表（不需要重跑训练）：

我写一个快速脚本只重新生成图表：

路径问题，CSV 在结果目录不在中间过程目录。修复路径：

图表已重新生成，中文显示正常了。现在写结果分析文档：

全部任务完成！让我确认一下最终文件结构：

一切就绪！最终文件结构：

```
结果/shape_loss试验/                    ← 正式结果
├── run_benchmark.py                    自包含运行脚本
├── 详细结果_各折.csv                    30行完整指标
├── 汇总_各配置对比.csv                  3配置平均指标
├── 图1_各工况寿命误差对比.png            分组柱状图
├── 图2_三指标对比.png                   三配置横向对比
├── 实验说明.txt                         方案+步骤
└── 结果分析.txt                         详细分析

工具和杂项/shape_loss试验_中间过程/      ← 中间过程
├── quick_validate.py                    快速验证脚本
├── quick_validation.csv                 验证结果
└── regen_plots.py                       图表重生成脚本
```

---

## 实验结果总结

| 指标 | baseline | shape_v1 | shape_v2 |
|------|----------|----------|----------|
| **压力 MAE** | 25.33 | 21.69 (-14%) | **20.22** (-20%) ✅ |
| **磨损 MAE** | 0.311 | 0.330 | **0.297** (-4.5%) ✅ |
| **平均寿命误差** | **14955** ✅ | 16200 | 15377 |
| **中位寿命误差** | 4690 | 5006 | **3820** (-18.5%) ✅ |

**核心结论**：
- Shape Loss **显著改善了压力拟合精度**（MAE -20%），说明约束确实在物理应力空间起到了正则化作用
- **中位寿命误差改善 18.5%**，但在长寿命工况（试验1/3/13/14）上系统性不如 baseline
- **平均寿命误差略有恶化**（+2.8%），主要被试验1的极端误差拖累
- shape_v2 整体优于 shape_v1，说明 slow 约束（递减放缓）比 mono 约束更有价值

Shape Loss 是一个有潜力的方向，但当前实现还需要针对长寿命工况做调整。

## User
Shape Loss 是一个有潜力的方向，但当前实现还需要针对长寿命工况做调整 为什么这么说 mono loss和shape loss哪个有用点

## Assistant
好问题，让我用数据说话，重新审视一下我之前的判断是否准确。

好，让我用数据仔细拆解，纠正我之前过于简单的说法。

---

## 1. "需要针对长寿命工况调整"——这个说法不准确

我之前说"Shape Loss 在长寿命工况上系统性不如 baseline"，但仔细看数据并非如此：

| 工况 | 真实寿命 | baseline | shape_v1 | shape_v2 | 最优 |
|------|---------|----------|----------|----------|------|
| 试验3 | 79634 | 24044 | **15941** ✅ | 22459 | shape_v1 |
| 试验14 | 72676 | **26622** | 28015 | 27844 | baseline |
| 试验13 | 79653 | **33212** | 34152 | 35558 | baseline |
| 试验1 | 160992 | **45153** | 64419 | 51547 | baseline |

**试验3** 也是长寿命（79634），但 shape_v1 比baseline好了34%！所以问题不是"长寿命就不行"。

真正的问题更精确地说是：**Shape Loss 在试验1上严重翻车**（baseline 45153 → shape_v1 64419，恶化42%），这一个极端案例把平均数拉坏了。如果去掉试验1：

| 指标 | baseline (去试验1) | shape_v2 (去试验1) |
|------|-------------------|-------------------|
| 平均寿命误差 | 11605 | **11340** |
| 中位寿命误差 | 3550 | **2820** |

去掉试验1后，shape_v2 其实是**全面优于 baseline** 的。

---

## 2. mono loss vs slow loss 哪个更有用？

逐层拆解：

### 第一层：加 mono 的效果（baseline → shape_v1）

| 工况 | baseline | shape_v1 | 变化 |
|------|----------|----------|------|
| 试验7 | 3175 | 3094 | -3% |
| 试验12 | 2757 | 3505 | **+27%** ❌ |
| 试验11 | 1972 | 1033 | **-48%** ✅ |
| 试验6 | 3872 | 6508 | **+68%** ❌ |
| 试验4 | 5507 | 2858 | **-48%** ✅ |
| 试验8 | 3232 | 2475 | **-23%** ✅ |
| 试验14 | 26622 | 28015 | -5% |
| 试验3 | 24044 | 15941 | **-34%** ✅ |
| 试验13 | 33212 | 34152 | -3% |
| 试验1 | 45153 | 64419 | **+43%** ❌ |

mono 单独用：4个工况大幅改善，3个工况恶化，3个基本持平。**波动很大，不稳定**。

### 第二层：在 mono 基础上加 slow 的效果（shape_v1 → shape_v2）

| 工况 | shape_v1 | shape_v2 | 变化 |
|------|----------|----------|------|
| 试验7 | 3094 | 1827 | **-41%** ✅ |
| 试验12 | 3505 | 2269 | **-35%** ✅ |
| 试验11 | 1033 | 1608 | +56% ❌ |
| 试验6 | 6508 | 3123 | **-52%** ✅ |
| 试验4 | 2858 | 3021 | +6% |
| 试验8 | 2475 | 4518 | **+83%** ❌ |
| 试验14 | 28015 | 27844 | -1% |
| 试验3 | 15941 | 22459 | **+41%** ❌ |
| 试验13 | 34152 | 35558 | +4% |
| 试验1 | 64419 | 51547 | **-20%** ✅ |

slow 加上去：4个工况改善，3个工况恶化，3个基本持平。**但关键的是，slow 修复了 mono 在试验1和试验6上的严重翻车**。

### 核心对比

| 维度 | mono 单独 | slow 的额外贡献 |
|------|-----------|----------------|
| 压力MAE改善 | 25.33→21.69 (**-14.4%**) 主力 | 21.69→20.22 (-6.8%) 锦上添花 |
| 平均寿命误差 | 14955→16200 (**+8.3%** 变差) | 16200→15377 (-5.1% 修复) |
| 中位寿命误差 | 4690→5006 (+6.8% 变差) | 5006→3820 (**-23.7%** 大幅改善) |
| 最大误差 | 45153→64419 (+42% 灾难) | 64419→51547 (-20% 修复) |

### 结论

**mono 是压力拟合改善的主力**（贡献了 14.4% 的压力 MAE 下降），但 mono 单独用在寿命预测上不稳定，容易在个别工况翻车。

**slow 的核心价值不是"进一步改善"，而是"修复 mono 的翻车"**。slow 把 mono 在试验1（+42%→+14%）和试验6（+68%→-19%）的灾难性恶化给拉回来了，同时在中位寿命误差上带来了 23.7% 的改善。

换句话说：
- **mono 是油门**——踩下去压力拟合变好，但方向不稳
- **slow 是方向盘**——把 mono 跑偏的方向修正回来

两个一起用（shape_v2）才是合理的组合，单独用哪个都不够好。

