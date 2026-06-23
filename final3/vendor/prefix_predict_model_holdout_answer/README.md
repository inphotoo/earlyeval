# SWE-smith Trajectory Prefix Success Prediction

在轨迹进行到第 `t` 步时，仅基于当前前缀的信息，预测该轨迹最终是否 `resolved=True`。

## 项目结构

```
swe_prefix_predict/
├── run_all.py               # 主流程脚本（一键运行全部）
├── config.py                # 全局配置（路径、超参数、GPU 等）
├── utils.py                 # 日志、计时器等工具
├── action_classifier.py     # 动作分类器（taxonomy 实现）
├── observation_parser.py    # Observation 文本信号提取
├── step_builder.py          # Step 重建（messages → step_table）
├── prefix_builder.py        # Prefix 构建（step → prefix_table + 特征 A~H）
├── feature_engineer.py      # 特征工程（dense + TF-IDF）
├── data_split.py            # 按 group_id 分组切分
├── trainer.py               # 模型训练（LR + LightGBM）
├── evaluator.py             # 评估、绘图、报告生成
├── action_taxonomy.yaml     # 动作分类体系元数据
├── feature_dictionary.md    # 完整特征字典文档
├── requirements.txt         # Python 依赖
├── data/                    # 生成的中间数据
│   ├── step_table.parquet
│   └── prefix_table.parquet
├── models/                  # 训练好的模型
│   ├── baseline_dense_lr.pkl
│   ├── baseline_dense_af_lr.pkl
│   ├── baseline_dense_af_thought_lr.pkl
│   ├── baseline_dense_full_lr.pkl
│   ├── baseline_tfidf_af_lr.pkl
│   ├── baseline_tfidf_af_thought_lr.pkl
│   ├── baseline_tfidf_full_lr.pkl
│   ├── baseline_lgbm_dense.lgb
│   ├── baseline_lgbm_dense_af.lgb
│   ├── baseline_lgbm_dense_af_thought.lgb
│   ├── baseline_lgbm_dense_full.lgb
│   ├── baseline_lgbm_tfidf_af.lgb
│   ├── baseline_lgbm_tfidf_af_thought.lgb
│   ├── baseline_lgbm_tfidf_full.lgb
│   └── feature_engineer_*.pkl
├── reports/                 # 评估报告和可视化
│   ├── evaluation_report.txt
│   ├── metrics_summary.csv
│   ├── evaluation_results.json
│   ├── test_predictions_all_models.csv
│   ├── test_predictions_all_models.parquet
│   ├── roc_curve_points_all_models.csv
│   ├── pr_curve_points_all_models.csv
│   ├── curve_data/
│   │   ├── roc_curve_points_<model>.csv
│   │   └── pr_curve_points_<model>.csv
│   ├── calibration_*.png
│   ├── roc_pr_*.png
│   ├── step_metrics_*.png
│   ├── feature_importance_*.png
│   ├── roc_comparison_all.png
│   ├── pr_comparison_all.png
│   └── precision_savings_curve_<model>.png  # Prec(S)/Prec(F) 约束下节省率对比
└── logs/                    # 详细运行日志
    ├── run_all.log
    ├── step_builder.log
    ├── prefix_builder.log
    ├── feature_engineer.log
    ├── trainer.log
    └── evaluator.log
```

## 环境要求

- Python >= 3.10
- GPU: NVIDIA A100-80G（1 号卡），LightGBM GPU 加速可选

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1) 直接完整执行（推荐）

```bash
cd machine/swe_prefix_predict7
pip install -r requirements.txt

# 方式 1：环境变量指定数据目录
export SWE_PARQUET_DIR=/path/to/tool-parquet-files
python run_all.py

# 方式 2：命令行指定数据目录
python run_all.py --data-dir /path/to/tool-parquet-files
```

### 2) 只测试部分数据（快速验证）

```bash
cd machine/swe_prefix_predict7

# 自动挑选一个 parquet，快速验证流程
python run_all.py --data-dir /path/to/tool-parquet-files --quick-verify

# 指定单个 parquet 进行验证
python run_all.py --single-parquet /path/to/tool-xxx.parquet --quick-verify
```

`--quick-verify` 会自动：
- 只跑一个 parquet 文件
- 跳过 LightGBM（`--skip-lgbm`）
- 跳过 ablation（`--skip-ablation`）

适合检查“代码是否跑通、报表是否生成、核心指标是否合理”。

### 3) 常用执行参数

```bash
# GPU 不可用时，LightGBM 改 CPU
python run_all.py --data-dir /path/to/data --no-gpu-lgbm

# 跳过 LightGBM
python run_all.py --data-dir /path/to/data --skip-lgbm

# 跳过 ablation
python run_all.py --data-dir /path/to/data --skip-ablation

# 若已有中间表，跳过重建
python run_all.py --skip-step-table --skip-prefix-table
```

### 4) 分步执行（调试时使用）

```bash
# 1. 构建 step table
python step_builder.py

# 2. 构建 prefix table
python prefix_builder.py

# 3. 运行训练评估（复用已有中间产物）
python run_all.py --skip-step-table --skip-prefix-table
```

## 新版 TF-IDF 与训练加速说明

### TF-IDF 维度控制（分块降维）

当前实现为“每个文本块独立 TF-IDF，再独立 SVD 降维，再拼接”：
- 保留不同文本来源（task/action/feedback/thought/content）的语义边界
- 避免直接混成单一词袋导致信息耦合
- 将总维度压到几百量级，训练和推理明显更快

关键配置（`config.py`）：
- `TFIDF_MAX_FEATURES`: 每个文本块 TF-IDF 的词表上限（向量化前）
- `TFIDF_ENABLE_SVD`: 是否启用 SVD 降维
- `TFIDF_SVD_DIM_PER_BLOCK`: 每个文本块降维后的目标维度（默认 64）

经验建议：
- 维度太大、速度慢：降低 `TFIDF_SVD_DIM_PER_BLOCK`（如 32/48）
- 精度下降明显：提高到 64/96
- 推荐先固定 64，观察 AUC/PR-AUC 后再调

### Logistic Regression 加速（GPU 优先）

- 训练器会优先尝试 `cuML`（GPU）逻辑回归
- 若环境缺少 RAPIDS 或 GPU 不可用，会自动回退到 sklearn CPU（`saga`）

说明：
- 不需要改命令，默认自动选择
- 若想强制 CPU，可在 `config.py` 把 `LR_PREFER_GPU=False`

## Baseline 模型 (14 个)

### Dense 系列 (渐进式)

| 编号 | 模型名称 | 特征组成 | 用途 |
|------|----------|----------|------|
| A | `A_Dense_LR` | Dense (A~H+J 组) | 结构化特征基线 |
| B | `B_Dense_AF_LR` | Dense + AF (Action+Feedback) | +基础文本 |
| C | `C_Dense_AF_Thought_LR` | Dense + AF + Thought | **+thought 贡献** |
| D | `D_Dense_Full_LR` | Dense + AF + Thought + AC | **主模型** (完整特征) |

### TF-IDF 系列 (渐进式)

| 编号 | 模型名称 | 特征组成 | 用途 |
|------|----------|----------|------|
| E | `E_TfIdf_AF_LR` | TF-IDF AF (5 路) | 纯文本基线 |
| F | `F_TfIdf_AF_Thought_LR` | TF-IDF AF + Thought (7 路) | +thought 文本 |
| G | `G_TfIdf_Full_LR` | TF-IDF Full (9 路) | 纯文本上界 |

### 非线性模型

| 编号 | 模型名称 | 特征组成 | 用途 |
|------|----------|----------|------|
| H | `H_LightGBM_Dense` | Dense (A~H+J 组) | 非线性上界 |
| I | `I_LightGBM_Dense_AF` | Dense + AF (Action+Feedback) | 非线性 + 基础文本 |
| J | `J_LightGBM_Dense_AF_Thought` | Dense + AF + Thought | 非线性 + thought |
| K | `K_LightGBM_Dense_Full` | Dense + AF + Thought + AC | 非线性完整主模型 |
| L | `L_LightGBM_TfIdf_AF` | TF-IDF AF (5 路) | 纯文本非线性基线 |
| M | `M_LightGBM_TfIdf_AF_Thought` | TF-IDF AF + Thought (7 路) | 纯文本非线性 + thought |
| N | `N_LightGBM_TfIdf_Full` | TF-IDF Full (9 路) | 纯文本非线性上界 |

## Ablation 实验

### 渐进式特征组 Ablation (4 个)

从简单到复杂，观察每增加一组特征带来的性能提升：

| 编号 | 模型名称 | 特征组成 | 对应 Baseline | 保存文件 |
|------|----------|----------|---------------|----------|
| Abl 1 | `Abl_DenseOnly_LR` | Dense only | Baseline A | `ablation_dense_only.pkl` |
| Abl 2 | `Abl_NoThoughtContent_LR` | Dense + AF | Baseline B | `ablation_dense_action_feedback.pkl` |
| Abl 3 | `Abl_NoAssistantContent_LR` | Dense + AF + Thought | Baseline C | `ablation_dense_action_feedback_thought.pkl` |
| Abl 4 | `Abl_Base_LR` | Dense + AF + Thought (参考基底) | Baseline C | 复用 `C_Dense_AF_Thought_LR` |

### 单组份 Ablation (6 个，从 Dense + AF + Thought 基底出发)

**重要：** Ablation 5~10 覆盖你关心的三个核心对照：去 task prompt、去 model_id、process-only。

| 编号 | 模型名称 | 移除内容 | 基底 | 研究问题 |
|------|----------|----------|------|----------|
| Abl 5 | `Abl_NoTaskPrompt_LR` | 去掉 task prompt | Dense + AF + Thought | 任务描述是否重要？ |
| Abl 6 | `Abl_NoFeedback_LR` | 去掉 feedback | Dense + AF + Thought | **环境反馈最关键？** |
| Abl 7 | `Abl_NoAction_LR` | 去掉 action | Dense + AF + Thought | agent 行动是否重要？ |
| Abl 8 | `Abl_NoThought_LR` | 去掉 thought | Dense + AF + Thought | **思考过程是否重要？** ⭐ |
| Abl 9 | `Abl_NoModel_LR` | 去掉 model_id | Dense + AF + Thought | 模型身份是否有影响？ |
| Abl 10 | `Abl_ProcessOnly_LR` | 去掉 task prompt + 去掉 model_id | Dense(no model_id) + action/feedback/thought(prefix+last) | **只凭过程是否可预测？** |

### 关键假设验证

通过这些 ablation，可以验证：

1. **Thought 的价值** - 对比 `Abl_Base_LR` (有 thought) vs `Abl_NoThought_LR` (无 thought)
2. **渐进提升** - 对比 Abl 1 → Abl 2 → Abl 3 → Abl 4 的 AUC 提升
3. **各文本组份的相对重要性** - 对比 Abl 5~8 的 AUC 下降幅度
4. **Assistant Content 的额外价值** - 对比 Baseline C vs Baseline D
5. **任务先验影响** - 对比 Baseline C vs Abl 5
6. **模型身份影响** - 对比 Baseline C vs Abl 9
7. **纯过程可预测性** - 对比 Baseline C vs Abl 10

## 评估指标

- ROC-AUC
- PR-AUC
- LogLoss
- Brier Score
- 按 prefix step 分桶表现
- 校准曲线
- 概率分布图
- 双端阈值决策精确率（成功端/失败端）
- 不同阈值下的样本占比与提前结束省步数统计

## 输出报告如何解读

`reports/evaluation_report.txt` 新增了阈值决策与提前结束统计，核心看三类信息：

1. 模型质量（整体）
- `ROC-AUC` / `PR-AUC`：越高越好
- `LogLoss` / `Brier`：越低越好

2. 阈值策略质量（双端）
- `Prec(S)`：判为成功端（高置信成功）后的精确率，越高越稳
- `Prec(F)`：判为失败端（高置信失败）后的精确率，越高越稳
- `Decide%`：该阈值下能提前做出决策的样本比例

3. 提前结束收益（效率）
- `AvgSave(dec)`：仅在已决策样本中，平均可省步数
- `AvgSave(all)`：按全样本折算的平均省步数，更适合评估全局收益

常见权衡：
- 阈值越高：`Prec(S/F)` 常提升，但 `Decide%` 会下降
- 阈值越低：覆盖更多样本，但误判风险可能上升

新增对照汇总：
- `Key Counterfactual Experiments` 区域会直接给出：
  - 去 task prompt 的 AUC 变化
  - 去 model_id 的 AUC 变化
  - process-only 相对主模型的 AUC 变化
- `Mixed-Model Implementation Checks` 区域会给出：
  - Dense+sparse 拼接后的维度/稀疏度
  - 列数是否匹配（防止拼接错位）
  - LR 迭代次数（收敛检查）
  - 是否做了 dense 标准化（当前默认 False）

## 关键设计决策

1. **只用 `tool` split**：保留原生 tool_calls 结构，适合精确 step 重建
2. **test 先于 run_python 判断**：避免 `python -m pytest` 被误分类
3. **按 trajectory 分组切分**：`group_id=traj_id`，并在切分前做全局 `instance_id` 去重
4. **Sample weight**：每条轨迹总权重归一，避免长轨迹偏倚
5. **禁止未来信息**：所有特征仅来自当前 prefix 可见内容

## 数据完整性校验（重要）

主流程会在切分前强制检查：
- `(group_id, prefix_step_idx)` 组合必须唯一
- 每个 `group_id` 只能对应一个 `traj_id`

若你之前生成过旧版 `prefix_table.parquet`（group_id=instance_id），会直接报错。  
修复方式：重新生成 step/prefix 表，再训练评估：

```bash
cd machine/swe_prefix_predict7
python run_all.py --data-dir /path/to/tool-parquet-files
```

如果你确认已有中间表是新版（group_id=traj_id），才使用：

```bash
python run_all.py --skip-step-table --skip-prefix-table
```
