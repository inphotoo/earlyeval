# Ablation 实验设计文档

## 📋 实验目标

通过系统的消融实验，量化不同特征组对预测性能的贡献，特别是验证 **thought** 和 **assistant_content** 等新引入特征的价值。

---

## 🎯 实验设计

### 实验 1：渐进式特征添加 (Incremental Ablation)

**目的：** 从简单到复杂，观察每增加一组特征带来的性能提升。

对应 `run_all.py` Phase 6 中的 Ablation 1–4：

```
实验流程：
  Abl 1: Dense (A~H+J)
    ↓ +action+feedback（去掉 thought / assistant_content）
  Abl 2: Dense + action + feedback
    ↓ +thought（仍不含 assistant_content）
  Abl 3: Dense + action + feedback + thought
    ↓ 复用 Baseline C (Dense + AF + Thought) 作为参考基底
  Abl 4: Abl_Base_LR = Baseline C 结果
```

| 实验编号 | 模型名称 | 特征组成 | 对应代码实现 | 验证问题 |
|----------|----------|----------|--------------|----------|
| Abl 1 | `Abl_DenseOnly_LR` | Dense (A~H+J 组) | 直接用 Dense，无 TF-IDF (`ablation_dense_only.pkl`) | 结构化特征 alone 能到什么程度？ |
| Abl 2 | `Abl_NoThoughtContent_LR` | Dense + AF (5 路 TF-IDF：task/prefix/last 的 action+feedback) | 在 Dense+Full 上移除 thought 与 assistant_content TF-IDF (`ablation_dense_action_feedback.pkl`) | action+feedback 文本是否有帮助？ |
| Abl 3 | `Abl_NoAssistantContent_LR` | Dense + AF + Thought (7 路 TF-IDF：AF + thought) | 在 Dense+Full 上仅移除 assistant_content TF-IDF (`ablation_dense_action_feedback_thought.pkl`) | **thought 是否有额外价值？** |
| Abl 4 | `Abl_Base_LR` | Dense + AF + Thought (参考基底) | 直接复用 Baseline C `C_Dense_AF_Thought_LR` 的模型与预测 | 作为后续单组份消融的基线 |

**关键假设：**
- ✅ Thought 应该带来显著提升 (假设：思考质量预示成功)
- ✅ Assistant_content 的提升可能较小 (因为与 thought/action 有重叠)

---

### 实验 2：单组份移除 (Leave-One-Out Ablation)

**目的：** 在 Dense + AF + Thought 基底上，逐一移除每个组份，观察性能下降。

**基线模型：** `Abl_Base_LR` (即 Baseline C：Dense + AF + Thought，不含 assistant_content)

与 `run_all.py` Phase 6 中的 Ablation 5–10 一一对应：

| 实验编号 | 模型名称 | 移除的组份 | 基底特征 | 对应代码实现 | 验证问题 |
|----------|----------|------------|----------|--------------|----------|
| Abl 5 | `Abl_NoTaskPrompt_LR` | task prompt TF-IDF | Dense + AF + Thought | 在 `X_dense_af_thought` 上移除 `tfidf_task_prompt` 所在列块 | 任务描述重要吗？ |
| Abl 6 | `Abl_NoFeedback_LR` | feedback TF-IDF (prefix/last) | Dense + AF + Thought | 在 `X_dense_af_thought` 上移除 `tfidf_prefix_feedback` 和 `tfidf_last_feedback` 列块 | **环境反馈最关键？** |
| Abl 7 | `Abl_NoAction_LR` | action TF-IDF (prefix/last) | Dense + AF + Thought | 在 `X_dense_af_thought` 上移除 `tfidf_prefix_action` 和 `tfidf_last_action` 列块 | agent 行动模式重要吗？ |
| Abl 8 | `Abl_NoThought_LR` | thought TF-IDF (prefix/last) | Dense + AF + Thought | 在 `X_dense_af_thought` 上移除 `tfidf_prefix_thought` 和 `tfidf_last_thought` 列块 | **思考文本有独特价值？** |
| Abl 9 | `Abl_NoModel_LR` | model_id (dense one-hot) | Dense + AF + Thought | 使用 `FeatureEngineer(include_model_id=False)` 重新构建 Dense + AF + Thought (`ablation_dense_af_thought_no_model.pkl`) | 模型身份是否有影响？ |
| Abl 10 | `Abl_ProcessOnly_LR` | task prompt + model_id | Dense(no model_id) + process text | 保留 dense(hand-crafted) + prefix/last 的 action、feedback、thought；移除 task prompt 与 model_id | **只凭过程能否预测成功率？** |

**关键假设：**
- ✅ Feedback 移除应该导致最大下降 (假设：反馈包含最直接的成功/失败信号)
- ✅ Thought 移除应该有中等下降 (假设：思考提供独特信息，但不如 feedback 直接)
- ✅ Model_id 移除可能有显著下降 (假设：不同模型如 Claude/GPT-4 能力差异明显)

---

## 📊 预期结果分析

### 结果可视化

将生成以下对比图：

1. **渐进式 Ablation 对比图**
   ```
   AUC
   0.85 |         ■ Abl_Full (0.842)
        |      ▲  Abl_NoAssistantContent (0.825)
   0.80 |   ◆     Abl_NoThoughtContent (0.798)
        | ■       Abl_DenseOnly (0.721)
   0.75 |
        +----------------------------------
          Dense  +A+F  +T   +AC
   ```

2. **Leave-One-Out 对比图**
   ```
   AUC
   0.85 | ■ Full (0.842)
        |
   0.80 | ◆ NoThought (0.815)  ← thought 贡献 ~2.7%
        | ▲ NoAction (0.798)   ← action 贡献 ~4.4%
   0.75 | ▼ NoFeedback (0.765) ← feedback 贡献 ~7.7%
        |
   0.70 +---------------------------
   ```

### 决策树：根据结果决定下一步

```
如果 Thought 移除导致 AUC 下降 > 5%:
  → Thought 非常重要！
  → 建议：进一步分析哪些 thought 模式最有价值
  → 可以发表关于"agent 思考质量量化"的论文

如果 Thought 移除导致 AUC 下降 < 1%:
  → Thought 没有独特价值
  → 可能原因：thought 信息已被 action/feedback 包含
  → 建议：考虑去掉 thought 以简化模型

如果 Assistant_content 移除导致 AUC 下降 > 3%:
  → Assistant_content 包含独特信息
  → 建议：保留，并进一步分析其组成

如果 Model_id 移除导致 AUC 下降 > 5%:
  → 不同模型能力差异显著
  → 建议：深入分析哪些模型在哪些任务上表现更好
```

---

## 🔬 深入分析方向

### 1. Thought 特征的重要性排名

训练完整模型后，检查 thought 相关特征的系数：

```python
# 查看 thought 相关 TF-IDF 特征的最高正/负系数
thought_features = [name for name in feature_names if 'thought' in name]
top_positive = sorted(zip(thought_features, coefficients), key=lambda x: -x[1])[:20]
top_negative = sorted(zip(thought_features, coefficients), key=lambda x: x[1])[:20]

# 预期发现：
# 正相关词汇："fix", "understand", "analyze", "debug", "test"
# 负相关词汇："try", "maybe", "guess", "random", "hope"
```

### 2. Thought 长度与成功率的关系

```python
# 分析 prefix_thought_chars 特征与 label 的相关性
import matplotlib.pyplot as plt

successful = prefix_df[prefix_df['label'] == 1]['prefix_thought_chars']
failed = prefix_df[prefix_df['label'] == 0]['prefix_thought_chars']

plt.boxplot([successful, failed], labels=['Successful', 'Failed'])
plt.ylabel('Thought characters')
plt.title('Thought length vs Success')
```

**预期：** 成功的轨迹有更长的 thought (更深入思考)

### 3. Thought-Action 重叠率分析

```python
# 分析 thought_action_overlap_avg 与成功率的关系
# 高重叠率 = thought 聚焦于具体行动
# 低重叠率 = thought 更抽象/战略性

# 假设：中等重叠率最好 (既有战略思考，又聚焦行动)
```

---

## 📈 成功标准

### 主要指标

- ✅ **完整模型 AUC > 0.85** (相比 dense only 提升 > 15%)
- ✅ **Thought 移除导致 AUC 下降 > 2%** (证明思考的价值)
- ✅ **所有 ablation 实验都有统计学意义** (p < 0.01)

### 次要指标

- ✅ 校准曲线良好 (预测概率与真实频率一致)
- ✅ 在不同 step bucket 中表现稳定
- ✅ 特征重要性可解释 (符合直觉)

---

## 🧪 运行命令

```bash
# 完整运行所有 ablation
cd /workspace/data/liuzijun/research/swebench/machine/swe_prefix_predict7
python run_all.py --data-dir ../../SWE-smith-trajectories/data

# 查看 ablation 结果
cat reports/evaluation_report.txt | grep "Abl_"

# 生成 ablation 对比图
python scripts/plot_ablation_comparison.py
```

---

## 📝 预期论文贡献

基于 ablation 实验结果，可以贡献：

1. **首个 SWE agent 思考质量量化研究**
   - 证明 thought 文本包含预测成功的独特信息
   
2. **特征重要性层次结构**
   - Feedback > Action > Thought > Task (假设)
   - 为未来研究提供优先级指导

3. **实用的早期预测工具**
   - 在 agent 执行到第 t 步时预测成功率
   - 支持早期终止低成功率轨迹，节省计算资源

4. **开源数据集和基线**
   - 发布 processed prefix 数据集
   - 提供可复现的 baseline 代码

---

## 📅 时间规划

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 运行完整 ablation 实验 | 2-3 小时 |
| 2 | 分析结果，生成图表 | 1-2 小时 |
| 3 | 深入分析 thought 特征 | 2-4 小时 |
| 4 | 撰写技术报告/论文 | 1-2 天 |

---

## 🎓 关键研究问题

1. **RQ1:** Agent 的思考过程 (thought) 是否包含预测成功的独特信息？
   - 验证方法：Ablation 8 (去掉 thought)

2. **RQ2:** Assistant content 是否比单独的 thought + action 提供更多信息？
   - 验证方法：对比 Ablation 3 vs Ablation 4

3. **RQ3:** 哪些类型的思考模式最预示成功？
   - 验证方法：分析 thought TF-IDF 特征的重要性

4. **RQ4:** 思考的"深度"(长度、密度)是否与成功率正相关？
   - 验证方法：分析 J 组 dense 特征的相关性

5. **RQ5:** 不同模型 (Claude/GPT-4) 的思考模式是否有显著差异？
   - 验证方法：对比 model_id 特征的系数

6. **RQ6:** 去除任务先验与模型身份后，仅过程状态是否仍具备较强可预测性？
   - 验证方法：对比 Baseline C 与 Abl 10 (`Abl_ProcessOnly_LR`)

---

**最后更新:** 2026-03-10  
**版本:** v2.0 (包含 thought/content 特征)
