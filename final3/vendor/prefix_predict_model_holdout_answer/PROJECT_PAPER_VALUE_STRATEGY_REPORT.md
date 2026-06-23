# Prefix Outcome Prediction / Early-Stop Paper Strategy

生成目的：把当前 `model_holdout_answer_calibrated_full` 项目的论文价值重新定位，避免继续陷入“为什么概率不完美”的焦虑，把结果组织成可发表/可毕业的研究叙事。

## 1. 先定性：这个项目不是失败

当前结果最重要的信息是：

> SWE-bench agent outcome 的 prefix prediction 很大一部分来自 task difficulty prior；过程信息确实有增益，但增益大小取决于 agent capability group。直接把 single-head final-success probability 当作跨模型绝对概率，会产生系统性 base-rate shift。

这不是坏结果。它可以转成论文贡献：

1. 系统评估 coding-agent early-stop prediction 的可行性。
2. 定量分解 task prior 和 process evidence 的贡献。
3. 证明 naive early stop 会因为 capability shift 造成 rate distortion。
4. 提出更安全的后处理策略：`min_step / consecutive / symmetric or asymmetric thresholds / valid-selected policy`。
5. 给出下一代方法方向：process-evidence residual、safe-stop dual-head、model-capability calibration。

## 2. 当前硬证据

### 2.1 主模型能预测，但静态 prior 很强

来自 consolidated report：

- `I/J` final-step ROC-AUC 约 `0.900`。
- `NoTaskSignal` / `NoTaskPromptTFIDF` 几乎不伤 AUC，说明 task prompt TF-IDF 不是唯一核心。
- `NoGoldAnswer` 从约 `0.900` 降到 `0.8271`，说明 structured gold answer 很关键。
- `NoTask+NoGold` final AUC 仍有 `0.8054`，且 step0 AUC 是 `0.5000`，说明纯过程信号确实存在。

### 2.2 step 曲线证明过程信息存在

关键 evidence：

- `NoTask+NoGold`: step0 `0.5000` → final `0.8054`。
- 主模型 `I/J`: step0 已经约 `0.889/0.885`，final 约 `0.900`。

解释：

- 主模型的大部分 AUC 来自 task / gold / difficulty prior。
- 但去掉 task+gold 后，过程信息仍能从 0.5 升到 0.8，这就是论文里的 process evidence 价值。

### 2.3 other-model prior baseline 是关键诊断

已跑出的 other-model prior AUC：

| Setting | Model step0 | Model last | Train mean-success prior | Last - Prior |
|---|---:|---:|---:|---:|
| bottom3 | `0.842` | `0.862` | `0.845` | `+0.018` |
| mid3 | `0.900` | `0.915` | `0.907` | `+0.008` |
| top3 | `0.950` | `0.949` | `0.959` | `-0.010` |

结论：

- `all_correct` 不强，因为太稀疏。
- `mean_other_success` 很强，几乎就是 task difficulty oracle。
- bottom/mid 仍有过程增益；top3 基本被 task prior 解释掉。

这正好构成论文的中心发现：

> Process evidence helps, but its marginal value is largest for weaker/mid agents and smallest for very strong agents whose success is already mostly determined by instance difficulty.

### 2.4 early-stop 可行，但必须安全化

当前安全策略结果：

- naive single-head threshold 会大量 step0 截断，容易被静态 prior 带偏。
- `min_step=10, consecutive=2` 明显降低 rate distortion。
- `strong_reg + raw + asymmetric s=0.90/f=0.20 + min10/k2`: mean abs drop `2.34pp`, save `37.2%`, acc `91.8%`。
- symmetric 旧口径 `strong_reg/raw thr=0.90/0.10`: mean abs drop `1.95pp`, save `30.6%`, acc `93.3%`。
- 更保守 `thr=0.95/0.05`: mean abs drop `1.15pp`, save `17.7%`, acc `94.6%`。

论文里应该说：

> Early stopping is possible, but there is a safety–savings frontier. Policies that ignore process maturity over-save and distort resolve rate; policies gated by process evidence preserve rate better at the cost of lower savings.

## 3. 推荐论文主问题

不要把论文主问题写成：

> Can we accurately predict final success probability?

这个太容易被 reviewer 打：task prior、leakage-like same-instance prior、model shift。

建议改成：

> When can we safely predict coding-agent outcomes from partial trajectories, and how much of that signal comes from task difficulty versus process evidence?

中文版本：

> 编程智能体的部分轨迹能否支持安全早停？这种可预测性究竟来自题目难度先验，还是来自智能体执行过程中的动态证据？

## 4. 推荐 RQ 结构

### RQ1: Outcome predictability under model holdout

问题：在 heldout agent model 上，prefix outcome prediction 能达到什么 AUC？随 step 如何变化？

已有结果：

- `I/J` final AUC 约 `0.900`。
- step-bucket curves。
- top/mid/bottom split curves。

需要呈现：

- step0 / step10 / final AUC。
- CI。
- raw vs calibrated 不作为主 AUC，因为 AUC 不受 calibration 单调变换影响。

### RQ2: Task prior vs process evidence

问题：模型到底学的是题目，还是过程？

已有结果：

- `NoTask+NoGold`: 0.5 → 0.8054。
- other-model `mean_success` prior AUC 约 `0.903~0.905`。
- process gain: bottom3 `+0.018`, mid3 `+0.008`, top3 `-0.010`。

这是论文最有价值的发现之一。

建议补强：

- hard subset: 只看 `train_other_mean_success ∈ [0.3,0.7]` 或 `[0.4,0.6]` 的题。
- residual ranking: 看模型是否能在固定 prior 桶内区分 success/failure。
- ablation: NoTask+NoGold 的 action distribution。

### RQ3: Safe early-stop policy frontier

问题：能不能省 token/step，同时不破坏 resolve rate？

已有结果：

- symmetric threshold + 95% CI。
- min_step/consecutive rescue。
- decision action analysis。

推荐主策略：

- safe conservative: `strong_reg/raw + symmetric 0.95/0.05 + min10/k2`。
- balanced: `strong_reg/raw + symmetric 0.90/0.10 + min10/k2`。
- aggressive diagnostic: `asymmetric 0.90/0.20 + min10/k2`。

论文要强调 frontier，不要只报一个阈值。

### RQ4: Failure modes under capability shift

问题：为什么跨模型会偏？

已有结果：

- bottom3: predicted/adjusted rate 偏高，FP-success 多。
- top3: predicted/adjusted rate 偏低，FN-failure 多。
- top/mid/bottom 的 base rates 差异极大。

推荐表述：

> The same task prior maps to different success probabilities for agents with different capabilities. A single global calibration cannot simultaneously fit weak and strong heldout models.

这能自然引出 future work：model-conditional calibration。

## 5. 最应该补的实验

### Must-have 1: Hard-subset AUC

目的：证明过程信息不是完全被 task prior 覆盖。

做法：

- 用 `train_other_mean_success` 分桶：
  - easy: `[0.8,1.0]`
  - medium/hard-ambiguous: `[0.3,0.7]`
  - hardest: `[0.0,0.3]`
- 每个桶内算：model step0、step10、last AUC。
- 如果 medium bucket 中 last AUC 明显高于 step0/prior，就是最强证据。

### Must-have 2: Residual / within-prior-bin analysis

目的：不要和 oracle prior 比绝对 AUC，而是问“在同等题目难度下，prefix process 能不能排序”。

做法：

- 按 `train_other_mean_success` 分 5 或 10 个 quantile。
- 每个 quantile 内算 model AUC。
- 或者 train 一个只用 prior 的 baseline，再看 model score residual 是否有 AUC/相关性。

### Must-have 3: Instance-holdout sanity check

目的：防 reviewer 说 same-instance prior 太强。

做法：

- 取一小版 instance-holdout，不需要全量深度模型都跑满。
- 只跑 `I/J` 或 `NoTask+NoGold`。
- 目标不是追求最高性能，而是证明：在完全新 instance 上，AUC 会下降，但 process-only signal 仍有一定预测力。

如果时间不够，至少写成 limitation，并把 model-holdout 说清楚。

### Must-have 4: Calibration / capability intercept

目的：解决你现在最焦虑的 top/bottom 偏移。

做法：

- 给每个 heldout model 少量 calibration trajectories，比如 20/50/100 条。
- 学一个 intercept：`logit(p') = logit(p) + b_model`。
- 看 Drop 是否从 `±3~5pp` 收敛。

这可以变成非常漂亮的实用方案：新模型上线后只需少量校准样本，就能安全早停。

## 6. 不建议继续深挖的方向

### 6.1 不要继续无止境调 global threshold

原因：top3/bottom3 方向相反，global threshold 无法同时解决。

### 6.2 不要把 calibrated probability 当最终卖点

当前 calibration 在 capability shift 下不稳定。可以作为辅助，但主卖点应是 safe-stop frontier 和 diagnostic decomposition。

### 6.3 不要把 `all_correct` 作为最强 baseline

`all_correct` 太稀疏，不代表 task prior 的真实强度。`mean_other_success` 才是强 baseline。

### 6.4 不要只说 “AUC=0.9 很高”

必须同时报告 other-model prior，否则会被质疑只是 same-instance difficulty。

## 7. 推荐论文标题/贡献写法

### 标题方向

1. **When Can Coding Agents Be Stopped Early? Disentangling Task Difficulty and Process Evidence in Prefix Outcome Prediction**
2. **Safe Early Stopping for Software-Engineering Agents under Model Holdout**
3. **Predicting Agent Success from Partial Trajectories: Task Priors, Process Evidence, and Capability Shift**

### Contributions

可以写四条：

1. We introduce a model-holdout benchmark for predicting SWE-bench agent outcomes from partial trajectories.
2. We quantify the relative contribution of task difficulty priors and dynamic process evidence via ablations and other-model baselines.
3. We show that naive final-success predictors induce systematic resolve-rate distortion under agent capability shift.
4. We propose and evaluate safe early-stop policies using delayed-start, consecutive evidence, and threshold frontiers, achieving meaningful step savings with bounded rate distortion.

## 8. 推荐 Figure / Table

### Figure 1: Task setup

- Trajectory prefix → predictor → success/failure / early stop。
- model-holdout split。

### Figure 2: Step AUC curves

- I/J vs NoTask+NoGold。
- 显示 NoTask+NoGold 从 0.5 升到 0.805。

### Figure 3: Other-model prior baseline

- model step0 / model last / train mean-success prior。
- 按 top/mid/bottom。

### Figure 4: Incremental gain over prior

- bottom3 +0.018, mid3 +0.008, top3 -0.010。
- 这是最诚实也最有洞察的图。

### Figure 5: Safe-stop frontier

- x=Save%, y=Drop or Mean Abs Drop。
- different thresholds with CI。

### Figure 6: Failure mode under capability shift

- top3: FN-failure；bottom3: FP-success。
- per-agent Drop bar chart。

### Table 1: Main AUC + ablations

- I/J, NoTask, NoGold, NoTask+NoGold。

### Table 2: Early-stop policies

- symmetric 0.90/0.10, 0.95/0.05, asymmetric 0.90/0.20, with Acc/Save/Drop/CI。

## 9. 最终叙事版本

建议你这样讲：

> We first show that prefix-based outcome prediction can reach high AUC under model holdout. However, a strong other-model baseline reveals that much of this predictability comes from instance-level task difficulty. Through task/gold ablations and process-only models, we show that dynamic process evidence still contributes, especially for weaker and mid-level agents. We then demonstrate that directly thresholding final-success probability is unsafe under capability shift: weak agents are overestimated and strong agents underestimated. Finally, we evaluate safe early-stop policies that delay decisions until sufficient process evidence accumulates, trading some savings for substantially improved resolve-rate stability.

中文理解：

> 我不是简单做了一个“成功概率预测器”，而是在研究 coding-agent 早停里什么时候能相信预测器。这个问题的核心不是 AUC 越高越好，而是要拆清楚：多少来自题目先验，多少来自过程证据；以及跨模型能力变化时，如何避免错误早停。

## 10. 最小可交付路线

如果时间紧，优先顺序：

1. 整理现有结果成论文图表。
2. 补 hard-subset AUC。
3. 补 within-prior-bin / residual analysis。
4. 补 small calibration intercept 实验。
5. 如果还有时间，再跑 instance-holdout sanity check。

这条路线比继续调 LightGBM 或全局阈值更能提升论文价值。

## 11. Top high-capability group 口径修正

论文主分析建议把 high-capability heldout group 写成 “top stable agents”，而不是严格 `top3`。当前 `gpt-5-2-codex` 在数据/触发行为上有异常，主表应排除它，只保留：

- `20251118_mini-v1.15.0_gemini-3-pro-preview-20251118`
- `20251124_mini-v1.16.0_claude-opus-4-5-20251101`

已经排除 codex5.2 的主报告包括 symmetric threshold、CI、other-model prior 和 hard-subset AUC。旧的 representative error cases 和 minstep-threshold diagnostic 仍可能包含 codex5.2，只用于诊断或过滤后使用。
