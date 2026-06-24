# Prefix Outcome Prediction / Early-Stop Paper Strategy

description:description `model_holdout_answer_calibrated_full` description,description"description"description,description/description.

## 1. description:description

description:

> SWE-bench agent outcome description prefix prediction description task difficulty prior;description,description agent capability group.description single-head final-success probability description,description base-rate shift.

description.description:

1. description coding-agent early-stop prediction description.
2. description task prior description process evidence description.
3. description naive early stop description capability shift description rate distortion.
4. description:`min_step / consecutive / symmetric or asymmetric thresholds / valid-selected policy`.
5. description:process-evidence residual,safe-stop dual-head,model-capability calibration.

## 2. description

### 2.1 description,description prior description

description consolidated report:

- `I/J` final-step ROC-AUC description `0.900`.
- `NoTaskSignal` / `NoTaskPromptTFIDF` description AUC,description task prompt TF-IDF description.
- `NoGoldAnswer` description `0.900` description `0.8271`,description structured gold answer description.
- `NoTask+NoGold` final AUC description `0.8054`,description step0 AUC description `0.5000`,description.

### 2.2 step description

description evidence:

- `NoTask+NoGold`: step0 `0.5000` → final `0.8054`.
- description `I/J`: step0 description `0.889/0.885`,final description `0.900`.

description:

- description AUC description task / gold / difficulty prior.
- description task+gold description,description 0.5 description 0.8,description process evidence description.

### 2.3 other-model prior baseline description

description other-model prior AUC:

| Setting | Model step0 | Model last | Train mean-success prior | Last - Prior |
|---|---:|---:|---:|---:|
| bottom3 | `0.842` | `0.862` | `0.845` | `+0.018` |
| mid3 | `0.900` | `0.915` | `0.907` | `+0.008` |
| top3 | `0.950` | `0.949` | `0.959` | `-0.010` |

description:

- `all_correct` description,description.
- `mean_other_success` description,description task difficulty oracle.
- bottom/mid description;top3 description task prior description.

description:

> Process evidence helps, but its marginal value is largest for weaker/mid agents and smallest for very strong agents whose success is already mostly determined by instance difficulty.

### 2.4 early-stop description,description

description:

- naive single-head threshold description step0 description,description prior description.
- `min_step=10, consecutive=2` description rate distortion.
- `strong_reg + raw + asymmetric s=0.90/f=0.20 + min10/k2`: mean abs drop `2.34pp`, save `37.2%`, acc `91.8%`.
- symmetric description `strong_reg/raw thr=0.90/0.10`: mean abs drop `1.95pp`, save `30.6%`, acc `93.3%`.
- description `thr=0.95/0.05`: mean abs drop `1.15pp`, save `17.7%`, acc `94.6%`.

description:

> Early stopping is possible, but there is a safety-savings frontier. Policies that ignore process maturity over-save and distort resolve rate; policies gated by process evidence preserve rate better at the cost of lower savings.

## 3. description

description:

> Can we accurately predict final success probability?

description reviewer description:task prior,leakage-like same-instance prior,model shift.

description:

> When can we safely predict coding-agent outcomes from partial trajectories, and how much of that signal comes from task difficulty versus process evidence?

description:

> description?description,description?

## 4. description RQ description

### RQ1: Outcome predictability under model holdout

description:description heldout agent model description,prefix outcome prediction description AUC?description step description?

description:

- `I/J` final AUC description `0.900`.
- step-bucket curves.
- top/mid/bottom split curves.

description:

- step0 / step10 / final AUC.
- CI.
- raw vs calibrated description AUC,description AUC description calibration description.

### RQ2: Task prior vs process evidence

description:description,description?

description:

- `NoTask+NoGold`: 0.5 → 0.8054.
- other-model `mean_success` prior AUC description `0.903~0.905`.
- process gain: bottom3 `+0.018`, mid3 `+0.008`, top3 `-0.010`.

description.

description:

- hard subset: description `train_other_mean_success ∈ [0.3,0.7]` description `[0.4,0.6]` description.
- residual ranking: description prior description success/failure.
- ablation: NoTask+NoGold description action distribution.

### RQ3: Safe early-stop policy frontier

description:description token/step,description resolve rate?

description:

- symmetric threshold + 95% CI.
- min_step/consecutive rescue.
- decision action analysis.

description:

- safe conservative: `strong_reg/raw + symmetric 0.95/0.05 + min10/k2`.
- balanced: `strong_reg/raw + symmetric 0.90/0.10 + min10/k2`.
- aggressive diagnostic: `asymmetric 0.90/0.20 + min10/k2`.

description frontier,description.

### RQ4: Failure modes under capability shift

description:description?

description:

- bottom3: predicted/adjusted rate description,FP-success description.
- top3: predicted/adjusted rate description,FN-failure description.
- top/mid/bottom description base rates description.

description:

> The same task prior maps to different success probabilities for agents with different capabilities. A single global calibration cannot simultaneously fit weak and strong heldout models.

description future work:model-conditional calibration.

## 5. description

### Must-have 1: Hard-subset AUC

description:description task prior description.

description:

- description `train_other_mean_success` description:
  - easy: `[0.8,1.0]`
  - medium/hard-ambiguous: `[0.3,0.7]`
  - hardest: `[0.0,0.3]`
- description:model step0,step10,last AUC.
- description medium bucket description last AUC description step0/prior,description.

### Must-have 2: Residual / within-prior-bin analysis

description:description oracle prior description AUC,description"description,prefix process description".

description:

- description `train_other_mean_success` description 5 description 10 description quantile.
- description quantile description model AUC.
- description train description prior description baseline,description model score residual description AUC/description.

### Must-have 3: Instance-holdout sanity check

description:description reviewer description same-instance prior description.

description:

- description instance-holdout,description.
- description `I/J` description `NoTask+NoGold`.
- description,description:description instance description,AUC description,description process-only signal description.

description,description limitation,description model-holdout description.

### Must-have 4: Calibration / capability intercept

description:description top/bottom description.

description:

- description heldout model description calibration trajectories,description 20/50/100 description.
- description intercept:`logit(p') = logit(p) + b_model`.
- description Drop description `±3~5pp` description.

description:description,description.

## 6. description

### 6.1 description global threshold

description:top3/bottom3 description,global threshold description.

### 6.2 description calibrated probability description

description calibration description capability shift description.description,description safe-stop frontier description diagnostic decomposition.

### 6.3 description `all_correct` description baseline

`all_correct` description,description task prior description.`mean_other_success` description baseline.

### 6.4 description "AUC=0.9 description"

description other-model prior,description same-instance difficulty.

## 7. description/description

### description

1. **When Can Coding Agents Be Stopped Early? Disentangling Task Difficulty and Process Evidence in Prefix Outcome Prediction**
2. **Safe Early Stopping for Software-Engineering Agents under Model Holdout**
3. **Predicting Agent Success from Partial Trajectories: Task Priors, Process Evidence, and Capability Shift**

### Contributions

description:

1. We introduce a model-holdout benchmark for predicting SWE-bench agent outcomes from partial trajectories.
2. We quantify the relative contribution of task difficulty priors and dynamic process evidence via ablations and other-model baselines.
3. We show that naive final-success predictors induce systematic resolve-rate distortion under agent capability shift.
4. We propose and evaluate safe early-stop policies using delayed-start, consecutive evidence, and threshold frontiers, achieving meaningful step savings with bounded rate distortion.

## 8. description Figure / Table

### Figure 1: Task setup

- Trajectory prefix → predictor → success/failure / early stop.
- model-holdout split.

### Figure 2: Step AUC curves

- I/J vs NoTask+NoGold.
- description NoTask+NoGold description 0.5 description 0.805.

### Figure 3: Other-model prior baseline

- model step0 / model last / train mean-success prior.
- description top/mid/bottom.

### Figure 4: Incremental gain over prior

- bottom3 +0.018, mid3 +0.008, top3 -0.010.
- description.

### Figure 5: Safe-stop frontier

- x=Save%, y=Drop or Mean Abs Drop.
- different thresholds with CI.

### Figure 6: Failure mode under capability shift

- top3: FN-failure;bottom3: FP-success.
- per-agent Drop bar chart.

### Table 1: Main AUC + ablations

- I/J, NoTask, NoGold, NoTask+NoGold.

### Table 2: Early-stop policies

- symmetric 0.90/0.10, 0.95/0.05, asymmetric 0.90/0.20, with Acc/Save/Drop/CI.

## 9. description

description:

> We first show that prefix-based outcome prediction can reach high AUC under model holdout. However, a strong other-model baseline reveals that much of this predictability comes from instance-level task difficulty. Through task/gold ablations and process-only models, we show that dynamic process evidence still contributes, especially for weaker and mid-level agents. We then demonstrate that directly thresholding final-success probability is unsafe under capability shift: weak agents are overestimated and strong agents underestimated. Finally, we evaluate safe early-stop policies that delay decisions until sufficient process evidence accumulates, trading some savings for substantially improved resolve-rate stability.

description:

> description"description",description coding-agent description.description AUC description,description:description,description;description,description.

## 10. description

description,description:

1. description.
2. description hard-subset AUC.
3. description within-prior-bin / residual analysis.
4. description small calibration intercept description.
5. description,description instance-holdout sanity check.

description LightGBM description.

## 11. Top high-capability group description

description high-capability heldout group description "top stable agents",description `top3`.description `gpt-5-2-codex` description/description,description,description:

- `20251118_mini-v1.15.0_gemini-3-pro-preview-20251118`
- `20251124_mini-v1.16.0_claude-opus-4-5-20251101`

description codex5.2 description symmetric threshold,CI,other-model prior description hard-subset AUC.description representative error cases description minstep-threshold diagnostic description codex5.2,description.
