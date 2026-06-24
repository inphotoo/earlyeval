# SWE-smith Trajectory Prefix Success Prediction

description `t` description,description,description `resolved=True`.

## description

```
swe_prefix_predict/
├── run_all.py               # description(description)
├── config.py                # description(description,description,GPU description)
├── utils.py                 # description,description
├── action_classifier.py     # description(taxonomy description)
├── observation_parser.py    # Observation description
├── step_builder.py          # Step description(messages → step_table)
├── prefix_builder.py        # Prefix description(step → prefix_table + description A~H)
├── feature_engineer.py      # description(dense + TF-IDF)
├── data_split.py            # description group_id description
├── trainer.py               # description(LR + LightGBM)
├── evaluator.py             # description,description,description
├── action_taxonomy.yaml     # description
├── feature_dictionary.md    # description
├── requirements.txt         # Python description
├── data/                    # description
│   ├── step_table.parquet
│   └── prefix_table.parquet
├── models/                  # description
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
├── reports/                 # description
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
│   └── precision_savings_curve_<model>.png  # Prec(S)/Prec(F) description
└── logs/                    # description
    ├── run_all.log
    ├── step_builder.log
    ├── prefix_builder.log
    ├── feature_engineer.log
    ├── trainer.log
    └── evaluator.log
```

## description

- Python >= 3.10
- GPU: NVIDIA A100-80G(1 description),LightGBM GPU description

## description

```bash
pip install -r requirements.txt
```

## description

### 1) description(description)

```bash
cd machine/swe_prefix_predict7
pip install -r requirements.txt

# description 1:description
export SWE_PARQUET_DIR=/path/to/tool-parquet-files
python run_all.py

# description 2:description
python run_all.py --data-dir /path/to/tool-parquet-files
```

### 2) description(description)

```bash
cd machine/swe_prefix_predict7

# description parquet,description
python run_all.py --data-dir /path/to/tool-parquet-files --quick-verify

# description parquet description
python run_all.py --single-parquet /path/to/tool-xxx.parquet --quick-verify
```

`--quick-verify` description:
- description parquet description
- description LightGBM(`--skip-lgbm`)
- description ablation(`--skip-ablation`)

description"description,description,description".

### 3) description

```bash
# GPU description,LightGBM description CPU
python run_all.py --data-dir /path/to/data --no-gpu-lgbm

# description LightGBM
python run_all.py --data-dir /path/to/data --skip-lgbm

# description ablation
python run_all.py --data-dir /path/to/data --skip-ablation

# description,description
python run_all.py --skip-step-table --skip-prefix-table
```

### 4) description(description)

```bash
# 1. description step table
python step_builder.py

# 2. description prefix table
python prefix_builder.py

# 3. description(description)
python run_all.py --skip-step-table --skip-prefix-table
```

## description TF-IDF description

### TF-IDF description(description)

description"description TF-IDF,description SVD description,description":
- description(task/action/feedback/thought/content)description
- description
- description,description

description(`config.py`):
- `TFIDF_MAX_FEATURES`: description TF-IDF description(description)
- `TFIDF_ENABLE_SVD`: description SVD description
- `TFIDF_SVD_DIM_PER_BLOCK`: description(description 64)

description:
- description,description:description `TFIDF_SVD_DIM_PER_BLOCK`(description 32/48)
- description:description 64/96
- description 64,description AUC/PR-AUC description

### Logistic Regression description(GPU description)

- description `cuML`(GPU)description
- description RAPIDS description GPU description,description sklearn CPU(`saga`)

description:
- description,description
- description CPU,description `config.py` description `LR_PREFER_GPU=False`

## Baseline description (14 description)

### Dense description (description)

| description | description | description | description |
|------|----------|----------|------|
| A | `A_Dense_LR` | Dense (A~H+J description) | description |
| B | `B_Dense_AF_LR` | Dense + AF (Action+Feedback) | +description |
| C | `C_Dense_AF_Thought_LR` | Dense + AF + Thought | **+thought description** |
| D | `D_Dense_Full_LR` | Dense + AF + Thought + AC | **description** (description) |

### TF-IDF description (description)

| description | description | description | description |
|------|----------|----------|------|
| E | `E_TfIdf_AF_LR` | TF-IDF AF (5 description) | description |
| F | `F_TfIdf_AF_Thought_LR` | TF-IDF AF + Thought (7 description) | +thought description |
| G | `G_TfIdf_Full_LR` | TF-IDF Full (9 description) | description |

### description

| description | description | description | description |
|------|----------|----------|------|
| H | `H_LightGBM_Dense` | Dense (A~H+J description) | description |
| I | `I_LightGBM_Dense_AF` | Dense + AF (Action+Feedback) | description + description |
| J | `J_LightGBM_Dense_AF_Thought` | Dense + AF + Thought | description + thought |
| K | `K_LightGBM_Dense_Full` | Dense + AF + Thought + AC | description |
| L | `L_LightGBM_TfIdf_AF` | TF-IDF AF (5 description) | description |
| M | `M_LightGBM_TfIdf_AF_Thought` | TF-IDF AF + Thought (7 description) | description + thought |
| N | `N_LightGBM_TfIdf_Full` | TF-IDF Full (9 description) | description |

## Ablation description

### description Ablation (4 description)

description,description:

| description | description | description | description Baseline | description |
|------|----------|----------|---------------|----------|
| Abl 1 | `Abl_DenseOnly_LR` | Dense only | Baseline A | `ablation_dense_only.pkl` |
| Abl 2 | `Abl_NoThoughtContent_LR` | Dense + AF | Baseline B | `ablation_dense_action_feedback.pkl` |
| Abl 3 | `Abl_NoAssistantContent_LR` | Dense + AF + Thought | Baseline C | `ablation_dense_action_feedback_thought.pkl` |
| Abl 4 | `Abl_Base_LR` | Dense + AF + Thought (description) | Baseline C | description `C_Dense_AF_Thought_LR` |

### description Ablation (6 description,description Dense + AF + Thought description)

**description:** Ablation 5~10 description:description task prompt,description model_id,process-only.

| description | description | description | description | description |
|------|----------|----------|------|----------|
| Abl 5 | `Abl_NoTaskPrompt_LR` | description task prompt | Dense + AF + Thought | description? |
| Abl 6 | `Abl_NoFeedback_LR` | description feedback | Dense + AF + Thought | **description?** |
| Abl 7 | `Abl_NoAction_LR` | description action | Dense + AF + Thought | agent description? |
| Abl 8 | `Abl_NoThought_LR` | description thought | Dense + AF + Thought | **description?** ⭐ |
| Abl 9 | `Abl_NoModel_LR` | description model_id | Dense + AF + Thought | description? |
| Abl 10 | `Abl_ProcessOnly_LR` | description task prompt + description model_id | Dense(no model_id) + action/feedback/thought(prefix+last) | **description?** |

### description

description ablation,description:

1. **Thought description** - description `Abl_Base_LR` (description thought) vs `Abl_NoThought_LR` (description thought)
2. **description** - description Abl 1 → Abl 2 → Abl 3 → Abl 4 description AUC description
3. **description** - description Abl 5~8 description AUC description
4. **Assistant Content description** - description Baseline C vs Baseline D
5. **description** - description Baseline C vs Abl 5
6. **description** - description Baseline C vs Abl 9
7. **description** - description Baseline C vs Abl 10

## description

- ROC-AUC
- PR-AUC
- LogLoss
- Brier Score
- description prefix step description
- description
- description
- description(description/description)
- description

## description

`reports/evaluation_report.txt` description,description:

1. description(description)
- `ROC-AUC` / `PR-AUC`:description
- `LogLoss` / `Brier`:description

2. description(description)
- `Prec(S)`:description(description)description,description
- `Prec(F)`:description(description)description,description
- `Decide%`:description

3. description(description)
- `AvgSave(dec)`:description,description
- `AvgSave(all)`:description,description

description:
- description:`Prec(S/F)` description,description `Decide%` description
- description:description,description

description:
- `Key Counterfactual Experiments` description:
  - description task prompt description AUC description
  - description model_id description AUC description
  - process-only description AUC description
- `Mixed-Model Implementation Checks` description:
  - Dense+sparse description/description
  - description(description)
  - LR description(description)
  - description dense description(description False)

## description

1. **description `tool` split**:description tool_calls description,description step description
2. **test description run_python description**:description `python -m pytest` description
3. **description trajectory description**:`group_id=traj_id`,description `instance_id` description
4. **Sample weight**:description,description
5. **description**:description prefix description

## description(description)

description:
- `(group_id, prefix_step_idx)` description
- description `group_id` description `traj_id`

description `prefix_table.parquet`(group_id=instance_id),description.  
description:description step/prefix description,description:

```bash
cd machine/swe_prefix_predict7
python run_all.py --data-dir /path/to/tool-parquet-files
```

description(group_id=traj_id),description:

```bash
python run_all.py --skip-step-table --skip-prefix-table
```
