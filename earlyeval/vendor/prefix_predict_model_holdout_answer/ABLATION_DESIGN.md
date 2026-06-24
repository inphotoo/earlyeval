# Ablation description

## 📋 description

description,description,description **thought** description **assistant_content** description.

---

## 🎯 description

### description 1:description (Incremental Ablation)

**description:** description,description.

description `run_all.py` Phase 6 description Ablation 1-4:

```
description:
  Abl 1: Dense (A~H+J)
    ↓ +action+feedback(description thought / assistant_content)
  Abl 2: Dense + action + feedback
    ↓ +thought(description assistant_content)
  Abl 3: Dense + action + feedback + thought
    ↓ description Baseline C (Dense + AF + Thought) description
  Abl 4: Abl_Base_LR = Baseline C description
```

| description | description | description | description | description |
|----------|----------|----------|--------------|----------|
| Abl 1 | `Abl_DenseOnly_LR` | Dense (A~H+J description) | description Dense,description TF-IDF (`ablation_dense_only.pkl`) | description alone description? |
| Abl 2 | `Abl_NoThoughtContent_LR` | Dense + AF (5 description TF-IDF:task/prefix/last description action+feedback) | description Dense+Full description thought description assistant_content TF-IDF (`ablation_dense_action_feedback.pkl`) | action+feedback description? |
| Abl 3 | `Abl_NoAssistantContent_LR` | Dense + AF + Thought (7 description TF-IDF:AF + thought) | description Dense+Full description assistant_content TF-IDF (`ablation_dense_action_feedback_thought.pkl`) | **thought description?** |
| Abl 4 | `Abl_Base_LR` | Dense + AF + Thought (description) | description Baseline C `C_Dense_AF_Thought_LR` description | description |

**description:**
- ✅ Thought description (description:description)
- ✅ Assistant_content description (description thought/action description)

---

### description 2:description (Leave-One-Out Ablation)

**description:** description Dense + AF + Thought description,description,description.

**description:** `Abl_Base_LR` (description Baseline C:Dense + AF + Thought,description assistant_content)

description `run_all.py` Phase 6 description Ablation 5-10 description:

| description | description | description | description | description | description |
|----------|----------|------------|----------|--------------|----------|
| Abl 5 | `Abl_NoTaskPrompt_LR` | task prompt TF-IDF | Dense + AF + Thought | description `X_dense_af_thought` description `tfidf_task_prompt` description | description? |
| Abl 6 | `Abl_NoFeedback_LR` | feedback TF-IDF (prefix/last) | Dense + AF + Thought | description `X_dense_af_thought` description `tfidf_prefix_feedback` description `tfidf_last_feedback` description | **description?** |
| Abl 7 | `Abl_NoAction_LR` | action TF-IDF (prefix/last) | Dense + AF + Thought | description `X_dense_af_thought` description `tfidf_prefix_action` description `tfidf_last_action` description | agent description? |
| Abl 8 | `Abl_NoThought_LR` | thought TF-IDF (prefix/last) | Dense + AF + Thought | description `X_dense_af_thought` description `tfidf_prefix_thought` description `tfidf_last_thought` description | **description?** |
| Abl 9 | `Abl_NoModel_LR` | model_id (dense one-hot) | Dense + AF + Thought | description `FeatureEngineer(include_model_id=False)` description Dense + AF + Thought (`ablation_dense_af_thought_no_model.pkl`) | description? |
| Abl 10 | `Abl_ProcessOnly_LR` | task prompt + model_id | Dense(no model_id) + process text | description dense(hand-crafted) + prefix/last description action,feedback,thought;description task prompt description model_id | **description?** |

**description:**
- ✅ Feedback description (description:description/description)
- ✅ Thought description (description:description,description feedback description)
- ✅ Model_id description (description:description Claude/GPT-4 description)

---

## 📊 description

### description

description:

1. **description Ablation description**
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

2. **Leave-One-Out description**
   ```
   AUC
   0.85 | ■ Full (0.842)
        |
   0.80 | ◆ NoThought (0.815)  ← thought description ~2.7%
        | ▲ NoAction (0.798)   ← action description ~4.4%
   0.75 | ▼ NoFeedback (0.765) ← feedback description ~7.7%
        |
   0.70 +---------------------------
   ```

### description:description

```
description Thought description AUC description > 5%:
  → Thought description!
  → description:description thought description
  → description"agent description"description

description Thought description AUC description < 1%:
  → Thought description
  → description:thought description action/feedback description
  → description:description thought description

description Assistant_content description AUC description > 3%:
  → Assistant_content description
  → description:description,description

description Model_id description AUC description > 5%:
  → description
  → description:description
```

---

## 🔬 description

### 1. Thought description

description,description thought description:

```python
# description thought description TF-IDF description/description
thought_features = [name for name in feature_names if 'thought' in name]
top_positive = sorted(zip(thought_features, coefficients), key=lambda x: -x[1])[:20]
top_negative = sorted(zip(thought_features, coefficients), key=lambda x: x[1])[:20]

# description:
# description:"fix", "understand", "analyze", "debug", "test"
# description:"try", "maybe", "guess", "random", "hope"
```

### 2. Thought description

```python
# description prefix_thought_chars description label description
import matplotlib.pyplot as plt

successful = prefix_df[prefix_df['label'] == 1]['prefix_thought_chars']
failed = prefix_df[prefix_df['label'] == 0]['prefix_thought_chars']

plt.boxplot([successful, failed], labels=['Successful', 'Failed'])
plt.ylabel('Thought characters')
plt.title('Thought length vs Success')
```

**description:** description thought (description)

### 3. Thought-Action description

```python
# description thought_action_overlap_avg description
# description = thought description
# description = thought description/description

# description:description (description,description)
```

---

## 📈 description

### description

- ✅ **description AUC > 0.85** (description dense only description > 15%)
- ✅ **Thought description AUC description > 2%** (description)
- ✅ **description ablation description** (p < 0.01)

### description

- ✅ description (description)
- ✅ description step bucket description
- ✅ description (description)

---

## 🧪 description

```bash
# description ablation
cd /workspace/data/liuzijun/research/swebench/machine/swe_prefix_predict7
python run_all.py --data-dir ../../SWE-smith-trajectories/data

# description ablation description
cat reports/evaluation_report.txt | grep "Abl_"

# description ablation description
python scripts/plot_ablation_comparison.py
```

---

## 📝 description

description ablation description,description:

1. **description SWE agent description**
   - description thought description
   
2. **description**
   - Feedback > Action > Thought > Task (description)
   - description

3. **description**
   - description agent description t description
   - description,description

4. **description**
   - description processed prefix description
   - description baseline description

---

## 📅 description

| description | description | description |
|------|------|----------|
| 1 | description ablation description | 2-3 description |
| 2 | description,description | 1-2 description |
| 3 | description thought description | 2-4 description |
| 4 | description/description | 1-2 description |

---

## 🎓 description

1. **RQ1:** Agent description (thought) description?
   - description:Ablation 8 (description thought)

2. **RQ2:** Assistant content description thought + action description?
   - description:description Ablation 3 vs Ablation 4

3. **RQ3:** description?
   - description:description thought TF-IDF description

4. **RQ4:** description"description"(description,description)description?
   - description:description J description dense description

5. **RQ5:** description (Claude/GPT-4) description?
   - description:description model_id description

6. **RQ6:** description,description?
   - description:description Baseline C description Abl 10 (`Abl_ProcessOnly_LR`)

---

**description:** 2026-03-10  
**description:** v2.0 (description thought/content description)
