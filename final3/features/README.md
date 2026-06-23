# features

`features/` 放特征系统的接口和迁移桥接。当前 final3 已经带有
`final3/vendor/prefix_predict_model_holdout_answer/` 的 answer-aware 代码快照；
默认 wrapper 调用这份 vendored 实现，旧目录只作为复现历史 run 的 fallback。

## 当前文件

- `answer.py`: 提供 `enrich_with_gold_answers(prefix_frame, verified_jsonl)`，调用 final3 vendored answer-aware 模块里的 `answer_features`。

## 使用方式

```python
from final3.features.answer import enrich_with_gold_answers

enriched = enrich_with_gold_answers(prefix_frame, "verified_answers.jsonl")
```

这个函数要求 final3 vendored answer-aware 模块路径存在。路径由
`final3/core/legacy_paths.py` 统一解析；如果 vendored copy 不存在，才回退到旧目录。

## 新代码放置规则

如果要迁移 dense、TF-IDF、gold-answer、text-pair 或 embedding cache 特征，先在这里建立清晰接口，再考虑是否真正复制旧实现。不要把大 cache 或 parquet 复制进 `features/`。
