# evaluation

`evaluation/` 放可复用的评估、指标和审计逻辑。它服务 policy、pipeline 和 report，但不直接负责命令行入口。

## 当前文件

- `metrics.py`: 轻量指标工具，例如 `safe_div`。
- `leakage.py`: 对 DataFrame 做 split 和 model-holdout 泄漏审计。

## 使用方式

```python
from final3.evaluation.leakage import audit_split_frame

audit = audit_split_frame(frame)
if not audit["ok"]:
    print(audit)
```

## 新代码放置规则

如果逻辑是指标、校准诊断、ranking、per-agent shift 或防泄漏审计，放这里。如果逻辑会改变 safe-stop 决策本身，放 `policies/`。
