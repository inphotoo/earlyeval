# core

`core/` 是 final3 的底层稳定层，放所有上层模块共享的数据契约和轻量公共能力。这里不启动训练，不写论文报告，也不处理某个具体 pipeline 的业务分支。

## 文件说明

- `contracts.py`: 定义 `TrajectoryRecord`、`PrefixRecord`、`PolicySpec`、`ExperimentRun`。
- `io.py`: 统一读写 CSV、TSV、JSON、JSONL、parquet，并创建输出目录。
- `splits.py`: 检查 instance split 和 model holdout 是否泄漏。
- `prefix.py`: 桥接 final3 vendored answer-aware prefix/step builder。
- `legacy_paths.py`: 集中解析 final3 vendored 模块、legacy 目录和共享数据目录。

## 常用方式

读取 prediction 表：

```python
from final3.core.io import read_table

frame = read_table("outputs/current_safe_stop_smoke/policy_decisions.csv")
```

检查 split 泄漏：

```python
from final3.core.splits import validate_model_holdout

report = validate_model_holdout(frame)
assert report.ok
```

## 新代码放这里的条件

只有当代码会被多个上层模块复用，并且不依赖具体实验、报告或模型时，才放进 `core/`。如果逻辑只属于 safe-stop 策略，放 `policies/`；如果只属于 benchmark adapter，放 `benchmarks/`。
