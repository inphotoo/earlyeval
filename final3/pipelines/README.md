# pipelines

`pipelines/` 放多个模块组合后的稳定流程。pipeline 是给用户运行的工作流，不是底层算法实现区。

## 当前 pipeline

`current_safe_stop.py` 提供 `run_current_safe_stop`：

- `mode=smoke`: 使用 `examples/smoke_predictions.csv`，输出到 `outputs/current_safe_stop_smoke/`。
- `mode=main`: 需要显式传入 `--predictions`，默认输出到 `outputs/current_safe_stop_main/`。
- `mode=full`: 当前故意不接重型实验，会报错提示先看 registry。

## 常用命令

```bash
python -m final3.cli pipeline current-safe-stop --mode smoke
```

主策略：

```bash
python -m final3.cli pipeline current-safe-stop \
  --mode main \
  --predictions /path/to/test_predictions_safe_stop.parquet
```

## 新 pipeline 怎么加

新增流程时，应在这里调用 `core/`、`policies/`、`models/`、`reports/` 等模块的业务函数，再到 `final3/cli.py` 暴露命令。不要把复杂逻辑直接写进 CLI。
