# policies

`policies/` 是当前 safe-stop 主线的核心模块。它负责读取策略 preset，对 prefix 级 prediction 表执行 success/failure 双头决策，并产出轨迹级决策和汇总指标。

## 当前文件

- `safe_stop.py`: dual-head safe-stop 决策、全局汇总、per-agent 汇总。
- `presets.py`: 读取 `configs/policy_presets.yaml`，并提供 `current_safe_stop` fallback。
- `apply.py`: 从文件读取 prediction 表，应用 policy，写出 CSV 和 metadata。

## 输入 prediction 表

当前主策略要求至少包含：

- `traj_id`
- `label`
- `prefix_step_idx`
- `orig_model_id` 或 `model_id`
- `prob_cal_safe_success__I_LightGBM_Dense_AF`
- `prob_cal_safe_failure__I_LightGBM_Dense_AF`

## 常用命令

```bash
python -m final3.cli policy apply \
  --preset current_safe_stop \
  --predictions examples/smoke_predictions.csv \
  --output-dir outputs/current_safe_stop_smoke
```

输出：

- `policy_decisions.csv`: 每条轨迹的首次 safe-stop 决策。
- `policy_summary.csv`: 全局 coverage、accuracy、step saving、resolve rate change。
- `policy_per_agent.csv`: 按 agent/model 的同类指标。
- `run_metadata.json`: preset、输入路径、输出路径。

## 新策略怎么加

先在 `configs/policy_presets.yaml` 增加 preset。如果需要新决策逻辑，再在 `safe_stop.py` 增加独立函数，并确保 `apply_policy` 的输入输出契约仍然清晰。
