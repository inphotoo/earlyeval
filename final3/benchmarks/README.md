# benchmarks

`benchmarks/` 负责把不同 benchmark 的原始记录转换为 final3 的统一 trajectory contract。它的边界是数据归一化和质量审计，不训练模型，不调阈值，不写论文结论。

## 当前入口

`normalize.py` 支持：

- `terminalbench`
- `swebench`
- `toolathlon`
- `generic`

命令：

```bash
python -m final3.cli data normalize \
  --benchmark terminalbench \
  --input examples/tiny_terminalbench.jsonl \
  --output-dir outputs/tiny_terminalbench_normalized
```

输出：

- `normalized_trajectories.jsonl`: 统一字段的 trajectory 记录。
- `quality_audit.csv`: 轨迹数、instance 数、model 数、resolved rate、空消息数。

## 输出契约

归一化后的记录应至少包含：

- `benchmark`
- `instance_id`
- `traj_id`
- `model_id`
- `resolved`
- `messages`
- `patch`

## 新 adapter 怎么加

新增 benchmark 时，先在 `normalize_record` 里建立字段映射；如果逻辑变复杂，再拆出专门函数。adapter 不应该直接依赖模型训练、policy 阈值或论文报告。
