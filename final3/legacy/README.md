# legacy

`legacy/` 只放旧入口迁移帮助和薄 wrapper。这里的目标是帮助从 final2 或旧 answer-aware 脚本迁移到 final3，不是继续发展新功能。

## 当前文件

- `wrappers.py`: 维护旧脚本名到 final3 命令的映射。

## 常用命令

```bash
python -m final3.cli legacy explain safe_stop_dual_head_retrain.py
```

如果某个旧脚本没有映射，命令会列出当前已知入口。

## 新迁移项怎么加

只在确实需要保留兼容说明时新增映射。不要把旧脚本完整复制进这里；优先把稳定逻辑迁移到 `core/`、`policies/`、`models/` 或 `reports/`。
