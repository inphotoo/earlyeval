# models

`models/` 放模型训练入口、模型 wrapper 和重型实验声明。这里的默认原则是：轻量命令可以 dry-run，重型训练必须显式 `--execute` 或专门 opt-in。

## 当前文件

- `dual_head_lightgbm.py`: 构建当前主策略对应的 LightGBM dual-head 训练命令。默认使用 final3 vendored trainer，只返回命令，不执行。
- `heavy.py`: 列出 BERT/CodeBERT、Qwen LoRA、LLM-logit judge、full ablation 等重型实验名称。

## 常用命令

查看 dual-head 训练命令：

```bash
python -m final3.cli train dual-head
```

真正启动训练：

```bash
python -m final3.cli train dual-head --execute
```

查看重型实验清单：

```bash
python -m final3.cli train list-heavy
```

## 新模型怎么放

新增模型时先提供一个可 dry-run 的命令构建函数，明确输入数据、输出目录、默认参数和是否重型。不要在 import 时加载模型权重或启动训练。
