# configs

`configs/` 放 final3 的可审计配置。配置文件负责描述策略、实验 registry 和路径模板；代码应读取配置，而不是把机器路径和阈值散落在脚本里。

## 文件说明

- `policy_presets.yaml`: safe-stop 策略 preset。当前主 preset 是 `current_safe_stop`。
- `paths.yaml`: 本机当前使用的路径配置；默认数据根目录是 `../data`。
- `paths.example.yaml`: 路径模板，和 `paths.yaml` 字段一致。
- `experiment_registry.yaml`: 按实验集组织的主实验、paper、benchmark、architecture baseline 清单。
- `rq_final.yaml`: 最终论文 RQ 实验集的主配置。它定义 SWEVerify / Toolathlon / TerminalBench 的 prefix parquet、leave-one-test-model split、SWEVerify-only 消融、模型比较和 latency 范围。

## 常用方式

应用默认策略：

```bash
python -m final3.cli policy apply \
  --preset current_safe_stop \
  --predictions examples/smoke_predictions.csv \
  --output-dir outputs/current_safe_stop_smoke
```

查看重型实验清单：

```bash
python -m final3.cli train list-heavy
```

实验前检查：

```bash
python -m final3.cli check preflight --experiment all
```

最终 RQ smoke：

```bash
python -m final3.cli experiment rq-final --stage smoke
```

## 新配置规则

新增配置时先写清楚字段含义和默认行为。路径模板不要直接保存敏感信息或机器专属绝对路径；如果必须使用本地绝对路径，优先放到本机私有配置，不要作为通用模板。
