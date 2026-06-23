# final3 Python 包

这里是 `SweBench_Organized_Package_final3` 的源码区。长期维护的 Python 代码都应该放在这个包下面，根目录只放文档、配置、manifest、示例、输出和测试。

## 入口

顶层 CLI 在 `cli.py`：

```bash
python -m final3.cli --help
python -m final3.cli pipeline current-safe-stop --mode smoke
```

CLI 只负责参数解析和分发。业务逻辑应写在子模块里，再由 CLI 调用。

## 子目录

```text
core/        数据契约、IO、split 检查、旧路径解析
checks/      实验前 preflight 检查
benchmarks/ benchmark adapter 和质量审计
features/   特征接口和旧 answer-aware 特征桥接
models/     训练入口、重型实验声明、模型 wrapper
policies/   safe-stop 策略、preset、prediction 表应用
evaluation/ 指标和泄漏审计
reports/    paper RQ 和诊断报告入口
experiments/按实验集合组织的实现：paper builder、paper bundle、registry
pipelines/  组合流程
legacy/     旧入口迁移说明
utils/      轻量通用工具
```

## 新代码规则

不要把新业务逻辑直接写进 `cli.py`。先在合适子目录中提供可测试函数，再把命令行参数映射到该函数。这样单元测试可以直接调用业务函数，不必模拟命令行。

不要在源码模块里硬编码大文件路径。需要历史产物时，优先通过 `core/paths.py`、`configs/paths.yaml`、`configs/experiment_registry.yaml` 或 `paper/data/input_manifest.csv` 说明来源。
