# reports

`reports/` 放论文表格、RQ、诊断报告等入口。报告入口默认应该是 dry-run 或显式执行，避免无意启动长任务。

## 当前入口

`paper_tables.py` 刷新当前 ICSE draft 的 CSV 表格，底层调用 `paper/icse_submission_draft/data/refresh_tables.py`。

```bash
python -m final3.cli report paper-tables
```

## 新报告怎么加

新增报告应明确输入路径、输出目录和是否会调用历史重型脚本。默认不要在 dry-run 之外写大量产物。
