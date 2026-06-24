# reports

`reports/` contains paper-table and diagnostic-report entrypoints. Report
commands should be dry-run or explicit about writing outputs.

## Paper Tables

`paper_tables.py` refreshes the current paper-facing CSV tables through the
ICSE draft reporting code.

```bash
python -m earlyeval.cli report paper-tables
```
