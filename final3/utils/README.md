# utils

`utils/` 放非常轻量、无业务语义的通用工具。当前只有日志 helper。

## 当前文件

- `logging.py`: 提供 `get_logger(name)`，统一基础日志格式。

## 使用方式

```python
from final3.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("message")
```

## 新代码放置规则

只有真正跨模块、无业务归属的小工具才放这里。不要把 policy、模型、数据转换或报告逻辑塞进 `utils/`。
