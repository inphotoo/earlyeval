# utils

`utils/` contains small cross-cutting helpers with no benchmark or policy
ownership.

## Logging

```python
from final3.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("message")
```

Keep policy, model, benchmark, and reporting logic in their owning modules.
