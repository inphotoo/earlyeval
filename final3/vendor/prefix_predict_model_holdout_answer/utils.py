"""
公共工具：日志、计时器、IO 辅助。
"""
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import config


def get_logger(name: str) -> logging.Logger:
    """获取带有统一格式的 logger。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        # 控制台
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(ch)
        # 文件
        fh = logging.FileHandler(config.LOG_DIR / f"{name}.log", mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(fh)
    return logger


def rebind_all_file_loggers():
    """
    将当前进程内所有 logger 的 FileHandler 重新绑定到 config.LOG_DIR。
    适用于运行时切换 run-name 后的日志目录重定向。
    """
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger_names = [name for name in logging.root.manager.loggerDict.keys()]
    for name in logger_names:
        logger = logging.getLogger(name)
        if not logger.handlers:
            continue

        has_file_handler = False
        kept_handlers = []
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                has_file_handler = True
                try:
                    h.close()
                except Exception:
                    pass
            else:
                kept_handlers.append(h)

        if not has_file_handler:
            continue

        logger.handlers = kept_handlers
        fh = logging.FileHandler(log_dir / f"{name}.log", mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(fh)


@contextmanager
def timer(logger: logging.Logger, task_name: str):
    """带日志的计时上下文。"""
    logger.info(f"[START] {task_name}")
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    logger.info(f"[DONE]  {task_name}  ({elapsed:.2f}s)")


def save_json(obj, path: Path):
    """保存 JSON 到文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path):
    """从文件加载 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
