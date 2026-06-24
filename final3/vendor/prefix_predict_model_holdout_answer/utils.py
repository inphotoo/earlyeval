'Public-release English note.'
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import config


def get_logger(name: str) -> logging.Logger:
    'Public-release English note.'
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        # Public-release English note.
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(ch)
        # Public-release English note.
        fh = logging.FileHandler(config.LOG_DIR / f"{name}.log", mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(fh)
    return logger


def rebind_all_file_loggers():
    'Public-release English note.'
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
    'Public-release English note.'
    logger.info(f"[START] {task_name}")
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    logger.info(f"[DONE]  {task_name}  ({elapsed:.2f}s)")


def save_json(obj, path: Path):
    'Public-release English note.'
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path):
    'Public-release English note.'
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
