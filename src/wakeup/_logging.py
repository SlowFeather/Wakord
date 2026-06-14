"""统一的日志配置。各模块用 ``get_logger(__name__)`` 获取 logger。"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> None:
    """初始化根 logger。可通过环境变量 ``WAKEUP_LOG_LEVEL`` 覆盖级别。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("WAKEUP_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
