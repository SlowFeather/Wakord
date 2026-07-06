"""统一的日志配置。各模块用 ``get_logger(__name__)`` 获取 logger。

格式与 ChatCaht 全家（ChatCaht/SpText/GVoice/LoLLama）统一：
``2026-07-06 10:09:29,554 INFO wakeup.service.server: 消息``
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_CONFIGURED = False


def _force_utf8_streams() -> None:
    """把 stdout/stderr 切到 UTF-8，避免 GBK 控制台遇到中文/emoji 直接崩溃。

    Windows 默认控制台编码是 GBK(cp936)，第三方库（如 torch.onnx 导出器）打印
    ✅ 这类字符时会抛 UnicodeEncodeError 中断流程；中文日志也会显示成乱码。
    用 ``errors="replace"`` 兜底，无法编码的字符降级而非报错。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def setup_logging(level: str | int | None = None) -> None:
    """初始化根 logger。可通过环境变量 ``WAKEUP_LOG_LEVEL`` 覆盖级别。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    _force_utf8_streams()

    if level is None:
        level = os.environ.get("WAKEUP_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
    )
    _CONFIGURED = True


def add_file_logging(path: str | Path, *, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> None:
    """给根 logger 追加一个滚动文件日志（UTF-8）。

    只在常驻服务（serve）里调用，训练/导出等一次性命令保持纯控制台输出。
    """
    setup_logging()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
