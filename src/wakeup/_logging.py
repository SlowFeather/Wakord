"""统一的日志配置。各模块用 ``get_logger(__name__)`` 获取 logger。"""

from __future__ import annotations

import logging
import os
import sys

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
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
