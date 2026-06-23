"""模型定义、训练与导出。

为避免在只用到轻量子模块（如 ``dataset``）时被动导入 torch，
这里采用 PEP 562 懒加载：访问到具体名字时才导入对应子模块。
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "WakeWordModel",
    "train_classifier",
    "export_onnx",
    "export_tensorflow",
    "prepare_data",
    "fit",
    "run_training",
]

_LAZY = {
    "WakeWordModel": ("model", "WakeWordModel"),
    "train_classifier": ("trainer", "train_classifier"),
    "export_onnx": ("export", "export_onnx"),
    "export_tensorflow": ("export", "export_tensorflow"),
    "prepare_data": ("pipeline", "prepare_data"),
    "fit": ("pipeline", "fit"),
    "run_training": ("pipeline", "run_training"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module, attr = _LAZY[name]
        mod = importlib.import_module(f".{module}", __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
