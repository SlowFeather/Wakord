"""WakeUp — 本地中文语音唤醒词训练与常驻监听服务（唤醒词：小元）。

子包：
    wakeup.data      —— 数据准备（中文 TTS 正样本、负样本特征、特征提取）
    wakeup.training  —— 模型定义、训练、导出
    wakeup.service   —— 实时监听服务（VAD 门控 + 唤醒词检测 + WebSocket 控制接口）
"""

from .config import Config, load_config

__version__ = "0.1.0"

__all__ = ["Config", "load_config", "__version__"]
