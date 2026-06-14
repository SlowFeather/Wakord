"""实时唤醒词监听服务。

组成：
    audio     —— 麦克风采集（sounddevice）
    vad       —— 轻量人声检测，用于功耗门控
    detector  —— VAD 门控 + openWakeWord 推理 + 冷却去重
    server    —— 可被外部程序控制的常驻服务（TCP / JSON-lines）
    client    —— 控制客户端
    protocol  —— 控制协议的消息常量与编解码
"""

from .detector import WakeWordDetector, DetectionEvent
from .server import WakeWordService
from .client import ServiceClient

__all__ = [
    "WakeWordDetector",
    "DetectionEvent",
    "WakeWordService",
    "ServiceClient",
]
