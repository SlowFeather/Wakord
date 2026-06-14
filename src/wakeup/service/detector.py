"""唤醒词检测器：把「VAD 门控 + openWakeWord 推理 + 冷却去重」封装成
一个 ``process(frame) -> Optional[DetectionEvent]`` 的状态机。

功耗优化逻辑：
* 静默时不调用唤醒词模型（只跑极廉价的 VAD）。
* 人声开始的瞬间，回灌一段「前导帧」给模型补足上下文（唤醒词很短，
  否则模型缓冲还没填满词就说完了）。
* 人声结束后用 hangover 多跑几帧，避免把词尾切掉。
* 命中后进入冷却期，防止一次唤醒重复上报。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._logging import get_logger
from ..config import Config
from .vad import VAD

logger = get_logger(__name__)


@dataclass
class DetectionEvent:
    model: str
    score: float
    ts: float


class WakeWordDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        svc = cfg.service
        self.threshold = svc.threshold
        self.cooldown = svc.cooldown_seconds
        self.hangover_frames = svc.hangover_frames

        self.vad = VAD(cfg)
        self.oww = self._load_model(cfg)
        # openWakeWord 用模型文件名（不含扩展名）作为输出键
        keys = list(self.oww.models.keys())
        self.model_name = svc.model_name if svc.model_name in keys else keys[0]
        logger.info("唤醒词模型已加载，输出键: %s", self.model_name)

        self._ring: deque[np.ndarray] = deque(maxlen=max(1, svc.preroll_frames))
        self._active = False
        self._hangover_left = 0
        self._last_trigger = 0.0
        # 最近一帧的预测分（活跃时更新），供前台调阈值时观察
        self.last_active_score = 0.0

    @staticmethod
    def _load_model(cfg: Config):
        import openwakeword
        from openwakeword.model import Model

        model_path = cfg.fs.model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到唤醒词模型: {model_path}\n请先运行 `wakeup train` 完成训练。"
            )

        # 首次使用时下载 openWakeWord 的特征提取模型（melspectrogram + embedding）
        try:
            openwakeword.utils.download_models()
        except Exception as exc:
            logger.debug("download_models 跳过/失败: %s", exc)

        logger.info("加载唤醒词模型: %s", model_path)
        return Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    def reset(self) -> None:
        self._ring.clear()
        self._active = False
        self._hangover_left = 0
        if hasattr(self.oww, "reset"):
            self.oww.reset()

    def _predict(self, frame: np.ndarray) -> float:
        scores = self.oww.predict(frame)
        return float(scores.get(self.model_name, 0.0))

    def process(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """送入一帧 int16 音频；命中唤醒词时返回事件，否则返回 None。"""
        self._ring.append(frame)

        speech = self.vad.is_speech(frame)
        if speech:
            self._hangover_left = self.hangover_frames
        elif self._hangover_left > 0:
            self._hangover_left -= 1
        now_active = speech or self._hangover_left > 0

        score: Optional[float] = None
        if now_active and not self._active:
            # 上升沿：清空模型缓冲，回灌前导帧补足上下文
            if hasattr(self.oww, "reset"):
                self.oww.reset()
            for f in list(self._ring):
                s = self._predict(f)
                score = s if score is None else max(score, s)
        elif now_active:
            score = self._predict(frame)
        else:
            # 静默：不跑唤醒词模型（省电）。刚从活跃转静默时清一次缓冲。
            if self._active and hasattr(self.oww, "reset"):
                self.oww.reset()

        self._active = now_active
        if score is not None:
            self.last_active_score = score

        if score is not None and score >= self.threshold:
            now = time.monotonic()
            if now - self._last_trigger >= self.cooldown:
                self._last_trigger = now
                event = DetectionEvent(self.model_name, score, time.time())
                logger.info("🔔 唤醒命中 score=%.3f", score)
                return event
        return None

    def last_score(self) -> float:
        """供前台调参用：返回当前帧的预测分（仅活跃时有意义）。"""
        return self._predict(self._ring[-1]) if self._ring else 0.0
