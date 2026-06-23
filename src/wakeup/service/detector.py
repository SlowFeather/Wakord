"""唤醒词检测器：把「openWakeWord 持续推理 + 阈值 + 冷却去重」封装成
一个 ``process(frame) -> Optional[DetectionEvent]`` 的状态机。

关键设计（两处踩坑修正）：
* **监听时每帧都跑唤醒词模型，让 openWakeWord 的流式缓冲常驻预热。**
  openWakeWord 需要约 2s 连续音频才能填满 16 帧 embedding 窗口；早期实现为省电在
  静默时跳过模型、人声起再 reset+回灌前导帧，结果短词到来时缓冲根本来不及填满，
  得分恒为 0、严重漏检。每帧推理很廉价（~1-2ms/帧），可靠性远比这点算力重要。
* **不做 VAD 触发门控。** openWakeWord 的分数比词晚约 1s 才到峰值（窗口要先填满词），
  而 VAD 在说词时就触发、结束得早；用 VAD 门控触发会把分数峰值挡在门外，实测漏检 >90%。
  改为纯按模型分 + 阈值判定；模型已在负样本上训练为判别器，误报靠阈值 + 冷却控制。
  （VAD 仍保留，仅供 ``wakeup listen --debug`` 诊断展示。）
* 命中后进入冷却期，防止一次唤醒重复上报。
* 真正的省电在服务层：``stop`` 时关闭麦克风、释放设备。
"""

from __future__ import annotations

import time
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

        # VAD 仅供 `wakeup listen --debug` 诊断展示；不参与触发判定（见模块说明）。
        self.vad = VAD(cfg)
        self.oww = self._load_model(cfg)
        # openWakeWord 用模型文件名（不含扩展名）作为输出键
        keys = list(self.oww.models.keys())
        self.model_name = svc.model_name if svc.model_name in keys else keys[0]
        logger.info("唤醒词模型已加载，输出键: %s", self.model_name)

        self._last_trigger = 0.0
        # 最近一帧的预测分，供前台调阈值时观察
        self.last_active_score = 0.0

    @staticmethod
    def _load_model(cfg: Config):
        from openwakeword.model import Model

        from ..data.oww_assets import ensure_feature_models

        model_path = cfg.fs.model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到唤醒词模型: {model_path}\n请先运行 `wakeup train` 完成训练。"
            )

        # 首次使用时下载 openWakeWord 的特征提取模型（melspectrogram + embedding）
        ensure_feature_models(cfg, include_vad=False)

        logger.info("加载唤醒词模型: %s", model_path)
        return Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    def reset(self) -> None:
        self.vad.reset()
        if hasattr(self.oww, "reset"):
            self.oww.reset()

    def _predict(self, frame: np.ndarray) -> float:
        scores = self.oww.predict(frame)
        return float(scores.get(self.model_name, 0.0))

    def process(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """送入一帧 int16 音频；命中唤醒词时返回事件，否则返回 None。

        每帧都跑模型保持缓冲预热，按模型分 + 阈值 + 冷却判定触发。不做 VAD 门控：
        openWakeWord 的分数比词晚 ~1s 才到峰值（embedding 窗口需先填满），VAD 在词时
        触发、分数峰值落在其后，门控会把绝大多数真实命中挡掉（实测漏检 >90%）。
        误报由阈值 + 冷却控制；模型本身已在负样本上训练为判别器。
        """
        score = self._predict(frame)
        self.last_active_score = score

        if score >= self.threshold:
            now = time.monotonic()
            if now - self._last_trigger >= self.cooldown:
                self._last_trigger = now
                event = DetectionEvent(self.model_name, score, time.time())
                logger.info("🔔 唤醒命中 score=%.3f", score)
                return event
        return None
