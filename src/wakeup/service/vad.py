"""轻量人声活动检测（VAD），用于功耗门控。

思路：唤醒词模型虽小，但持续推理仍耗电。先用一个**极廉价**的 VAD 判断
"当前有没有人声"，只有有人声时才唤起唤醒词模型 —— 静默时几乎零算力。

后端：
    webrtc  —— Google WebRTC VAD，准、快、便宜（需 webrtcvad / webrtcvad-wheels）
    energy  —— 纯能量阈值，零依赖兜底
    none    —— 不门控，始终认为有人声（最高功耗，调试用）
    auto    —— 优先 webrtc，装不上则退回 energy
"""

from __future__ import annotations

import numpy as np

from .._logging import get_logger
from ..config import Config

logger = get_logger(__name__)


class VAD:
    def __init__(self, cfg: Config):
        self.sample_rate = cfg.service.sample_rate
        self.energy_threshold = cfg.service.energy_threshold
        self.backend = self._select_backend(cfg.service.vad_backend,
                                             cfg.service.vad_aggressiveness)

    def _select_backend(self, name: str, aggressiveness: int):
        if name in ("auto", "webrtc"):
            try:
                import webrtcvad

                vad = webrtcvad.Vad(aggressiveness)
                logger.info("VAD 后端: webrtc (aggressiveness=%d)", aggressiveness)
                return ("webrtc", vad)
            except Exception as exc:
                if name == "webrtc":
                    raise
                logger.info("webrtcvad 不可用(%s)，回退到 energy 后端", exc)
        if name == "none":
            logger.info("VAD 后端: none（不门控）")
            return ("none", None)
        logger.info("VAD 后端: energy (threshold=%.4f)", self.energy_threshold)
        return ("energy", None)

    def is_speech(self, frame: np.ndarray) -> bool:
        """判断一帧 int16 音频是否包含人声。"""
        kind, obj = self.backend
        if kind == "none":
            return True
        if kind == "energy":
            return self._energy_is_speech(frame)
        return self._webrtc_is_speech(frame, obj)

    # -- energy 后端 --
    def _energy_is_speech(self, frame: np.ndarray) -> bool:
        x = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x)))
        return rms >= self.energy_threshold

    # -- webrtc 后端 --
    def _webrtc_is_speech(self, frame: np.ndarray, vad) -> bool:
        # WebRTC VAD 只接受 10/20/30ms 帧。把 80ms 帧切成 20ms 子帧，
        # 任一子帧判为人声即认为有人声。
        sub = int(self.sample_rate * 0.02)  # 20ms
        pcm = frame.astype(np.int16).tobytes()
        step = sub * 2  # int16 -> 2 bytes
        for off in range(0, len(pcm) - step + 1, step):
            chunk = pcm[off : off + step]
            try:
                if vad.is_speech(chunk, self.sample_rate):
                    return True
            except Exception:
                return self._energy_is_speech(frame)
        return False
