"""麦克风采集：基于 sounddevice，按固定帧长输出 int16 单声道音频。

作为上下文管理器使用，``with`` 退出时自动关闭音频流、释放麦克风（省电关键）。
"""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np

from .._logging import get_logger
from ..config import Config

logger = get_logger(__name__)


class AudioInput:
    def __init__(self, cfg: Config, device: int | str | None = None):
        self.sample_rate = cfg.service.sample_rate
        self.frame_samples = cfg.service.frame_samples
        self.device = device
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            logger.debug("音频流状态: %s", status)
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # 处理跟不上时丢弃最旧帧，避免无限堆积造成延迟
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Empty:
                pass

    def start(self) -> "AudioInput":
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("麦克风已开启 (%d Hz, 帧长 %d)", self.sample_rate, self.frame_samples)
        return self

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """取一帧 int16 音频；超时返回 None。"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("麦克风已关闭")

    def __enter__(self) -> "AudioInput":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def list_devices() -> str:
    """返回可读的音频设备列表，便于排查麦克风。"""
    import sounddevice as sd

    return str(sd.query_devices())
