"""人声活动检测（VAD），用于功耗门控。

唤醒词模型（melspectrogram + embedding + 分类器）持续推理很耗电。先用一个
**廉价**的 VAD 判断"当前有没有人声"，只有有人声时才唤起唤醒词模型 —— 静默时
只跑 VAD，几乎零算力。

后端（按 auto 的优先级）：
    silero  —— Silero 神经网络 VAD，准、抗噪、能抓住轻声起音；openWakeWord 自带其 onnx。
              虽是模型但极小（LSTM，每帧 ~1ms），仍远比唤醒词流水线便宜。
    webrtc  —— Google WebRTC VAD，最廉价，低功耗首选（需 webrtcvad-wheels）。
    energy  —— 自适应能量阈值（自动跟踪本底噪声），零依赖兜底。
    none    —— 不门控，始终认为有人声（最高功耗，调试用）。
    auto    —— 依次尝试 silero → webrtc → energy。

所有后端按 80ms/帧（int16）连续喂入；silero 依赖跨帧的流式状态，reset() 会清零。
"""

from __future__ import annotations

import numpy as np

from .._logging import get_logger
from ..config import Config

logger = get_logger(__name__)


class _Silero:
    """Silero VAD 的薄封装，维护 LSTM 流式状态（仿 openWakeWord 的调用方式）。"""

    def __init__(self, model_path: str, sample_rate: int, n_threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = n_threads
        so.intra_op_num_threads = n_threads
        self.sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(sample_rate).astype(np.int64)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def prob(self, frame: np.ndarray, chunk: int = 640) -> float:
        """返回该帧的平均人声概率（把 80ms 帧切成 ~40ms 子块逐块送入）。"""
        preds = []
        for i in range(0, len(frame), chunk):
            sub = (frame[i : i + chunk] / 32767.0).astype(np.float32)
            if sub.size == 0:
                continue
            out, self._h, self._c = self.sess.run(
                None, {"input": sub[None, :], "h": self._h, "c": self._c, "sr": self._sr}
            )
            preds.append(float(out[0][0]))
        return float(np.mean(preds)) if preds else 0.0


class VAD:
    def __init__(self, cfg: Config):
        svc = cfg.service
        self.sample_rate = svc.sample_rate
        self.energy_threshold = svc.energy_threshold
        self.silero_threshold = svc.vad_silero_threshold
        self._aggr = svc.vad_aggressiveness
        self._noise_floor: float | None = None
        self._cfg = cfg
        self.kind, self._impl = self._select(svc.vad_backend)

    # -- 后端选择 --
    def _select(self, name: str):
        if name == "none":
            logger.info("VAD 后端: none（不门控）")
            return "none", None

        candidates = ["silero", "webrtc"] if name == "auto" else [name]
        for cand in candidates:
            impl = self._build(cand)
            if impl is not None or cand == "none":
                return cand, impl
            if name != "auto":  # 明确指定却建不起来：energy 兜底
                break

        logger.info("VAD 后端: energy（自适应, 基准阈值=%.4f）", self.energy_threshold)
        return "energy", None

    def _build(self, kind: str):
        if kind == "silero":
            try:
                from ..data.oww_assets import ensure_feature_models, oww_models_dir

                ensure_feature_models(self._cfg, include_vad=True)
                path = oww_models_dir(self._cfg) / "silero_vad.onnx"
                impl = _Silero(str(path), self.sample_rate)
                logger.info("VAD 后端: silero（神经网络, threshold=%.2f）", self.silero_threshold)
                return impl
            except Exception as exc:  # 缺 onnxruntime / 模型下载失败等
                logger.info("silero 不可用(%s)", exc)
                return None
        if kind == "webrtc":
            try:
                import webrtcvad

                vad = webrtcvad.Vad(self._aggr)
                logger.info("VAD 后端: webrtc (aggressiveness=%d)", self._aggr)
                return vad
            except Exception as exc:
                logger.info("webrtcvad 不可用(%s)", exc)
                return None
        return None  # energy / 未知 -> 交给兜底

    # -- 对外接口 --
    def reset(self) -> None:
        """清空流式状态（麦克风重开 / 监听重启时调用）。"""
        self._noise_floor = None
        if self.kind == "silero":
            self._impl.reset()

    def is_speech(self, frame: np.ndarray) -> bool:
        """判断一帧 int16 音频是否包含人声。"""
        if self.kind == "none":
            return True
        if self.kind == "silero":
            return self._impl.prob(frame) >= self.silero_threshold
        if self.kind == "webrtc":
            return self._webrtc_is_speech(frame)
        return self._energy_is_speech(frame)

    # -- energy 后端（自适应本底噪声）--
    def _energy_is_speech(self, frame: np.ndarray) -> bool:
        x = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x)))
        if self._noise_floor is None:
            self._noise_floor = rms
        # 阈值取「配置基准」与「本底噪声 ×3」的较大者，自动适应安静/嘈杂环境
        threshold = max(self.energy_threshold, self._noise_floor * 3.0)
        speech = rms >= threshold
        if not speech:  # 仅在非人声时缓慢更新本底，避免被人声拉高
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        return speech

    # -- webrtc 后端 --
    def _webrtc_is_speech(self, frame: np.ndarray) -> bool:
        # WebRTC VAD 只接受 10/20/30ms 帧；把 80ms 帧切成 20ms 子帧，任一为人声即算有人声
        sub = int(self.sample_rate * 0.02)
        pcm = frame.astype(np.int16).tobytes()
        step = sub * 2  # int16 -> 2 bytes
        for off in range(0, len(pcm) - step + 1, step):
            try:
                if self._impl.is_speech(pcm[off : off + step], self.sample_rate):
                    return True
            except Exception:
                return self._energy_is_speech(frame)
        return False
