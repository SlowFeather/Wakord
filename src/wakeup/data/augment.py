"""音频层数据增强：给干净的 TTS「小元」叠加噪声 / 混响 / 增益 / 麦克风频响，
逼近真实麦克风+房间的录音条件，缩小「TTS→真人/真麦」的域差距。

为什么在音频层而不是 embedding 层做：openWakeWord 的 embedding 有时间感受野，
直接拼接不同来源的 embedding 帧会产生真实流式里不存在的拼缝。把增强施加在波形上、
再统一过特征提取器，得到的才是分布一致的"脏"特征。

只用 numpy/scipy，不引入额外依赖。增强只作用在"词"音频上（后续补零得到的静音尾
保持为常量），以兼容 ``dataset.word_blocks`` 靠静音尾帧定位词边界的逻辑。
"""

from __future__ import annotations

import numpy as np
import scipy.signal


def _add_noise(x: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """按目标信噪比叠加粉噪（白噪过一阶低通，更接近真实环境噪声谱）。"""
    sig_power = float(np.mean(x ** 2)) + 1e-9
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise = scipy.signal.lfilter([1.0], [1.0, -0.95], noise).astype(np.float32)
    noise_power = float(np.mean(noise ** 2)) + 1e-9
    target_power = sig_power / (10 ** (snr_db / 10.0))
    noise *= np.sqrt(target_power / noise_power)
    return (x + noise).astype(np.float32)


def _reverb(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """合成一个指数衰减的随机房间脉冲响应并卷积，模拟近场混响。"""
    rt = rng.uniform(0.05, 0.25)
    length = int(sr * rt)
    if length < 4:
        return x
    t = np.arange(length, dtype=np.float32)
    ir = rng.standard_normal(length).astype(np.float32) * np.exp(-3.0 * t / length)
    ir[0] = 1.0  # 直达声
    y = scipy.signal.fftconvolve(x, ir)[: len(x)].astype(np.float32)
    cur = np.sqrt(float(np.mean(y ** 2)) + 1e-9)
    ref = np.sqrt(float(np.mean(x ** 2)) + 1e-9)
    return (y * (ref / cur)).astype(np.float32)


def _mic_filter(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """随机带通，模拟不同麦克风/信道的频响（截掉极低频与高频）。"""
    low = rng.uniform(60.0, 200.0)
    high = rng.uniform(3000.0, 7000.0)
    sos = scipy.signal.butter(2, [low, high], btype="band", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, x).astype(np.float32)


def augment_audio(word: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """对一段词音频（float32, 取值约 [-1,1]）施加随机组合增强，返回增强后的音频。

    每种变换按概率独立施加，保证多样性；最后随机增益并裁剪到 [-1, 1]。
    """
    y = np.asarray(word, dtype=np.float32).copy()
    if rng.random() < 0.6:
        y = _mic_filter(y, sr, rng)
    if rng.random() < 0.5:
        y = _reverb(y, sr, rng)
    if rng.random() < 0.85:
        y = _add_noise(y, rng, snr_db=rng.uniform(8.0, 28.0))
    y *= rng.uniform(0.5, 1.15)
    return np.clip(y, -1.0, 1.0).astype(np.float32)
