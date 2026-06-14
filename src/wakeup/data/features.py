"""用 openWakeWord 的特征提取器把正样本 wav 转成 embedding 特征。

注意：openWakeWord 的特征提取器（melspectrogram + Google speech embedding）
是**语言无关**的，对中文音频同样适用 —— 这是中文唤醒词不损失精度的根本原因。
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
from tqdm import tqdm

from .._logging import get_logger
from ..config import Config

logger = get_logger(__name__)


def _load_clip(path: str, target_sr: int, target_len: int) -> np.ndarray | None:
    """读取单个 wav，转单声道 / 重采样 / 定长 / int16。失败返回 None。"""
    try:
        data, sr = sf.read(path)
    except Exception as exc:
        logger.debug("读取失败 %s: %s", path, exc)
        return None

    if data.ndim > 1:  # 立体声 -> 单声道
        data = np.mean(data, axis=1)

    if sr != target_sr:
        n = round(len(data) * float(target_sr) / sr)
        data = scipy.signal.resample(data, n)

    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)), "constant")
    else:
        data = data[:target_len]

    data = np.clip(data, -1.0, 1.0)
    return (data * 32767).astype(np.int16)


def extract_positive_features(cfg: Config, *, batch_size: int = 50) -> np.ndarray:
    """提取正样本特征并缓存到 ``positive_features.npy``。"""
    from openwakeword.utils import AudioFeatures

    fs = cfg.fs
    folder = fs.positive_dir
    target_sr = cfg.data.sample_rate
    target_len = target_sr * cfg.data.clip_seconds

    wav_files = sorted(glob.glob(str(Path(folder) / "*.wav")))
    logger.info("找到 %d 个音频文件，开始提取特征...", len(wav_files))
    if not wav_files:
        raise RuntimeError(f"目录中没有 wav 文件: {folder}")

    extractor = AudioFeatures()
    features_list = []

    for i in tqdm(range(0, len(wav_files), batch_size), desc="特征提取"):
        batch = wav_files[i : i + batch_size]
        clips = [c for c in (_load_clip(w, target_sr, target_len) for w in batch)
                 if c is not None]
        if not clips:
            continue
        try:
            embeddings = extractor.embed_clips(np.array(clips))
            features_list.extend(embeddings)
        except Exception as exc:
            logger.warning("批次特征提取失败: %s", exc)

    features = np.array(features_list)
    logger.info("正样本特征提取完毕，形状: %s", features.shape)

    np.save(fs.positive_features, features)
    return features
