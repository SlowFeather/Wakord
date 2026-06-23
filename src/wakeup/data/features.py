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


def _load_audio(path: str, target_sr: int) -> np.ndarray | None:
    """读取单个音频，转单声道 / 重采样为 float32（不定长、不转 int16）。失败返回 None。"""
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
    return data.astype(np.float32)


def _finalize_clip(audio: np.ndarray, target_len: int) -> np.ndarray:
    """定长（补零/截断）+ 裁剪 + 转 int16。"""
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), "constant")
    else:
        audio = audio[:target_len]
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def _load_clip(path: str, target_sr: int, target_len: int) -> np.ndarray | None:
    """读取单个 wav，转单声道 / 重采样 / 定长 / int16。失败返回 None。"""
    audio = _load_audio(path, target_sr)
    if audio is None:
        return None
    return _finalize_clip(audio, target_len)


def extract_features_from_dir(cfg: Config, folder: Path, *, batch_size: int = 50,
                              desc: str = "特征提取", augment_variants: int = 0) -> np.ndarray:
    """提取某目录下所有 wav 的 openWakeWord embedding 特征 (N, T, 96)。

    ``augment_variants>0`` 时，每条音频在原始版本之外再额外生成这么多个**音频层增强**
    变体（噪声/混响/增益/麦克风频响）一并提特征，用于缩小与真人/真麦的域差距。
    增强只施加在词音频上，补零得到的静音尾保持不变，兼容下游 word_blocks。
    """
    from openwakeword.utils import AudioFeatures

    from .augment import augment_audio
    from .oww_assets import ensure_feature_models

    # 首次运行时把 openWakeWord 的特征模型下到位，否则 AudioFeatures() 会因缺文件报错
    ensure_feature_models(cfg, include_vad=False)

    target_sr = cfg.data.sample_rate
    target_len = target_sr * cfg.data.clip_seconds
    rng = np.random.default_rng(cfg.train.seed)

    wav_files = sorted(
        f for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg")
        for f in glob.glob(str(Path(folder) / ext))
    )
    if augment_variants > 0:
        logger.info("%s：%s 下找到 %d 个音频（每条额外 %d 个增强变体）",
                    desc, folder, len(wav_files), augment_variants)
    else:
        logger.info("%s：%s 下找到 %d 个音频", desc, folder, len(wav_files))
    if not wav_files:
        raise RuntimeError(f"目录中没有音频文件: {folder}")

    extractor = AudioFeatures()
    features_list = []
    for i in tqdm(range(0, len(wav_files), batch_size), desc=desc):
        batch = wav_files[i : i + batch_size]
        clips: list[np.ndarray] = []
        for w in batch:
            raw = _load_audio(w, target_sr)
            if raw is None:
                continue
            clips.append(_finalize_clip(raw, target_len))
            for _ in range(augment_variants):
                clips.append(_finalize_clip(augment_audio(raw, target_sr, rng), target_len))
        if not clips:
            continue
        try:
            features_list.extend(extractor.embed_clips(np.array(clips)))
        except Exception as exc:
            logger.warning("批次特征提取失败: %s", exc)

    features = np.array(features_list)
    logger.info("%s 完毕，形状: %s", desc, features.shape)
    return features


def _augment_variants(cfg: Config) -> int:
    """TTS 正样本要生成的音频增强变体数（关或非正时为 0）。"""
    return cfg.data.audio_augment_variants if cfg.data.audio_augment else 0


def extract_positive_features(cfg: Config, *, batch_size: int = 50) -> np.ndarray:
    """提取 TTS 正样本特征并缓存到 ``positive_features.npy``。"""
    features = extract_features_from_dir(
        cfg, cfg.fs.positive_dir, batch_size=batch_size, desc="TTS 正样本特征",
        augment_variants=_augment_variants(cfg),
    )
    np.save(cfg.fs.positive_features, features)
    return features


def _has_audio(folder: Path) -> bool:
    return folder.exists() and any(
        folder.glob(ext) for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg")
    )


def extract_user_features(cfg: Config, *, batch_size: int = 50) -> np.ndarray | None:
    """提取用户真实录音特征（若有）并缓存到 ``user_features.npy``。无录音则返回 None。

    真实录音通常只有几十条，这里同样施加音频增强（距离/噪声/增益变体），用少量样本
    撑出更鲁棒的覆盖；配合 ``train.real_sample_prob`` 在增强时优先采样真实词块。
    """
    folder = cfg.fs.user_positive_dir
    if not _has_audio(folder):
        return None
    features = extract_features_from_dir(
        cfg, folder, batch_size=batch_size, desc="真实录音正样本特征",
        augment_variants=_augment_variants(cfg),
    )
    np.save(cfg.fs.user_features, features)
    return features


def extract_edge_features(cfg: Config, *, batch_size: int = 50) -> np.ndarray | None:
    """提取 Edge TTS 多音色样本特征（若有）并缓存到 ``edge_features.npy``。无则返回 None。"""
    folder = cfg.fs.tts_edge_dir
    if not _has_audio(folder):
        return None
    features = extract_features_from_dir(
        cfg, folder, batch_size=batch_size, desc="Edge TTS 正样本特征",
        augment_variants=_augment_variants(cfg),
    )
    np.save(cfg.fs.edge_features, features)
    return features


# --------------------------------------------------------------------------- #
# 「读缓存，没有再提取」包装：让特征提取只在样本变化时发生一次
# --------------------------------------------------------------------------- #
def _load_or_extract(cache: Path, extract, label: str, *, force: bool):
    if not force and cache.exists():
        feats = np.load(cache)
        logger.info("复用缓存%s特征: %s -> %s", label, feats.shape, cache)
        return feats
    return extract()


def prepare_positive_features(cfg: Config, *, force: bool = False, batch_size: int = 50) -> np.ndarray:
    """TTS 正样本特征：有缓存直接读，否则提取并缓存。"""
    return _load_or_extract(
        cfg.fs.positive_features,
        lambda: extract_positive_features(cfg, batch_size=batch_size),
        "TTS 正样本", force=force,
    )


def prepare_edge_features(cfg: Config, *, force: bool = False, batch_size: int = 50) -> np.ndarray | None:
    """Edge TTS 正样本特征：无样本返回 None；有缓存直接读，否则提取并缓存。"""
    if not _has_audio(cfg.fs.tts_edge_dir):
        return None
    return _load_or_extract(
        cfg.fs.edge_features,
        lambda: extract_edge_features(cfg, batch_size=batch_size),
        "Edge TTS 正样本", force=force,
    )


def prepare_user_features(cfg: Config, *, force: bool = False, batch_size: int = 50) -> np.ndarray | None:
    """用户真实录音特征：无录音返回 None；有缓存直接读，否则提取并缓存。"""
    if not _has_audio(cfg.fs.user_positive_dir):
        return None
    return _load_or_extract(
        cfg.fs.user_features,
        lambda: extract_user_features(cfg, batch_size=batch_size),
        "真实录音", force=force,
    )
