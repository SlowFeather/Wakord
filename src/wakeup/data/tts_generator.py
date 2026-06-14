"""用 sherpa-onnx 的中文 VITS 模型批量合成「小元」正样本。

通过随机说话人 id（0~173）与随机语速制造多说话人、多语速的变体，
提升训练泛化能力 —— 这正是让中文唤醒词「不损失精度」的关键之一。
"""

from __future__ import annotations

import random
import tarfile

import soundfile as sf
from tqdm import tqdm

from .._logging import get_logger
from ..config import Config
from ._download import download

logger = get_logger(__name__)


def _ensure_tts_model(cfg: Config):
    """下载并解压 vits-zh-aishell3，返回 (onnx, lexicon, tokens) 路径。"""
    fs = cfg.fs
    tts_dir = fs.tts_dir
    onnx = tts_dir / "vits-aishell3.onnx"

    if not onnx.exists():
        archive = download(cfg.data.tts_model_url, fs.tts_archive, desc="TTS 模型")
        logger.info("解压 TTS 模型...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(tts_dir.parent)

    lexicon = tts_dir / "lexicon.txt"
    tokens = tts_dir / "tokens.txt"
    missing = [p for p in (onnx, lexicon, tokens) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"TTS 模型文件缺失: {missing}")
    return onnx, lexicon, tokens


def _build_tts(cfg: Config):
    import sherpa_onnx

    onnx, lexicon, tokens = _ensure_tts_model(cfg)
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(onnx),
                lexicon=str(lexicon),
                tokens=str(tokens),
            ),
            provider="cpu",
            num_threads=2,
        )
    )
    return sherpa_onnx.OfflineTts(tts_config)


def generate_positive_samples(cfg: Config, *, force: bool = False) -> int:
    """合成正样本 wav，返回成功生成的数量。

    ``force=False`` 时若目标数量已存在则跳过。
    """
    fs = cfg.fs
    fs.ensure_dirs()
    out_dir = fs.positive_dir

    existing = list(out_dir.glob("*.wav"))
    if not force and len(existing) >= cfg.data.num_samples:
        logger.info("已有 %d 个正样本，跳过生成（--force 可强制重生成）", len(existing))
        return len(existing)

    logger.info("初始化中文 TTS（sherpa-onnx vits-zh-aishell3）...")
    tts = _build_tts(cfg)

    word = cfg.data.target_word
    n = cfg.data.num_samples
    logger.info("开始合成 %d 个「%s」样本...", n, word)

    ok = 0
    for i in tqdm(range(n), desc="TTS 合成"):
        sid = random.randint(cfg.data.speaker_id_min, cfg.data.speaker_id_max)
        speed = random.uniform(cfg.data.speed_min, cfg.data.speed_max)
        path = out_dir / f"sample_{i:04d}.wav"
        try:
            audio = tts.generate(word, sid=sid, speed=speed)
            sf.write(str(path), audio.samples, audio.sample_rate)
            ok += 1
        except Exception as exc:  # 单条失败不影响整体
            logger.debug("第 %d 条合成失败: %s", i, exc)

    logger.info("正样本生成完成：%d 个文件", ok)
    return ok
