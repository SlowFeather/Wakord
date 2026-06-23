"""用 Microsoft Edge TTS 合成多音色「小元」，扩充 TTS 正样本多样性。

sherpa-onnx 的 aishell3 虽有 174 个说话人，但共用同一套声码器/通道，音色多样性有限。
Edge TTS 提供几十种风格各异的中文神经音色（普通话/台普/方言、男女老少），跨引擎多样性
能明显改善对真实嗓音的泛化。免费、在线、直接出 mp3（soundfile 可直接解码）。

注意：Edge TTS 需联网（走微软在线端点）；弱网下本模块带重试 + 断点跳过（已存在的不重生成）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .._logging import get_logger
from ..config import Config

logger = get_logger(__name__)

# 覆盖普通话(陆/台)与若干带口音音色，男女老少尽量分散
DEFAULT_VOICES = [
    "zh-CN-XiaoxiaoNeural",        # 女
    "zh-CN-XiaoyiNeural",          # 女
    "zh-CN-YunxiNeural",           # 男
    "zh-CN-YunjianNeural",         # 男
    "zh-CN-YunyangNeural",         # 男(播音)
    "zh-CN-YunxiaNeural",          # 男(偏年轻)
    "zh-TW-HsiaoChenNeural",       # 台普 女
    "zh-TW-HsiaoYuNeural",         # 台普 女
    "zh-TW-YunJheNeural",          # 台普 男
    "zh-CN-liaoning-XiaobeiNeural",  # 东北口音 女
    "zh-CN-shaanxi-XiaoniNeural",    # 陕西口音 女
]
DEFAULT_RATES = ["-20%", "-10%", "+0%", "+10%", "+25%"]
DEFAULT_PITCHES = ["-15Hz", "+0Hz", "+15Hz"]


async def _save_one(word: str, voice: str, rate: str, pitch: str,
                    path: Path, retries: int = 3) -> bool:
    import edge_tts

    for attempt in range(1, retries + 1):
        try:
            comm = edge_tts.Communicate(word, voice, rate=rate, pitch=pitch)
            await comm.save(str(path))
            if path.exists() and path.stat().st_size > 0:
                return True
        except Exception as exc:
            logger.debug("生成失败(%s, %s, %s) 第%d次: %s", voice, rate, pitch, attempt, exc)
            await asyncio.sleep(min(2 ** attempt, 8))
    return False


async def _run(word, out: Path, combos, retries):
    ok = 0
    for idx, (voice, rate, pitch) in enumerate(combos):
        path = out / f"edge_{idx:04d}.mp3"
        if path.exists() and path.stat().st_size > 0:
            ok += 1
            continue
        if await _save_one(word, voice, rate, pitch, path, retries):
            ok += 1
            if ok % 20 == 0:
                logger.info("已生成 %d/%d", ok, len(combos))
    return ok


def generate_edge_samples(cfg: Config, *, count: int | None = None,
                          out_dir: str | Path | None = None, retries: int = 3) -> int:
    """合成多音色「小元」到 ``tts_edge_dir``，返回成功条数。"""
    word = cfg.data.target_word
    out = Path(out_dir) if out_dir else cfg.fs.tts_edge_dir
    out.mkdir(parents=True, exist_ok=True)

    combos = [(v, r, p) for v in DEFAULT_VOICES for r in DEFAULT_RATES for p in DEFAULT_PITCHES]
    if count is not None:
        # 用固定步长抽样，保证截断后仍覆盖多种音色
        import random

        random.Random(cfg.train.seed).shuffle(combos)
        combos = combos[:count]

    logger.info("Edge TTS 合成「%s」：%d 个 voice×rate×pitch 组合 -> %s",
                word, len(combos), out)
    ok = asyncio.run(_run(word, out, combos, retries))
    logger.info("Edge TTS 完成：成功 %d/%d 条（目录共 %d 条）",
                ok, len(combos), len(list(out.glob('*.mp3'))))
    return ok
