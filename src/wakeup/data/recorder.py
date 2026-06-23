"""用麦克风录制真实唤醒词样本，用于 few-shot 个性化。

TTS 合成的正样本和你真实嗓音/麦克风存在域差距，纯 TTS 召回会遇到瓶颈。录几十条
你自己的「小元」混入训练，能显著提升对你本人的召回率（业界自定义唤醒词通用做法）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._logging import get_logger
from ..config import Config

logger = get_logger(__name__)


def record_samples(cfg: Config, count: int = 30, seconds: float = 1.5,
                   out_dir: str | Path | None = None) -> int:
    """交互式录制 ``count`` 条唤醒词样本，返回成功录制数。"""
    import sounddevice as sd
    import soundfile as sf

    sr = cfg.data.sample_rate
    out = Path(out_dir) if out_dir else cfg.fs.user_positive_dir
    out.mkdir(parents=True, exist_ok=True)
    existing = len(list(out.glob("*.wav")))

    word = cfg.data.target_word
    print(f"\n将录制 {count} 条「{word}」到 {out}（已有 {existing} 条）")
    print("提示：每条按 Enter 开始，听到开始后立刻清晰说一次「" + word + "」。")
    print("建议变换语速/语气/与麦克风距离，多样化更利于泛化。Ctrl+C 结束。\n")

    saved = 0
    for i in range(count):
        try:
            input(f"[{i + 1}/{count}] 按 Enter 开始录音（{seconds}s）...")
        except (KeyboardInterrupt, EOFError):
            print("\n已中止。")
            break

        print(f"   🔴 录音中（{seconds}s）——现在清晰说「{word}」！")
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
        sd.wait()
        mono = audio[:, 0]

        path = out / f"user_{existing + saved:04d}.wav"
        sf.write(str(path), mono, sr)

        rms = float(np.sqrt(np.mean((mono / 32768.0) ** 2)))
        warn = "  ⚠️ 声音偏轻，靠近麦克风些" if rms < 0.01 else ""
        print(f"   ✔ {path.name}  RMS={rms:.3f}{warn}")
        saved += 1

    total = len(list(out.glob("*.wav")))
    print(f"\n完成，本次 {saved} 条，目录共 {total} 条。重新训练：wakeup train\n")
    return saved
