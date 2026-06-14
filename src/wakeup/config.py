"""集中式配置。

设计目标：
* 所有可调参数都有合理默认值（dataclass 字段），开箱即用。
* 可用 YAML 文件按需覆盖（只写想改的字段）。
* 所有派生路径集中在 :class:`Paths`，避免散落的字符串拼接。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# 下载地址集中放这里，方便统一维护 / 镜像替换。
TTS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "tts-models/vits-zh-aishell3.tar.bz2"
)
NEGATIVE_FEATURES_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/"
    "resolve/main/validation_set_features.npy"
)


# --------------------------------------------------------------------------- #
# 各配置段
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    base_dir: str = "artifacts"
    # 部署用成品模型；训练结束后会把 onnx 复制到这里。
    model_path: str = "models/xiaoyuan.onnx"


@dataclass
class DataConfig:
    target_word: str = "小元"
    num_samples: int = 1000
    sample_rate: int = 16000
    clip_seconds: int = 3
    speaker_id_min: int = 0
    speaker_id_max: int = 173
    speed_min: float = 0.8
    speed_max: float = 1.4
    tts_model_url: str = TTS_MODEL_URL
    negative_features_url: str = NEGATIVE_FEATURES_URL


@dataclass
class TrainConfig:
    target_frames: int = 16
    embedding_dim: int = 96
    neg_pos_ratio: int = 20
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    val_split: float = 0.2
    seed: int = 42


@dataclass
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    model_name: str = "xiaoyuan"
    sample_rate: int = 16000
    frame_samples: int = 1280  # openWakeWord 单帧 80ms @ 16k
    threshold: float = 0.5
    cooldown_seconds: float = 2.0
    start_listening: bool = False
    vad_backend: str = "auto"  # auto | webrtc | energy | none
    vad_aggressiveness: int = 2
    energy_threshold: float = 0.012
    hangover_frames: int = 8
    preroll_frames: int = 16


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

    @property
    def fs(self) -> "Paths":
        return Paths(self.paths)


# --------------------------------------------------------------------------- #
# 派生路径
# --------------------------------------------------------------------------- #
class Paths:
    """根据 ``base_dir`` 推导出所有训练相关路径。"""

    def __init__(self, cfg: PathsConfig):
        self.base = Path(cfg.base_dir)
        self.model_path = Path(cfg.model_path)

    @property
    def data_dir(self) -> Path:
        return self.base / "data"

    @property
    def positive_dir(self) -> Path:
        return self.data_dir / "positive_wavs"

    @property
    def negative_features(self) -> Path:
        return self.data_dir / "negative_features.npy"

    @property
    def positive_features(self) -> Path:
        return self.data_dir / "positive_features.npy"

    @property
    def tts_dir(self) -> Path:
        # 解压后的 sherpa-onnx 中文 TTS 模型目录
        return self.base / "tts" / "vits-zh-aishell3"

    @property
    def tts_archive(self) -> Path:
        return self.base / "tts" / "vits-zh-aishell3.tar.bz2"

    @property
    def model_dir(self) -> Path:
        return self.base / "model_output"

    @property
    def best_ckpt(self) -> Path:
        return self.model_dir / "best.pth"

    @property
    def onnx_model(self) -> Path:
        return self.model_dir / "wakeword.onnx"

    @property
    def tf_model(self) -> Path:
        return self.model_dir / "tf_model"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.positive_dir, self.tts_dir.parent,
                  self.model_dir, self.model_path.parent):
            d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# YAML 加载与合并
# --------------------------------------------------------------------------- #
def _merge(dc: Any, overrides: dict) -> None:
    """把 dict 覆盖到 dataclass 实例上（就地、递归）。"""
    valid = {f.name: f for f in fields(dc)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"未知配置项: {key!r}（属于 {type(dc).__name__}）")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)


def load_config(path: str | Path | None = None) -> Config:
    """加载配置；``path`` 为空时返回全默认配置。"""
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _merge(cfg, data)
    return cfg
