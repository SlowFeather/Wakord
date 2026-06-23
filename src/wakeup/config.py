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
    oww_models_dir: str = "artifacts/oww_models"


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
    tts_model_sha256: str | None = None
    negative_features_url: str = NEGATIVE_FEATURES_URL
    # 音频层增强：给 TTS 正样本叠加噪声/混响/增益/麦克风频响，缩小与真人/真麦的域差距。
    # 每条 TTS 样本额外生成 audio_augment_variants 个增强变体一并提特征。
    audio_augment: bool = True
    audio_augment_variants: int = 2


@dataclass
class TrainConfig:
    target_frames: int = 16
    embedding_dim: int = 96
    neg_pos_ratio: int = 8
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    val_split: float = 0.2
    seed: int = 42
    device: str = "auto"  # auto | cuda | cpu；auto 时有 GPU 自动用 GPU
    # 正样本增强：把"小元"词帧放到 16 帧窗口的随机位置 + 真实背景填充，
    # 生成这么多个增强正样本（提升对齐鲁棒性与召回）。
    positive_target: int = 4000
    augment_noise: float = 0.0  # 给增强样本叠加的高斯噪声标准差（0=关）
    # 若有用户真实录音（wakeup record），增强时按此概率优先采样真实词帧，
    # 用于跨越 TTS→真人嗓音 的域差距（few-shot 个性化）。
    real_sample_prob: float = 0.5


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
    vad_backend: str = "auto"  # auto | silero | webrtc | energy | none
    vad_aggressiveness: int = 2  # webrtc 0~3
    vad_silero_threshold: float = 0.5  # silero 人声概率阈值 0~1
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
        self.oww_models_dir = Path(cfg.oww_models_dir)

    @property
    def data_dir(self) -> Path:
        return self.base / "data"

    @property
    def positive_dir(self) -> Path:
        return self.data_dir / "positive_wavs"

    @property
    def user_positive_dir(self) -> Path:
        # 用户用麦克风录制的真实「小元」样本（few-shot 个性化）
        return self.data_dir / "user_positive"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def eval_positive_dir(self) -> Path:
        return self.eval_dir / "positive"

    @property
    def eval_negative_dir(self) -> Path:
        return self.eval_dir / "negative"

    @property
    def tts_edge_dir(self) -> Path:
        # Edge TTS 多音色合成的「小元」样本（扩充 TTS 多样性）
        return self.data_dir / "tts_edge"

    @property
    def negative_features(self) -> Path:
        return self.data_dir / "negative_features.npy"

    @property
    def positive_features(self) -> Path:
        return self.data_dir / "positive_features.npy"

    @property
    def edge_features(self) -> Path:
        # Edge TTS 多音色样本的特征缓存
        return self.data_dir / "edge_features.npy"

    @property
    def user_features(self) -> Path:
        # 用户真实录音的特征缓存
        return self.data_dir / "user_features.npy"

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
        for d in (
            self.data_dir,
            self.positive_dir,
            self.eval_positive_dir,
            self.eval_negative_dir,
            self.tts_dir.parent,
            self.model_dir,
            self.model_path.parent,
            self.oww_models_dir,
        ):
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


def _require_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_config(cfg: Config) -> None:
    """Validate user-facing configuration values that commonly break runtime behavior."""
    _require_positive("data.num_samples", cfg.data.num_samples)
    _require_positive("data.sample_rate", cfg.data.sample_rate)
    _require_positive("data.clip_seconds", cfg.data.clip_seconds)
    if cfg.data.speaker_id_min > cfg.data.speaker_id_max:
        raise ValueError("data.speaker_id_min must be <= data.speaker_id_max")
    if cfg.data.speed_min <= 0 or cfg.data.speed_max <= 0 or cfg.data.speed_min > cfg.data.speed_max:
        raise ValueError("data speed range must be positive and ordered")
    if cfg.data.audio_augment_variants < 0:
        raise ValueError("data.audio_augment_variants must be >= 0")

    _require_positive("train.target_frames", cfg.train.target_frames)
    _require_positive("train.embedding_dim", cfg.train.embedding_dim)
    _require_positive("train.neg_pos_ratio", cfg.train.neg_pos_ratio)
    _require_positive("train.epochs", cfg.train.epochs)
    _require_positive("train.batch_size", cfg.train.batch_size)
    _require_positive("train.learning_rate", cfg.train.learning_rate)
    _require_positive("train.positive_target", cfg.train.positive_target)
    if cfg.train.positive_target < 2:
        raise ValueError("train.positive_target must be at least 2")
    if not 0 < cfg.train.val_split < 1:
        raise ValueError(f"train.val_split must be between 0 and 1, got {cfg.train.val_split}")
    _require_range("train.real_sample_prob", cfg.train.real_sample_prob, 0.0, 1.0)
    if cfg.train.augment_noise < 0:
        raise ValueError("train.augment_noise must be >= 0")
    if cfg.train.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("train.device must be one of: auto, cuda, cpu")

    if not 1 <= cfg.service.port <= 65535:
        raise ValueError(f"service.port must be 1..65535, got {cfg.service.port}")
    _require_positive("service.sample_rate", cfg.service.sample_rate)
    _require_positive("service.frame_samples", cfg.service.frame_samples)
    _require_range("service.threshold", cfg.service.threshold, 0.0, 1.0)
    _require_positive("service.cooldown_seconds", cfg.service.cooldown_seconds)
    if cfg.service.vad_backend not in {"auto", "silero", "webrtc", "energy", "none"}:
        raise ValueError("service.vad_backend must be one of: auto, silero, webrtc, energy, none")
    if cfg.service.vad_aggressiveness not in {0, 1, 2, 3}:
        raise ValueError("service.vad_aggressiveness must be 0, 1, 2, or 3")
    _require_range("service.vad_silero_threshold", cfg.service.vad_silero_threshold, 0.0, 1.0)
    if cfg.service.energy_threshold < 0:
        raise ValueError("service.energy_threshold must be >= 0")
    if cfg.service.hangover_frames < 0 or cfg.service.preroll_frames < 0:
        raise ValueError("service.hangover_frames and service.preroll_frames must be >= 0")


def load_config(path: str | Path | None = None) -> Config:
    """加载配置；``path`` 为空时返回全默认配置。"""
    cfg = Config()
    if path is None:
        validate_config(cfg)
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _merge(cfg, data)
    validate_config(cfg)
    return cfg
