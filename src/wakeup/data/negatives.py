"""负样本：直接下载 openWakeWord 官方预计算特征（音乐/噪音/日常对话）。

比下载几十 GB 原始音频快得多，且与训练用的特征提取器同源。
"""

from __future__ import annotations

import numpy as np

from .._logging import get_logger
from ..config import Config
from ._download import download

logger = get_logger(__name__)


def download_negative_features(cfg: Config) -> np.ndarray:
    fs = cfg.fs
    fs.ensure_dirs()
    path = download(
        cfg.data.negative_features_url, fs.negative_features, desc="负样本特征"
    )
    features = np.load(path)
    logger.info("加载负样本特征: %s", features.shape)
    return features
