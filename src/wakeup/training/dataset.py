"""把正/负样本特征对齐成统一形状并切分训练/验证集。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._logging import get_logger
from ..config import Config

logger = get_logger(__name__)


def fix_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    """强制把特征的时间维对齐到 ``target_frames``。

    支持输入:
        (B, D)        —— 缺时间维，复制扩充
        (B, T, D)     —— 截断或 0 填充到 target_frames
    """
    if features.ndim == 2:
        features = np.expand_dims(features, 1)  # (B, 1, D)
        return np.repeat(features, target_frames, axis=1)  # (B, T, D)

    _, t, _ = features.shape
    if t == target_frames:
        return features
    if t > target_frames:
        return features[:, :target_frames, :]
    pad = target_frames - t
    return np.pad(features, ((0, 0), (0, pad), (0, 0)), mode="constant")


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


def build_dataset(
    cfg: Config,
    positive_features: np.ndarray,
    negative_features: np.ndarray,
) -> Dataset:
    tf = cfg.train
    rng = np.random.default_rng(tf.seed)

    pos = fix_frames(positive_features, tf.target_frames)
    logger.info("正样本对齐后: %s", pos.shape)

    n_pos = len(pos)
    n_neg = min(len(negative_features), n_pos * tf.neg_pos_ratio)
    idx = rng.choice(len(negative_features), n_neg, replace=False)
    neg = fix_frames(negative_features[idx], tf.target_frames)
    logger.info("负样本对齐后: %s", neg.shape)

    X = np.concatenate([pos, neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32)
    logger.info("合并后训练集: X=%s  正:%d  负:%d", X.shape, n_pos, n_neg)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(len(X) * (1 - tf.val_split))
    return Dataset(X[:split], y[:split], X[split:], y[split:])
