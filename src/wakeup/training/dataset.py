"""把正/负样本特征对齐成统一形状并切分训练/验证集。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._logging import get_logger
from ..config import Config

logger = get_logger(__name__)


def fix_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    """把 (B, T, D) 特征的时间维截断/0 填充对齐到 ``target_frames``。"""
    _, t, _ = features.shape
    if t == target_frames:
        return features
    if t > target_frames:
        return features[:, :target_frames, :]
    pad = target_frames - t
    return np.pad(features, ((0, 0), (0, pad), (0, 0)), mode="constant")


def negative_windows(
    stream: np.ndarray, n_windows: int, target_frames: int, rng: np.random.Generator
) -> np.ndarray:
    """从负样本帧流里切出 ``n_windows`` 个「连续 target_frames 帧」的时间窗。

    关键修复：负样本必须和正样本一样具备真实的时间结构。早期实现把**单帧复制
    16 次**，使每个负样本是 16 个完全相同的帧；模型于是学成「帧间有没有变化」
    而非「是不是小元」——对任何真实音频（一定有帧间变化）都打满分。这里改为从
    背景特征流里随机取连续 16 帧的窗口。
    """
    stream = np.asarray(stream)
    if stream.ndim == 3:  # 已是窗口形式，直接抽样
        idx = rng.choice(len(stream), min(n_windows, len(stream)), replace=False)
        return fix_frames(stream[idx], target_frames)

    n = len(stream)
    max_start = n - target_frames
    if max_start < 1:
        raise ValueError(f"负样本帧数不足({n})，无法构成 {target_frames} 帧时间窗")
    starts = rng.integers(0, max_start + 1, size=n_windows)
    return np.stack([stream[s : s + target_frames] for s in starts])


def split_indices(n_items: int, val_split: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint train/validation indices with at least one item per side when possible."""
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
    if n_items < 2:
        raise ValueError("need at least two source items to build a validation split")
    perm = rng.permutation(n_items)
    n_val = int(round(n_items * val_split))
    n_val = min(max(1, n_val), n_items - 1)
    return perm[n_val:], perm[:n_val]


def split_negative_stream(
    stream: np.ndarray, val_split: float, target_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a continuous negative stream into non-overlapping train/validation regions."""
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
    stream = np.asarray(stream)
    if stream.ndim == 3:
        split = int(round(len(stream) * (1 - val_split)))
        split = min(max(1, split), len(stream) - 1)
        return stream[:split], stream[split:]

    min_len = target_frames + 1
    if len(stream) < min_len * 2:
        raise ValueError(
            f"negative feature stream is too short ({len(stream)} frames) for a leak-free split"
        )
    split = int(round(len(stream) * (1 - val_split)))
    split = min(max(min_len, split), len(stream) - min_len)
    return stream[:split], stream[split:]


def word_blocks(
    pos_feats: np.ndarray, target_frames: int, tol: float = 1e-3, min_len: int = 3
) -> list[np.ndarray]:
    """从补零的正样本特征里截出「词帧」块，去掉尾部恒定的静音帧。

    TTS 片段被补零到 3 秒，尾部是一段**完全相同**的静音 embedding；据此定位词的
    结束位置，只保留词帧。这样后续才能把词放到窗口任意位置、用真实背景填充其余帧。
    """
    blocks = []
    for f in pos_feats:  # (T, D)
        diff = np.abs(f - f[-1]).max(axis=1)  # 每帧与静音尾帧的差异
        active = np.flatnonzero(diff > tol)
        k = int(active[-1]) + 1 if active.size else len(f)
        k = max(min_len, min(k, target_frames))
        blocks.append(f[:k].astype(np.float32))
    return blocks


def augment_positives(
    tts_blocks: list[np.ndarray],
    neg_stream: np.ndarray,
    n_target: int,
    target_frames: int,
    rng: np.random.Generator,
    noise: float = 0.0,
    real_blocks: list[np.ndarray] | None = None,
    real_prob: float = 0.5,
) -> np.ndarray:
    """把词帧块放到 16 帧窗口的**随机位置**，其余帧用**真实背景帧**填充。

    动机：实时是滑动窗推理，词会出现在窗口任意位置。让模型见到「词出现在任意位置 +
    真实背景」可显著改善对齐鲁棒性与召回，并消除补零伪影。

    若提供 ``real_blocks``（用户真实录音的词帧），按 ``real_prob`` 的概率优先采样它们，
    哪怕只有几十条也能在增强后获得足够权重 —— 这是跨越 TTS→真人嗓音 域差距的关键。
    """
    if not tts_blocks and not real_blocks:
        raise ValueError("need at least one positive word block")
    d = neg_stream.shape[1]
    max_start = len(neg_stream) - target_frames
    if max_start < 0:
        raise ValueError(
            f"negative background is too short ({len(neg_stream)} frames) for {target_frames}-frame windows"
        )
    has_real = bool(real_blocks)
    out = np.empty((n_target, target_frames, d), dtype=np.float32)
    for i in range(n_target):
        use_real = has_real and (not tts_blocks or rng.random() < real_prob)
        if use_real:
            block = real_blocks[rng.integers(len(real_blocks))]
        else:
            block = tts_blocks[rng.integers(len(tts_blocks))]
        k = len(block)
        s = int(rng.integers(0, max_start + 1))
        win = neg_stream[s : s + target_frames].astype(np.float32).copy()
        off = int(rng.integers(0, target_frames - k + 1))
        win[off : off + k] = block  # 用词帧覆盖背景窗口的一段
        if noise > 0:
            win = win + rng.normal(0.0, noise, win.shape).astype(np.float32)
        out[i] = win
    return out


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
    user_features: np.ndarray | None = None,
) -> Dataset:
    tf = cfg.train
    rng = np.random.default_rng(tf.seed)
    neg_stream = np.asarray(negative_features)

    tts_blocks = word_blocks(positive_features, tf.target_frames)
    lens = np.array([len(b) for b in tts_blocks])
    logger.info("TTS word blocks: count=%d length=%d..%d mean=%.1f",
                len(tts_blocks), lens.min(), lens.max(), lens.mean())

    real_blocks = None
    if user_features is not None and len(user_features):
        real_blocks = word_blocks(user_features, tf.target_frames)
        logger.info("User recording blocks: count=%d real_sample_prob=%.0f%%",
                    len(real_blocks), tf.real_sample_prob * 100)

    tts_train_idx, tts_val_idx = split_indices(len(tts_blocks), tf.val_split, rng)
    tts_train = [tts_blocks[i] for i in tts_train_idx]
    tts_val = [tts_blocks[i] for i in tts_val_idx]

    real_train = real_val = None
    if real_blocks is not None:
        if len(real_blocks) >= 2:
            real_train_idx, real_val_idx = split_indices(len(real_blocks), tf.val_split, rng)
            real_train = [real_blocks[i] for i in real_train_idx]
            real_val = [real_blocks[i] for i in real_val_idx]
        else:
            logger.warning("Fewer than 2 user recordings; validation will skip real blocks to avoid leakage")
            real_train = real_blocks

    neg_train_stream, neg_val_stream = split_negative_stream(
        neg_stream, tf.val_split, tf.target_frames
    )

    n_pos_val = int(round(tf.positive_target * tf.val_split))
    n_pos_val = min(max(1, n_pos_val), tf.positive_target - 1)
    n_pos_train = tf.positive_target - n_pos_val

    pos_train = augment_positives(
        tts_train, neg_train_stream, n_pos_train, tf.target_frames, rng, tf.augment_noise,
        real_blocks=real_train, real_prob=tf.real_sample_prob,
    )
    pos_val = augment_positives(
        tts_val, neg_val_stream, n_pos_val, tf.target_frames, rng, 0.0,
        real_blocks=real_val, real_prob=tf.real_sample_prob,
    )
    logger.info("Positive augmentation: train=%s val=%s (split before augment)", pos_train.shape, pos_val.shape)

    neg_train = negative_windows(
        neg_train_stream, n_pos_train * tf.neg_pos_ratio, tf.target_frames, rng
    )
    neg_val = negative_windows(
        neg_val_stream, n_pos_val * tf.neg_pos_ratio, tf.target_frames, rng
    )
    logger.info("Negative windows: train=%s val=%s (non-overlapping regions)", neg_train.shape, neg_val.shape)

    X_train = np.concatenate([pos_train, neg_train]).astype(np.float32)
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))]).astype(np.float32)
    X_val = np.concatenate([pos_val, neg_val]).astype(np.float32)
    y_val = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))]).astype(np.float32)

    train_perm = rng.permutation(len(X_train))
    val_perm = rng.permutation(len(X_val))
    X_train, y_train = X_train[train_perm], y_train[train_perm]
    X_val, y_val = X_val[val_perm], y_val[val_perm]

    logger.info("Train set: X=%s pos=%d neg=%d", X_train.shape, int(y_train.sum()), int((y_train == 0).sum()))
    logger.info("Validation set: X=%s pos=%d neg=%d", X_val.shape, int(y_val.sum()), int((y_val == 0).sum()))
    return Dataset(X_train, y_train, X_val, y_val)
