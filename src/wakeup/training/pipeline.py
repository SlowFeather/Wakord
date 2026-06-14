"""端到端训练流水线编排：数据 -> 特征 -> 训练 -> 导出。

每一步都可单独跳过，便于在已有中间产物时增量执行。
"""

from __future__ import annotations

import numpy as np

from .._logging import get_logger
from ..config import Config
from ..data.features import extract_positive_features
from ..data.negatives import download_negative_features
from ..data.tts_generator import generate_positive_samples
from .dataset import build_dataset
from .export import export_onnx, export_tensorflow
from .trainer import train_classifier

logger = get_logger(__name__)


def run_training(
    cfg: Config,
    *,
    skip_tts: bool = False,
    force_tts: bool = False,
    export_tf: bool = False,
    simplify: bool = True,
) -> float:
    """执行完整训练流程，返回最优验证准确率。"""
    fs = cfg.fs
    fs.ensure_dirs()

    logger.info("[1/5] 准备中文正样本")
    if not skip_tts:
        generate_positive_samples(cfg, force=force_tts)
    else:
        logger.info("跳过 TTS 合成（--skip-tts）")

    logger.info("[2/5] 准备负样本特征")
    negative_features = download_negative_features(cfg)

    logger.info("[3/5] 提取正样本特征")
    if fs.positive_features.exists() and not force_tts:
        positive_features = np.load(fs.positive_features)
        logger.info("复用已缓存的正样本特征: %s", positive_features.shape)
    else:
        positive_features = extract_positive_features(cfg)

    logger.info("[4/5] 训练分类器")
    data = build_dataset(cfg, positive_features, negative_features)
    model, best_acc = train_classifier(cfg, data)

    logger.info("[5/5] 导出模型")
    export_onnx(cfg, model, simplify=simplify)
    if export_tf:
        export_tensorflow(cfg)

    logger.info("训练流程完成 ✅  最佳验证准确率 = %.2f%%", best_acc * 100)
    return best_acc
