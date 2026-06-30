"""端到端训练流水线，拆成两个可独立运行的阶段：

* :func:`prepare_data` —— **慢**：合成/录制样本 → 下载负样本 → 提取并缓存所有特征。
  只在样本（TTS/Edge/录音）发生变化时才需要重跑。
* :func:`fit` —— **快**：直接读缓存特征 → 训练 → 导出 ONNX。调超参时反复跑这个即可。

:func:`run_training` 把两者串起来，保持 ``wakeup train`` 一条龙的旧行为。
"""

from __future__ import annotations

import numpy as np

from .._logging import get_logger
from ..config import Config
from ..data.features import (
    prepare_edge_features,
    prepare_positive_features,
    prepare_user_features,
)
from ..data.negatives import download_negative_features
from ..data.tts_generator import generate_positive_samples

logger = get_logger(__name__)


def prepare_data(
    cfg: Config,
    *,
    skip_tts: bool = False,
    force_tts: bool = False,
    gen_voices: bool = False,
    voices_count: int | None = None,
    force_features: bool = False,
) -> None:
    """阶段一（慢）：准备样本与负样本，并把所有正样本特征提取、缓存到 ``.npy``。

    特征有缓存时默认复用；新增了录音/音色后用 ``force_features=True`` 重建缓存。
    ``gen_voices=True`` 时额外用 Edge TTS 多音色扩充（需联网，失败非致命）。
    """
    fs = cfg.fs
    fs.ensure_dirs()

    logger.info("[1/3] 准备中文正样本")
    if not skip_tts:
        generate_positive_samples(cfg, force=force_tts)
    else:
        logger.info("跳过 TTS 合成（--skip-tts）")

    if gen_voices:
        from ..data.tts_edge import generate_edge_samples

        logger.info("用 Edge TTS 多音色扩充正样本（需联网）...")
        try:
            generate_edge_samples(cfg, count=voices_count)
        except Exception as exc:  # 联网失败不应中断准备流程
            logger.warning("Edge TTS 多音色合成失败（将仅用已有样本继续）: %s", exc)

    logger.info("[2/3] 下载负样本特征")
    download_negative_features(cfg)

    logger.info("[3/3] 提取并缓存正样本特征")
    force_feat = force_features or force_tts
    pos = prepare_positive_features(cfg, force=force_feat)
    logger.info("TTS 正样本特征就绪: %s", pos.shape)
    if prepare_edge_features(cfg, force=force_feat) is None:
        logger.info("无 Edge TTS 样本（wakeup gen-voices 可生成）")
    if prepare_user_features(cfg, force=force_feat) is None:
        logger.info("无用户真实录音（wakeup record 可录制）")

    logger.info("数据与特征准备完成 ✅  现在可反复运行 `wakeup fit` 训练调参")


def fit(cfg: Config, *, export_tf: bool = False, simplify: bool = True) -> float:
    """阶段二（快）：从缓存特征训练并导出 ONNX，返回推荐阈值下的验证 F1。"""
    try:
        from .dataset import build_dataset
        from .export import export_onnx, export_tensorflow
        from .trainer import train_classifier
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "缺少训练依赖 torch。请安装训练依赖后再运行 `wakeup fit` 或 `wakeup train`，"
                "例如：`uv sync --extra train` 或 `uv sync --extra all`。"
            ) from exc
        raise

    fs = cfg.fs
    fs.ensure_dirs()

    if not fs.positive_features.exists():
        raise FileNotFoundError(
            f"找不到正样本特征缓存: {fs.positive_features}\n"
            "请先运行 `wakeup prepare`（或一条龙的 `wakeup train`）。"
        )
    if not fs.negative_features.exists():
        raise FileNotFoundError(
            f"找不到负样本特征: {fs.negative_features}\n请先运行 `wakeup prepare`。"
        )

    logger.info("[1/3] 加载缓存特征")
    positive_features = np.load(fs.positive_features)
    logger.info("TTS 正样本特征: %s", positive_features.shape)
    if fs.edge_features.exists():
        edge_features = np.load(fs.edge_features)
        positive_features = np.concatenate([positive_features, edge_features])
        logger.info("合并 Edge TTS 后 TTS 正样本: %s", positive_features.shape)
    user_features = np.load(fs.user_features) if fs.user_features.exists() else None
    if user_features is None:
        logger.info("无用户真实录音特征，仅用 TTS 正样本")
    negative_features = np.load(fs.negative_features)
    logger.info("负样本特征: %s", negative_features.shape)

    logger.info("[2/3] 训练分类器")
    data = build_dataset(cfg, positive_features, negative_features, user_features)
    model, best_f1 = train_classifier(cfg, data)

    logger.info("[3/3] 导出模型")
    export_onnx(cfg, model, simplify=simplify)
    if export_tf:
        export_tensorflow(cfg)

    logger.info("训练流程完成 ✅  推荐阈值下验证 F1 = %.3f", best_f1)
    return best_f1


def run_training(
    cfg: Config,
    *,
    skip_tts: bool = False,
    force_tts: bool = False,
    gen_voices: bool = False,
    voices_count: int | None = None,
    force_features: bool = False,
    export_tf: bool = False,
    simplify: bool = True,
) -> float:
    """一条龙：``prepare_data`` + ``fit``（向后兼容 ``wakeup train``）。"""
    prepare_data(
        cfg,
        skip_tts=skip_tts,
        force_tts=force_tts,
        gen_voices=gen_voices,
        voices_count=voices_count,
        force_features=force_features,
    )
    return fit(cfg, export_tf=export_tf, simplify=simplify)
